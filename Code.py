# # # # Optimal classification model for BLE RSSI dataset # # # #

import itertools
import operator
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree as tr
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, \
    f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, \
    cross_val_score
from random import shuffle
import warnings, collections


class TrainTestSplit:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


class DataInfo:
    def __init__(self, raw_x, x, y, class_names, no_of_beacons,
                 y_test=None, predictions=None):
        if predictions is None:
            predictions = []
        self.raw_x = raw_x
        self.x = x
        self.y = y
        self.class_names = class_names
        self.no_of_beacons = no_of_beacons
        self.y_test = y_test
        self.predictions = predictions


class Statistics:
    def __init__(self):
        self.accuracy = None
        self.f1_score = None
        self.precision = None
        self.recall = None
        self.unpredicted = None


def print_menu():
    """
    Displays a list of classifiers to the user
    :return: None
    """
    print("\nMenu")
    print("`````")
    print("  1. Linear SVM")
    print("  2. RBF SVM")
    print("  3. Naive Bayes")
    print("  4. KNN")
    print("  5. LR")
    print("  6. Decision Tree Classifier")
    print("  q/Q. quit")


def get_choice():
    """
    Accepts a choice from the user for choice based execution
    :return: choice
    """
    print_menu()
    usr_input = input(">>> ")
    print()
    choice = usr_input.lower()
    return choice


def menu(di_obj):
    """
    Choices:
        1. Linear SVM
        2. RBF SVM
        3. Naive Bayes
        4. KNN
        5. LR
        6. Decision Tree Classifier
        q/Q. quit
    :return: None
    """
    choice = get_choice()
    predictions = []  # for function-wide access

    while choice != "q" and choice != "Q":
        tt_obj = []

        if choice == "1":  # Linear SVM
            print("Linear SVM :")
            print("````````````")
            print("For default .i.e. c =", 1)
            stat_obj, predictions = linear_svc(split_dataset(di_obj.x, di_obj.y, di_obj.class_names))
            print_stats(stat_obj)
            opt_c, opt_accuracy = optimum_c_linear_svc(di_obj)
            print("\nOptimal c:", opt_c)
            print("For optimum c:")
            tt_obj = split_dataset(di_obj.x, di_obj.y, di_obj.class_names)
            stat_obj, predictions = linear_svc(tt_obj, c=opt_c)
            print_stats(stat_obj)

        elif choice == "2":  # RBF SVM
            print("Radial Basis Function SVC :")
            print("```````````````````````````")
            print("For default .i.e. c =", 1, "and gamma = ", 0.2)
            stat_obj, predictions = rbf_svc(split_dataset(di_obj.x, di_obj.y, di_obj.class_names))
            print_stats(stat_obj)
            opt_gamma, opt_c, opt_accuracy = optimum_c_gamma_rbf_svc(di_obj)
            print("\nOptimal c:", opt_c)
            print("Optimal gamma:", opt_gamma)
            print("For optimum c and gamma:")
            tt_obj = split_dataset(di_obj.x, di_obj.y, di_obj.class_names)
            stat_obj, predictions = rbf_svc(tt_obj, c=opt_c, g=opt_gamma)
            print_stats(stat_obj)

        elif choice == "3":  # Naive bayes classifier
            print("Naive Bayes :")
            print("`````````````")
            tt_obj = split_dataset(di_obj.x, di_obj.y, di_obj.class_names)
            stat_obj, predictions = naive_bayes(tt_obj)
            print_stats(stat_obj)

        elif choice == "4":  # KNN classifier
            print("KNN :")
            print("`````")
            print("For default .i.e. k =", 3)
            stat_obj, predictions = knn(split_dataset(di_obj.x, di_obj.y, di_obj.class_names))
            print_stats(stat_obj)
            opt_k, opt_accuracy = optimum_k_knn(di_obj)
            print("\nOptimal k:", opt_k)
            print("For optimum k:")
            tt_obj = split_dataset(di_obj.x, di_obj.y, di_obj.class_names)
            stat_obj, predictions = knn(tt_obj, k=opt_k)
            print_stats(stat_obj)

        elif choice == "5":  # Logistic regression (LR)
            print("Logistic Regression :")
            print("`````````````````````")
            tt_obj = split_dataset(di_obj.x, di_obj.y, di_obj.class_names)
            stat_obj, predictions = lr(tt_obj)
            print_stats(stat_obj)

        elif choice == "6":  # Decisison tree classifiers
            print("Decision Tree Classifier :")
            print("``````````````````````````")
            tt_obj = split_dataset(di_obj.x, di_obj.y, di_obj.class_names)
            stat_obj, predictions = decision_tree_clf(tt_obj)
            print_stats(stat_obj)

        else:
            print("Invalid choice! Please choose again!\n")

        setattr(di_obj, "predictions", predictions)
        setattr(di_obj, "y_test", tt_obj.y_test)
        plot(di_obj)
        choice = get_choice()  # next choice

    print("Thank you! Goodbye!")


def classification_stats(tt_obj, predictions):
    """
    Assigns appropriate computed values to class variables
    :param tt_obj: Train-Test partition of the input data
    :param predictions: Expected class for the testing data
    :return: stat_obj - contains prediction statistics
    """
    stat_obj = Statistics()
    stat_obj.unpredicted = set(np.unique(tt_obj.y_test)) - set(np.unique(predictions))
    stat_obj.accuracy = accuracy_score(tt_obj.y_test, predictions)
    stat_obj.f1_score = f1_score(tt_obj.y_test, predictions, average='weighted', labels=np.unique(predictions))
    stat_obj.recall = recall_score(tt_obj.y_test, predictions, average='weighted', labels=np.unique(predictions))
    stat_obj.precision = precision_score(tt_obj.y_test, predictions, average='weighted', labels=np.unique(predictions))
    return stat_obj


def print_stats(stat_obj):
    """
    Prints out classification stats
    :param stat_obj: Prediction statistics
    :return: None
    """
    if stat_obj.unpredicted != set():
        print('Unpredicted:', stat_obj.unpredicted)
    print('Accuracy:', stat_obj.accuracy)
    print('F1 score:', stat_obj.f1_score)
    print('Recall:', stat_obj.recall)
    print('Precision:', stat_obj.precision)


def confusion_matrix(y_true, y_pred):
    """
    Generates the confusion matrix for the model
    :param y_true: true labels corresponding to the testing set
    :param y_pred: predicted labels for the above
    :return: 2D Confusion matrix
    """
    size = np.size(np.unique(y_true))
    conf_matrix = []
    for i in range(0, size):
        row = []
        for j in range(0, size):
            row.append(0)
        conf_matrix.append(row)

    for i in range(0, np.size(y_true)):
        true_label = int(y_true[i])
        pred_label = int(y_pred[i])
        conf_matrix[true_label - 1][pred_label - 1] += 1

    return np.array(conf_matrix)


def plot_conf_matrix(y_true, y_pred,
                     normalize=False,
                     title='Confusion Matrix',
                     cmap=plt.cm.Greys):
    """
    Plots the confusion matrix on a colormap if normalize=False and
         the row_normalized confusion matrix if normalize=True
    :param y_true: True classes for the testing data
    :param y_pred: Predicted class for the testing data
    :param normalize: if True ===> Row normalize
    :param title: Plot title (default: Confusion Matrix)
    :param cmap: color map spec (default: greys)
    :return: None
    """
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate normalized values (row-wise) if desired
    if normalize:
        norm_conf_matrix = []
        for i in range(0, np.size(conf_matrix[0])):
            row = conf_matrix[i]
            if row.sum() != 0:
                row = np.round(row.astype('float') / row.sum(), 2)  # float: entries in float
            else:
                for k in range(0, len(row)):
                    row[k] = 0.0
            norm_conf_matrix.append(row)
        conf_matrix = np.array(norm_conf_matrix)

    # Place Numbers as Text on Confusion Matrix Plot
    thresh = conf_matrix.max() / 2
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, conf_matrix[i, j],
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black",
                 fontsize=10)

    # Configure Confusion Matrix Plot Aesthetics
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=11)
    axis_label_pos = np.arange(len(conf_matrix[0]))
    axis_labels = list(np.arange(1, np.size(conf_matrix[0]) + 1, 1))
    plt.xticks(axis_label_pos, axis_labels)
    plt.yticks(axis_label_pos, axis_labels)
    plt.ylabel('True zone')
    plt.xlabel('Predicted zone')

    # Plot
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_zones(x, y, zones, no_of_beacons):
    """
    Plots RSSI values vs Beacon for each zone
    :param x: RSSI values
    :param y: Corresponding zones
    :param zones: zone names
    :param no_of_beacons: no of features for each data point
    :return:
    """
    beacons = []
    for i in range(0, no_of_beacons):
        beacons.append(i + 1)
    beacons = np.array(beacons)

    rssi_range = np.array([-40, -60, -80, -100, -120, -140, -160, -180, -200])

    for i in zones:
        for j in range(0, len(y)):
            if int(y[j]) == i:
                x_float = []
                for k in range(0, len(x[j])):
                    x_float.append(float(x[j][k]))
                plt.plot(beacons, x_float, 'rx', markersize=3.6)
        x_axis_labels = np.arange(1, no_of_beacons + 1, step=1)
        plt.xticks(x_axis_labels, beacons)
        plt.yticks(np.arange(min(rssi_range), max(rssi_range) + 1, 20))
        plt.xlabel('Beacon')
        plt.ylabel('RSSI')
        plt.title("Zone " + str(i))
        plt.grid()
        plt.show()


def plot(di_obj):
    """
    Plots:
        1. RSSI vs beacon for each zone
        2. The confusion matrix (for default params)
        3. The row-normalized confusion matrix
    :param di_obj: contains info regarding the input data
    :return: None
    """
    # RSSI-Beacon plots zones
    plot_zones(di_obj.raw_x, di_obj.y,
               di_obj.class_names, di_obj.no_of_beacons)

    # Plot confusion matrix (NOT row-normalized)
    # plot_conf_matrix(di_obj.y_test, di_obj.predictions,
    #                  classes=di_obj.class_names,
    #                  title='Confusion matrix')

    # Plot row-normalized confusion matrix
    plot_conf_matrix(di_obj.y_test, di_obj.predictions,
                     normalize=True,
                     title='Row-normalized confusion matrix')


def resample(tt_obj, class_names):
    """
    Reduce the class imbalance by over sampling the instances of the minority classes
    :param tt_obj: Train-Test partition
    :param class_names: zones
    :return:
    """
    data_size = len(tt_obj.y_train) + len(tt_obj.y_test)
    over_samples_x = []
    over_samples_y = []
    for i in class_names:
        class_i_x = []
        class_i_y = []
        for j in range(0, len(tt_obj.x_train)):
            if int(tt_obj.y_train[j]) == i:
                class_i_x.append(tt_obj.x_train[j])
                class_i_y.append(i)
        if 0 < len(class_i_y) < 0.2 * data_size:
            factor = 0.2 * data_size / len(class_i_x) - 1
            over_sampling_factor = int(math.ceil(factor))
            for j in range(0, over_sampling_factor):
                for k in range(0, len(class_i_y)):
                    over_samples_x.append(class_i_x[k])
                    over_samples_y.append(class_i_y[k])

    x_shuffle = []
    y_shuffle = []

    index_shuffle = np.arange(0, len(over_samples_x), 1)
    shuffle(index_shuffle)
    for i in index_shuffle:
        x_shuffle.append(over_samples_x[i])
        y_shuffle.append(over_samples_y[i])

    res_x_train = list(tt_obj.x_train)
    res_y_train = list(tt_obj.y_train)

    for i in range(0, len(x_shuffle)):
        res_x_train.append(x_shuffle[i])
        res_y_train.append(y_shuffle[i])

    res_tt_obj = TrainTestSplit(res_x_train, tt_obj.x_test,
                                res_y_train, tt_obj.y_test)

    return res_tt_obj


def split_dataset(x, y, class_names):
    """
    Partitions the input data into train and test set
    :param class_names: zones
    :param x: RSSI
    :param y: Corresponding Zones
    :return: tt_obj, Train-Test partition of x,y
    """
    y_test = []
    tt_obj = TrainTestSplit([], [], [], [])
    size = np.size(class_names)
    while np.size(np.unique(y_test)) != size:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        tt_obj = TrainTestSplit(x_train, x_test, y_train, y_test)
    res_tt_obj = resample(tt_obj, class_names)
    return res_tt_obj
    # return tt_obj

def linear_svc(tt_obj, c=1):
    """
    Support Vector Machine (Linear kernel)
    :param tt_obj: Train-Test partition
    :param c: Regularization parameter (default: c=1)
    :return: stat_obj - Classification statistics
             predictions - predicted classes for test set
    """
    svm_clf = SVC(kernel='linear', C=c)
    svm_clf.fit(tt_obj.x_train, tt_obj.y_train)
    predictions = svm_clf.predict(tt_obj.x_test)

    # model stats
    stat_obj = classification_stats(tt_obj, predictions)
    return stat_obj, predictions


def rbf_svc(tt_obj, c=1, g=0.2):
    """
    Support Vector Machine (Radial basis function kernel)
    :param tt_obj: Train-Test partition
    :param c: Regularization parameter (default: c=1)
    :param g: gamma parameter (degault: g=0.2)
    :return: stat_obj - Classification statistics
             predictions - predicted classes for test set
    """
    rbf_clf = SVC(kernel='rbf', C=c, gamma=g).fit(tt_obj.x_train, tt_obj.y_train)
    predictions = rbf_clf.predict(tt_obj.x_test)

    # model stats
    stat_obj = classification_stats(tt_obj, predictions)
    return stat_obj, predictions


def naive_bayes(tt_obj):
    # Naive bayes classifier
    """
    Naive Bayes classifier
    :param tt_obj: Train-Test partition
    :return: stat_obj - Classification statistics
             predictions - predicted classes for test set
    """
    nb_clf = GaussianNB()
    nb_clf.fit(tt_obj.x_train, tt_obj.y_train)
    predictions = nb_clf.predict(tt_obj.x_test)

    # model stats
    stat_obj = classification_stats(tt_obj, predictions)
    return stat_obj, predictions


def knn(tt_obj, k=3):
    """
    K-nearest neighbours classifier
    :param tt_obj: Train-Test partition
    :param k: # neighbours (default: k=1)
    :return: stat_obj - Classification statistics
             predictions - predicted classes for test set
    """
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(tt_obj.x_train, tt_obj.y_train)

    predictions = knn_clf.predict(tt_obj.x_test)

    # model stats
    stat_obj = classification_stats(tt_obj, predictions)
    return stat_obj, predictions


def lr(tt_obj):
    """
    Logistic regression
    :param tt_obj: Train-Test partition
    :return: stat_obj - Classification statistics
             predictions - predicted classes for test set
    """
    lr_clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    lr_clf.fit(tt_obj.x_train, tt_obj.y_train)

    predictions = lr_clf.predict(tt_obj.x_test)

    # model stats
    stat_obj = classification_stats(tt_obj, predictions)
    return stat_obj, predictions


def decision_tree_clf(tt_obj):
    """
    Decision tree classifier
    :param tt_obj: Train-Test partition
    :return: stat_obj - Classification statistics
             predictions - predicted classes for test set
    """
    dt_clf = tr.DecisionTreeClassifier()
    dt_clf.fit(tt_obj.x_train, tt_obj.y_train)

    predictions = dt_clf.predict(tt_obj.x_test)

    # model stats
    stat_obj = classification_stats(tt_obj, predictions)
    return stat_obj, predictions


def optimum_c_linear_svc(di_obj):
    """
    Analysis of the variation of the accuracy with C 
    :param di_obj: contains info regarding the input data
    :return: optimum_c and optimum_accuracy
    """
    plt.title("Variation of scores of linear SVC with C", fontsize=11)
    plt.xlabel('C')
    plt.ylabel('Mean score')
    x = np.arange(1, 21, 1)

    c_wise_scores = []
    for c in range(1, 21):
        svm_clf = SVC(kernel='linear', C=c)
        scores = cross_val_score(svm_clf, di_obj.x, di_obj.y, cv=5)
        c_wise_scores.append(scores.mean())

    y = c_wise_scores

    plt.plot(x, y, 'b-')
    plt.grid()
    plt.show()

    index, value = max(enumerate(y), key=operator.itemgetter(1))
    return (index + 1), value


def single_optimum_c_gamma_rbf_svc(di_obj):
    """
    Score matrix for single attempt of rnf_svc
    :param di_obj: contains info regarding the input data
    :return: 2D score_matrix (c vs gamma), c_values, gamma_values
    """
    score = []
    c_range = [-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]
    gamma_range = [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5]
    c_values = []
    gamma_values = []

    for g in gamma_range:
        row_score = []
        for c in c_range:
            tt_obj = split_dataset(di_obj.x, di_obj.y, di_obj.class_names)
            stat_obj, predictions = rbf_svc(tt_obj, 2 ** c, 2 ** g)
            c_values.append(np.round(2 ** c, 4))
            row_score.append(stat_obj.accuracy)
        gamma_values.append(np.round(2 ** g, 5))
        score.append(row_score)

    return score, c_values, gamma_values


def optimum_c_gamma_rbf_svc(di_obj, cmap=plt.cm.Greys):
    """
    Analysis of the variation of the accuracy with C and gamma by considering 100 attempts
    :param di_obj: contains info regarding the input data
    :param cmap: color map (default: Greys)
    :return: optimum_gamma, optimum_c, optimum_accuracy
    """
    title = "Variation of scores of rbf SVC with C and gamma"

    multi_attempt_sum = []
    for i in range(1, 12):
        row = []
        for j in range(1, 12):
            row.append(0)
        multi_attempt_sum.append(row)

    c_range = [-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]
    gamma_range = [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5]
    c_values = []
    gamma_values = []

    for i in range(0, 100):  # 100 attempts
        print(i)
        single_attempt_score, c_values, gamma_values = \
            single_optimum_c_gamma_rbf_svc(di_obj)
        for j in range(0, len(single_attempt_score)):
            for k in range(0, len(single_attempt_score)):
                multi_attempt_sum[j][k] += single_attempt_score[j][k]

    multi_attempt_average = [[entry / 100 for entry in row] for row in multi_attempt_sum]
    # print(multi_attempt_average)
    # Configuring the plot aesthetics
    plt.imshow(multi_attempt_average, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=11)
    x_axis_labels = np.arange(len(gamma_range))
    y_axis_labels = np.arange(len(c_range))
    plt.xticks(x_axis_labels, gamma_values, fontsize=6)
    plt.yticks(y_axis_labels, c_values, fontsize=7)
    plt.xlabel('gamma')
    plt.ylabel('C')

    # Plot
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # computation of the optimum
    x_index = 0
    y_index = 0
    value = np.argmax(multi_attempt_average)
    for i in range(0, len(multi_attempt_average)):
        for j in range(0, len(multi_attempt_average[0])):
            if multi_attempt_average[i][j] == value:
                x_index = i
                y_index = j
    return gamma_values[x_index], c_values[y_index], value


def optimum_k_knn(di_obj):
    """
    Analysis of the variation of the accuracy with k (# neighbours)
    :param di_obj: contains info regarding the input data
    :return: optimum_k, optimum_accuracy
    """
    plt.title("Variation of scores of KNN with k", fontsize=11)
    plt.xlabel('k')
    plt.ylabel('Mean score')
    x = np.arange(1, 11, 1)

    k_wise_scores = []
    for k in range(1, 11):
        knn_clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn_clf, di_obj.x, di_obj.y, cv=5)
        k_wise_scores.append(scores.mean())

    y = k_wise_scores

    plt.plot(x, y, 'b-')
    plt.grid()
    plt.show()

    index, value = max(enumerate(y), key=operator.itemgetter(1))
    return (index + 1), value


def main():
    warnings.filterwarnings("ignore")

    # Reading the data from file
    file = pd.read_csv(r'C:\Users\DELL\PycharmProjects\Classifier\Data_labeled.csv', sep=',', header=None)

    # Storing the data as a python array
    array = file.values

    # Ignoring the column labels
    array = array[1:len(array), :]

    # Splitting into RSSI values(X) and corresponding zones(Y)
    raw_x = array[:, 1:14]
    x = normalize(raw_x)
    y = array[:, 0]

    # for Data_labeled
    class_names = [1, 2, 3, 4, 5, 6, 7]

    # Beacon specifications
    no_of_beacons = len(array[0]) - 1

    # DataInfo object
    di_obj = DataInfo(raw_x, x, y, class_names, no_of_beacons)

    menu(di_obj)  # class menu() for a choice driven execution


if __name__ == "__main__":
    main()
