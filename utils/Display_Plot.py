import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix



# 定义一个ROC曲线绘制函数，sklearn画的太糊了
def plot_roc_curve(y_true, y_score, model_name:str, save_path:str=None, is_save:bool=False, save_name:str=None):

    """
    y_true: 实际标签
    y_score: 预测的概率
    model_name: 模型名称
    save_path：保存路径
    is_save: 是否保存图片，默认为False，不保存
    save_name: 保存图片的名称，默认为None
    """
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10,6), facecolor="white", dpi=300)
    plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='%s(AUC = %0.2f)' % (model_name, roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    if is_save and save_name is None:
        plt.savefig(save_path+"\\ROC of %s.png"%model_name)
    if is_save and save_name is not None:
        plt.savefig(save_path+"\\%s.png"%save_name)
    plt.show()



# 定义一个绘制混淆矩阵的函数，便于多次调用
def plot_confusion_matrix(y_true, y_pred, accuracy:float, save_path:str, is_save:bool=False, save_name:str=None):

    """
    y_true: 实际标签
    y_pred: 预测的类别
    save_path：保存路径
    is_save: 是否保存图片，默认为False，不保存
    save_name: 保存图片的名称，默认为None
    """

    clf_matrix = confusion_matrix(y_true, y_pred)
    print("混淆矩阵:\n", clf_matrix)
    plt.figure(figsize=(9,9), facecolor="white", dpi=300)
    sns.heatmap(clf_matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.title('Accuracy Score: %.2f'%accuracy, size = 15);
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    if is_save and save_name is not None:
        plt.savefig(save_path+"\\%s.png"%save_name)
    if is_save and save_name is None:
        plt.savefig(save_path+"\\Confusion Matrix.png")
    plt.show()



# 定义一个可视化特征重要性的函数便于多次调用
def plot_feature_importance(importances, feature_name, save_path:str, is_save:bool=False, save_name:str=None):
    
    """
    importances: 特征重要性
    feature_name: 特征名称，用于绘图
    save_path：保存路径
    is_save: 是否保存图片，默认为False，不保存
    save_name: 保存图片的名称，默认为None
    """

    plt.figure(figsize=(10,6), facecolor="white", dpi=300)
    plt.title('Feature Importances')
    weights = pd.Series(importances, index=feature_name)
    weights.sort_values()[-15:].plot(kind = 'barh')
    plt.xlabel('Relative Importance')
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    if is_save and save_name is not None:
        plt.savefig(save_path+"\\%s.png"%save_name)
    if is_save and save_name is None:
        plt.savefig(save_path+"\\Feature Importances.png")
    plt.show()

# TODO 数据的缩尾处理？  