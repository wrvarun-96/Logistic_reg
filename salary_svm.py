import pandas as pd
import numpy as np
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest,chi2
from pandas_profiling import ProfileReport
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier




salary_train=pd.read_csv("C:/Users/W R VARUN/Desktop/Data Science/SVM/SalaryData_Train.csv")
salary_test=pd.read_csv("C:/Users/W R VARUN/Desktop/Data Science/SVM/SalaryData_Test.csv")
#print(salary_train)
#print(salary_test)

"------------------------------------------------EDA-----------------------------------------------------------------"

#TO KNOW ALL ABOUT THE DATASET
profile=ProfileReport(salary_train,explorative=True)
#print(profile)



#DETECTING THE OUTLIERS FOR BOTH TRAIN AND TEST
salary_train.boxplot()
salary_test.boxplot()
plt.show()

#train dataset
Q1=salary_train.quantile(0.25)       
#print(Q1)
Q3=salary_train.quantile(0.75)
#print(Q3)
IQR=Q3-Q1
#print(IQR)
salary_train_outliers=salary_train[~((salary_train<(Q1-1.5*IQR)) | (salary_train>(Q3+1.5*IQR)))]

#test dataset
Q1=salary_test.quantile(0.25)       
#print(Q1)
Q3=salary_test.quantile(0.75)
#print(Q3)
IQR=Q3-Q1
#print(IQR)
salary_test_outliers=salary_test[~((salary_test<(Q1-1.5*IQR)) | (salary_test>(Q3+1.5*IQR)))]


#REPLACING NaN VALUES WITH MODE OF RESPETIVE COLUMNS HAVING NaN
for i in salary_train_outliers.columns:
    if salary_train_outliers[i].isnull().sum()>0:
        salary_train_outliers[i].fillna(statistics.mode(salary_train_outliers[i]),inplace=True)
        #print(salary_train_outliers[i].isnull().sum())
#print(salary_train_outliers.isnull().sum())

for i in salary_test_outliers.columns:
    if salary_test_outliers[i].isnull().sum()>0:
        salary_test_outliers[i].fillna(statistics.mode(salary_test_outliers[i]),inplace=True)
        #print(salary_train_outliers[i].isnull().sum())
#print(salary_test_outliers.isnull().sum())



#HANDLING THE SKEWED DATA WHICH 'AGE' USING LOG FUNCTION
salary_train_new['age']=np.log(salary_train_new['age'])
sns.distplot(salary_train_new['age']) 
plt.show()

salary_test_new['age']=np.log(salary_test_new['age'])
sns.distplot(salary_test_new['age']) 
plt.show()



#CREATING DUMMIES FOR CATEGORICAL DATA
salary_train_dummies=pd.get_dummies(salary_train_new[['workclass','education','maritalstatus','occupation','relationship','race','sex','native']])
#print(salary_train_dummies.isnull().sum())

salary_test_dummies=pd.get_dummies(salary_test_new[['workclass','education','maritalstatus','occupation','relationship','race','sex','native']])
#print(salary_test_dummies)



#CONCATINATING DUMMIES AND ORIGINAL DATA AND DROPPING UNWANTED COLUMNS
salary_train_final=pd.concat([salary_train_new,salary_train_dummies],axis=1)
salary_train_final=salary_train_final.drop(salary_train_final[['workclass','education','maritalstatus','occupation','relationship','race','sex','native','capitalgain','capitalloss']],axis=1)
#print(salary_train_final.isnull().sum())

salary_test_final=pd.concat([salary_test_new,salary_test_dummies],axis=1)
salary_test_final=salary_test_final.drop(salary_test_final[['workclass','education','maritalstatus','occupation','relationship','race','sex','native','capitalgain','capitalloss']],axis=1)
#print(salary_test_final.isnull().sum())



#DROPPING TARGET VARIABLE FOR FURTHER PROCESSING OF REMAINING DATAS 
feature_scale_train=salary_train_final.drop(['Salary'],axis=1)
#print(feature_scale_train)

feature_scale_test=salary_test_final.drop(['Salary'],axis=1)
#print(feature_scale_test)



#PREPROCESSING OF FEATURES USING MINMAXSCALER
scaler_train=MinMaxScaler().fit_transform(feature_scale_train)
sal_train=pd.concat([salary_train_final.iloc[:,3],pd.DataFrame(scaler_train,columns=feature_scale.columns)],axis=1)
#print(sal_train)

scaler_test=MinMaxScaler().fit_transform(feature_scale_test)
sal_test=pd.concat([salary_test_final.iloc[:,3],pd.DataFrame(scaler_test,columns=feature_scale.columns)],axis=1)
#print(sal_test)



#FEATURE SELECTION IS DONE HERE WITH REDUCED FEATURES TO 35
X_train=sal_train.iloc[:,1:]
Y_train=sal_train.iloc[:,0]
select_train=SelectKBest(chi2,k=35).fit(X_train,Y_train)
#print(select_train.get_support())

X_test=sal_test.iloc[:,1:]
Y_test=sal_test.iloc[:,0]
select_test=SelectKBest(chi2,k=35).fit(X_test,Y_test)
#print(select_test.get_support())

#STORE IN LIST THE IMP FEATURES
selected_features_train=X_train.columns[(select_train.get_support())]
#print(selected_features_train)

selected_features_test=X_test.columns[(select_test.get_support())]
#print(selected_features_test)


x_new_train=selected_features_train
x_new_test=selected_features_test


#LISTING ALL UNWANTED FEATURES
unwanted=[]
for i in sal_train.columns:
    if i not in x_new_train:
        unwanted.append(i)
#print(unwanted)


#DEOPPING ALL UNWANTED FEATURES
salary_done_train=sal_train.drop(['hoursperweek','workclass_ Local-gov','workclass_ Self-emp-not-inc','workclass_ State-gov','workclass_ Without-pay','education_ 12th','education_ 1st-4th','education_ 5th-6th','education_ 9th','education_ Assoc-acdm',
 'education_ Assoc-voc','education_ Preschool','education_ Some-college','maritalstatus_ Married-AF-spouse','maritalstatus_ Married-spouse-absent','occupation_ Armed-Forces',
 'occupation_ Craft-repair','occupation_ Priv-house-serv','occupation_ Protective-serv','occupation_ Sales','occupation_ Tech-support','occupation_ Transport-moving','race_ Amer-Indian-Eskimo','race_ Asian-Pac-Islander',
 'race_ Other','race_ White','native_ Cambodia','native_ Canada','native_ China','native_ Columbia','native_ Cuba','native_ Dominican-Republic','native_ Ecuador','native_ El-Salvador','native_ England','native_ France',
 'native_ Germany','native_ Greece','native_ Guatemala','native_ Haiti','native_ Honduras','native_ Hong','native_ Hungary','native_ India','native_ Iran','native_ Ireland',
 'native_ Italy','native_ Jamaica','native_ Japan','native_ Laos','native_ Nicaragua','native_ Outlying-US(Guam-USVI-etc)','native_ Peru','native_ Philippines',
 'native_ Poland','native_ Portugal','native_ Puerto-Rico','native_ Scotland','native_ South','native_ Taiwan','native_ Thailand',
 'native_ Trinadad&Tobago','native_ Vietnam','native_ Yugoslavia'],axis=1

salary_done_test=sal_test.drop(['hoursperweek','workclass_ Local-gov','workclass_ Self-emp-not-inc','workclass_ State-gov','workclass_ Without-pay','education_ 12th','education_ 1st-4th','education_ 5th-6th','education_ 9th','education_ Assoc-acdm',
 'education_ Assoc-voc','education_ Preschool','education_ Some-college','maritalstatus_ Married-AF-spouse','maritalstatus_ Married-spouse-absent','occupation_ Armed-Forces',
 'occupation_ Craft-repair','occupation_ Priv-house-serv','occupation_ Protective-serv','occupation_ Sales','occupation_ Tech-support','occupation_ Transport-moving','race_ Amer-Indian-Eskimo','race_ Asian-Pac-Islander',
 'race_ Other','race_ White','native_ Cambodia','native_ Canada','native_ China','native_ Columbia','native_ Cuba','native_ Dominican-Republic','native_ Ecuador','native_ El-Salvador','native_ England','native_ France',
 'native_ Germany','native_ Greece','native_ Guatemala','native_ Haiti','native_ Honduras','native_ Hong','native_ Hungary','native_ India','native_ Iran','native_ Ireland',
 'native_ Italy','native_ Jamaica','native_ Japan','native_ Laos','native_ Nicaragua','native_ Outlying-US(Guam-USVI-etc)','native_ Peru','native_ Philippines',
 'native_ Poland','native_ Portugal','native_ Puerto-Rico','native_ Scotland','native_ South','native_ Taiwan','native_ Thailand',
 'native_ Trinadad&Tobago','native_ Vietnam','native_ Yugoslavia'],axis=1


#COUNT FOR TARGET VALUES
#print(salary_done_train['Salary'].value_counts())
#print(salary_done_test['Salary'].value_counts())

"---------------------------------------------------MODELLING---------------------------------------------------"


#MODEL1
X=salary_done_train.iloc[:,1:]
Y=salary_done_train.iloc[:,0]

X_TEST=salary_done_test.iloc[:,1:]
Y_TEST=salary_done_test.iloc[:,0]

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


#Model1
vector=SVC(random_state=0).fit(X,Y)
#print(vector.score(X,Y))									#79% Accuracy
#print(vector.score(X_TEST,Y_TEST))							#78% Accuracy

pred=vector.predict(X)
pred1=vector.predict(X_TEST)

report_train=classification_report(pred,Y)
#print(report_train)										
report_test=classification_report(pred1,Y_TEST)
#print(report_test)												


#Model2

#Class imbalance
smot=SMOTE()
x_reshape_train,y_reshape_train=smot.fit_sample(X,Y)

smot=SMOTE()
x_reshape_test,y_reshape_test=smot.fit_sample(X_TEST,Y_TEST)

y_reshape_test.value_counts()

support=SVC(kernel='rbf',C=100,gamma='auto',random_state=0).fit(x_reshape_train,y_reshape_train)
#print(support.score(x_reshape_train,y_reshape_train))												#82% Accuracy
#print(support.score(x_reshape_test,y_reshape_test))												#80% Accuracy


pred=vector.predict(x_reshape_train)
pred1=vector.predict(x_reshape_test)

report_train=classification_report(pred,y_reshape_train)
#print(report_train)													#WEIGHTED AVG EACH OF PRECISION & RECALL IS 95%
report_test=classification_report(pred1,y_reshape_test)
#print(report_test)														#WEIGHTED AVG EACH OF PRECISION & RECALL IS 95%

"---------------------------------------------------------------------------------------------------------------------------------------------"