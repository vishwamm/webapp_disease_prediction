from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app1 = Flask(__name__)

model = RandomForestClassifier()
data = "dataset-2.csv"
data = data.head(300)
symptom_columns = ['a']
encoded_symptoms_list = []
for symptom_column in symptom_columns:
    encoded_symptom = pd.get_dummies(data[symptom_column], prefix=symptom_column)
    encoded_symptoms_list.append(encoded_symptom)
encoded_symptoms = pd.concat(encoded_symptoms_list, axis=1)
data = pd.concat([data, encoded_symptoms], axis=1)
data.drop(symptom_columns, axis=1, inplace=True)
X = data.drop("Disease", axis=1)
y = data["Disease"]
model.fit(X, y)

@app1.route('/', methods=['GET', 'POST'])
def index():
    predicted_disease = None
    
    if request.method == 'POST':
        user_symptoms = []
        for symptom in encoded_symptoms.columns:
            user_input = int(request.form[symptom])
            user_symptoms.append(user_input)
        user_input_df = pd.DataFrame([user_symptoms], columns=encoded_symptoms.columns)
        predicted_disease = model.predict(user_input_df)[0]
    
    return render_template('index.html', predicted_disease=predicted_disease, symptom_columns=encoded_symptoms.columns)

if __name__ == '__main__':
    app1.run(debug=True)