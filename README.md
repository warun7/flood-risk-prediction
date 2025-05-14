# Multimodal Flood Risk Prediction with Uncertainty Quantification

## 1. Introduction
Accurate and timely flood forecasting remains a critical challenge in disaster management, particularly in regions with diverse terrain and limited historical data. Traditional methods often treat flood prediction as a binary classification task, using either remote sensing or environmental data alone. This can reduce the robustness and interpretability of the forecasts.

This study presents a multimodal regression-based framework for predicting flood risk using temporal satellite imagery and spatial tabular data. The core idea is to improve predictive accuracy and provide more reliable early warnings by producing flood probability estimates as intervals (via Quantile Regression), allowing for better uncertainty calibration.

## 2. Project Overview & Architecture
This project develops a system for flood risk prediction by integrating:
* **Temporal Satellite Imagery:** Seven-day sequences of Sentinel-2 imagery from Google Earth Engine leading up to flood events.
* **Spatial Tabular Data:** Environmental and infrastructural data (rainfall, river discharge, temperature, elevation, land cover, soil type) from public datasets.

The general architectural flow involves:
1.  **Data Collection & Preprocessing:** Gathering and preparing satellite and tabular data.
2.  **Feature Extraction:**
    * **Visual Features:** Using a pre-trained geospatial model (like IBM Granite or a custom CNN) to extract features from satellite image sequences.
    * **Tabular Features:** Processing structured tabular data, potentially with an MLP.
3.  **Temporal Analysis:** Employing an LSTM to capture trends from the sequence of extracted visual features.
4.  **Feature Fusion & Final Prediction:** Combining these multimodal features and feeding them into final predictive models. This project explores Quantile Regression for interval-based probability estimates, along with standard MLP and Logistic Regression classifiers.

*(Conceptual diagrams illustrating the overall pipeline, individual model components like the "SimplerCNNLSTM", "Geospatial Flood Detection (IBM Granite based)", the initial "MLP for Tabular Data", and the "Pipeline for Training Models on Combined Feature Vectors" were developed during the project to visualize these stages.)*

## 3. Key Features
* **Multimodal Data Fusion:** Integrates temporal satellite imagery with diverse spatial tabular data.
* **Temporal Trend Analysis:** Uses LSTMs to understand dynamic changes from image sequences.
* **Uncertainty Quantification:** Leverages Quantile Regression to provide prediction intervals, offering a richer understanding of risk.
* **Pre-trained Model Utilization:** Explores the use of advanced pre-trained models (like IBM Granite) for robust visual feature extraction.
* **Comparative Model Analysis:** Evaluates different final prediction models (Quantile Regression MLP, standard MLP, Logistic Regression) on combined features.
* **Data Balancing:** Addresses class imbalance in training data using techniques like SMOTE.

## 4. Dataset
* **Satellite Data:**
    * Source: Google Earth Engine (via `data_collection.ipynb`).
    * Type: Sentinel-2 imagery.
    * Temporal Extent: 7-day sequences prior to recorded flood events.
    * Processing (in `preprocessing_eda.ipynb`, `cnn_lstm.ipynb`, `ibm-granite-model_and_lstm.ipynb`): Cloud masking, normalization, resampling, patch extraction.
* **Tabular Data:**
    * Source: Public datasets (details likely in `data_collection.ipynb` or `preprocessing_eda.ipynb`).
    * Features: Rainfall, river discharge, temperature, elevation, land cover, soil type.
    * Processing (in `preprocessing_eda.ipynb`): Cleaning, encoding, spatial matching.
* **Combined Data:**
    * The process of combining image-derived features and tabular features, potentially after oversampling (see `oversampling.ipynb`), is explored to create a final dataset for model training in `final_model_on_combined_feature_vectors.ipynb`.

## 5. Methodology & File Descriptions

The project workflow is primarily implemented across several Jupyter Notebooks:

1.  **`data_collection.ipynb`**:
    * **Purpose:** Contains scripts to interact with Google Earth Engine to search for, select, and download Sentinel-2 satellite imagery for specified areas of interest and time periods relevant to flood events.
    * **Key Operations:** API calls to GEE, defining regions/dates, initial data download.

2.  **`preprocessing_eda.ipynb`**:
    * **Purpose:** Handles preprocessing of both satellite imagery and tabular data, along with Exploratory Data Analysis (EDA).
    * **Key Operations (Satellite):** Cloud masking, image normalization, resampling, extraction of image patches/features (e.g., textural features as hinted in snippets, saving to `.pkl` or `.npy`).
    * **Key Operations (Tabular):** Data cleaning (handling missing values, outliers), feature encoding (e.g., one-hot for categorical), scaling/normalization, EDA to understand feature distributions and correlations.

3.  **`cnn_lstm.ipynb`**:
    * **Purpose:** Implements and experiments with a CNN-LSTM architecture for processing sequences of satellite image data. The CNN extracts spatial features from each image in the sequence, and the LSTM models temporal dependencies among these features.
    * **Architecture:** [CNN LSTM Arch](images/cnn_lstm.jpg)

    * **Key Operations:** Defining the CNN-LSTM model, creating PyTorch Datasets/DataLoaders for image sequences, training the model, evaluating its performance on flood event classification/prediction based on image sequences.

4.  **`ibm-granite-model_and_lstm.ipynb`**:
    * **Purpose:** Explores the use of a more advanced pre-trained geospatial model (referred to as "IBM Granite" or similar, as in the "Geospatial Flood Detection" diagram) as a feature extractor for satellite images. The extracted features are then likely fed into an LSTM for temporal analysis.
    * **Architecture:** [IBM Granite + LSTM Arch](images/ibm_lstm.jpg)

    * **Key Operations:** Loading/interfacing with the pre-trained Granite model, extracting features from image sequences, training an LSTM on these features, evaluation.

5.  **`oversampling.ipynb`**:
    * **Purpose:** Addresses class imbalance in the dataset, particularly for the labels indicating flood occurrence. This is crucial for training unbiased models.
    * **Key Operations:** Likely applies techniques like SMOTE (Synthetic Minority Over-sampling Technique) to the (potentially combined) feature set to create a more balanced dataset for training the final predictive models. Outputs a resampled dataset (e.g., `combined_data_smote_10000_scaled_numeric.csv`).

6.  **`final_model_on_combined_feature_vectors.ipynb`**:
    * **Purpose:** This notebook focuses on training and evaluating the final predictive models using a combined feature set derived from both image (LSTM/pre-trained model outputs) and tabular data, likely after oversampling.
    * **Key Operations:**
        * Loading the combined and potentially oversampled features.
        * Implementing, training, and evaluating:
            * **Quantile Regression MLP:** To predict flood probability as intervals (0.05, 0.50, 0.95 quantiles).
            * **Standard MLP Classifier:** For binary flood prediction.
            * **Logistic Regression Classifier:** As a baseline binary classifier.
        * This notebook reflects the "Pipeline for Training Models on Combined Feature Vectors" diagram's final stage.
        * Evaluation includes Pinball loss, standard classification metrics (Accuracy, Precision, Recall, F1, AUC), and interval coverage.
    * **Architecture:** [Final Model Arch](images/final.jpg)

    

7.  **`ensemble_learning.ipynb`**:
    * **Purpose:** Explores combining predictions from multiple models (an ensemble) to potentially improve overall prediction accuracy and robustness.
    * **Key Operations:** May involve techniques like averaging predictions, weighted averaging, or using a meta-learner (stacking) on outputs from models trained in other notebooks (e.g., the CNN-LSTM, Granite-LSTM, or the models from `final_model_on_combined_feature_vectors.ipynb`).

## 6. Installation
1.  Clone this repository.
2.  Run the Notebook you want to checkout, provided you have the data for it.
3.  The Notebooks have the installations for the required libs.

## 7. Usage
1.  Set up the environment as described in the Installation section.
2.  Run the Jupyter Notebooks in a logical order, typically starting with data collection and preprocessing, then model training, and finally evaluation or ensembling.
    ```bash
    jupyter notebook
    ```
    Then open and run cells within:
    * `data_collection.ipynb`
    * `preprocessing_eda.ipynb`
    * Experiment with `cnn_lstm.ipynb` and/or `ibm-granite-model_and_lstm.ipynb` for feature extraction.
    * Run `oversampling.ipynb` if dealing with imbalanced classes for the final modeling stage.
    * Execute `final_model_on_combined_feature_vectors.ipynb` to train and evaluate the primary predictive models.
    * Explore `ensemble_learning.ipynb` for combining model outputs.

    *Note: Ensure any required input data files (CSVs, model weights, etc.) are present in the root directory or update paths within the notebooks accordingly.*

## 8. Evaluation
The project employs various evaluation metrics:
* **For Quantile Regression:** Pinball Loss, Prediction Interval Coverage.
* **For Classification tasks (derived from median quantile or direct classifiers):**
    * Accuracy, Precision, Recall, F1-Score.
    * Confusion Matrix.
    * ROC AUC score.

## 9. Limitations
* **Data Scarcity for Combined Models:** The initial Python snippets for `final_model_on_combined_feature_vectors.ipynb` indicated experiments on a very small combined dataset (N=37 after feature extraction and sampling for some LSTM features). While the project aims for larger sequence analysis, results from such small-scale final model training must be interpreted with caution regarding generalizability.
* **Dependency on Pre-trained Models:** The performance of visual feature extraction heavily relies on the quality and suitability of the chosen pre-trained models.
* **Computational Resources:** Training deep learning models, especially CNNs and LSTMs on sequences of large satellite images, can be computationally intensive.

## 10. Future Work & Deployment Vision
* **Scale Up Dataset:** Acquire and process a significantly larger and more diverse dataset of satellite sequences and corresponding tabular data across India.
* **Refine Models:** Further optimize hyperparameters and architectures for all model components.
* **Operational Deployment:**
    * Develop a real-time system using:
        * **Google Earth Engine API:** For continuous satellite data ingestion (Sentinel-1 for all-weather, Sentinel-2).
        * **IMD API:** For live weather data and forecasts for regions in India.
    * Automate the entire pipeline from data collection, preprocessing, feature extraction, prediction, to visualization.
    * **Output:** A dynamic, interactive map displaying flood risk levels (with uncertainty intervals) for various regions in India, serving as an early warning tool for disaster management authorities.

*(Please replace any generic paths or filenames with your actual ones. If you create a `requirements.txt`, list your packages there.)*