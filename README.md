# Bike-aluguel-Dio
Repositorio do curso AI900 Dio
Este repositório foi criado para armazenar o material do curso AI Machine Learning na Prática no Azure ML
A escolha de um algoritmo foi citada no curso a tarefa foi regressão.
Foi muito vago o curso não diria que seria um básico pois nem isso foi.
azureml:azureml_dio-aluguel-bike_2_output_mlflow_log_model_247586834:1
jason
{
    "runId": "dio-aluguel-bike",
    "runUuid": "114befc9-48ce-4762-96f3-63c5817fd851",
    "parentRunUuid": null,
    "rootRunUuid": "114befc9-48ce-4762-96f3-63c5817fd851",
    "target": "Serverless",
    "status": "Completed",
    "parentRunId": null,
    "dataContainerId": "dcid.dio-aluguel-bike",
    "createdTimeUtc": "2024-12-05T20:44:53.3886899+00:00",
    "startTimeUtc": "2024-12-05T20:45:09.986Z",
    "endTimeUtc": "2024-12-05T20:56:03.630Z",
    "error": null,
    "warnings": null,
    "tags": {
        "_aml_system_automl_mltable_data_json": "{\"Type\":\"MLTable\",\"TrainData\":{\"Uri\":\"azureml://locations/eastus/workspaces/a6989bb3-272d-4303-95b5-e92aa39fc8e3/data/Dio-aluguel-bike/versions/1\",\"ResolvedUri\":null,\"AssetId\":null},\"TestData\":null,\"ValidData\":null}",
        "model_explain_run": "best_run",
        "_aml_system_automl_run_workspace_id": "a6989bb3-272d-4303-95b5-e92aa39fc8e3",
        "_aml_system_azureml.automlComponent": "AutoML",
        "pipeline_id_000": "faf12f74cf9bbd358ca5525682c5030d36f7be7c;b76be6b5846772ee1128c4d415381c1e9fed455e;__AutoML_Ensemble__",
        "score_000": "0.09357675586470318;0.09663945142276895;0.08858516656802386",
        "predicted_cost_000": "0;0.5;0",
        "fit_time_000": "0.052579999999999995;0.035232;1",
        "training_percent_000": "100;100;100",
        "iteration_000": "0;1;2",
        "run_preprocessor_000": "MaxAbsScaler;MaxAbsScaler;",
        "run_algorithm_000": "LightGBM;RandomForest;VotingEnsemble",
        "automl_best_child_run_id": "dio-aluguel-bike_2"
    },
    "properties": {
        "num_iterations": "3",
        "training_type": "TrainFull",
        "acquisition_function": "EI",
        "primary_metric": "normalized_root_mean_squared_error",
        "train_split": "0",
        "acquisition_parameter": "0",
        "num_cross_validation": "",
        "target": "Serverless",
        "AMLSettingsJsonString": "{\"is_subgraph_orchestration\":false,\"is_automode\":true,\"path\":\"./sample_projects/Dio-aluguel-bike\",\"subscription_id\":\"24edc700-341a-45c6-b837-f9f956bec465\",\"resource_group\":\"AI900-LAB\",\"workspace_name\":\"LaboratorioAI900\",\"iterations\":3,\"primary_metric\":\"normalized_root_mean_squared_error\",\"task_type\":\"regression\",\"IsImageTask\":false,\"IsTextDNNTask\":false,\"validation_size\":0.1,\"n_cross_validations\":null,\"preprocess\":true,\"is_timeseries\":false,\"time_column_name\":null,\"grain_column_names\":null,\"max_cores_per_iteration\":-1,\"max_concurrent_iterations\":3,\"max_nodes\":3,\"iteration_timeout_minutes\":15,\"enforce_time_on_windows\":false,\"experiment_timeout_minutes\":15,\"exit_score\":\"NaN\",\"experiment_exit_score\":\"NaN\",\"whitelist_models\":[\"RandomForest\",\"LightGBM\"],\"blacklist_models\":null,\"blacklist_algos\":[\"TensorFlowDNN\",\"TensorFlowLinearRegressor\"],\"auto_blacklist\":false,\"blacklist_samples_reached\":false,\"exclude_nan_labels\":false,\"verbosity\":20,\"model_explainability\":false,\"enable_onnx_compatible_models\":false,\"enable_feature_sweeping\":false,\"send_telemetry\":true,\"enable_early_stopping\":true,\"early_stopping_n_iters\":20,\"distributed_dnn_max_node_check\":false,\"enable_distributed_featurization\":true,\"enable_distributed_dnn_training\":true,\"enable_distributed_dnn_training_ort_ds\":false,\"ensemble_iterations\":3,\"enable_tf\":false,\"enable_cache\":false,\"enable_subsampling\":false,\"metric_operation\":\"minimize\",\"enable_streaming\":false,\"use_incremental_learning_override\":false,\"force_streaming\":false,\"enable_dnn\":false,\"is_gpu_tmp\":false,\"enable_run_restructure\":false,\"featurization\":\"auto\",\"vm_type\":\"Standard_DS3_v2\",\"vm_priority\":\"dedicated\",\"label_column_name\":\"rentals\",\"weight_column_name\":null,\"miro_flight\":\"default\",\"many_models\":false,\"many_models_process_count_per_node\":0,\"automl_many_models_scenario\":null,\"enable_batch_run\":true,\"save_mlflow\":true,\"track_child_runs\":true,\"test_include_predictions_only\":false,\"enable_mltable_quick_profile\":\"True\",\"has_multiple_series\":false,\"_enable_future_regressors\":false,\"enable_ensembling\":true,\"enable_stack_ensembling\":false,\"ensemble_download_models_timeout_sec\":300.0,\"stack_meta_learner_train_percentage\":0.2}",
        "DataPrepJsonString": null,
        "EnableSubsampling": "False",
        "runTemplate": "AutoML",
        "azureml.runsource": "automl",
        "_aml_internal_automl_best_rai": "False",
        "ClientType": "Mfe",
        "_aml_system_scenario_identification": "Remote.Parent",
        "PlatformVersion": "DPV2",
        "environment_cpu_name": "AzureML-ai-ml-automl",
        "environment_cpu_label": "7",
        "environment_gpu_name": "AzureML-ai-ml-automl-gpu",
        "environment_gpu_label": "6",
        "root_attribution": "automl",
        "attribution": "AutoML",
        "Orchestrator": "AutoML",
        "CancelUri": "https://eastus.api.azureml.ms/jasmine/v1.0/subscriptions/24edc700-341a-45c6-b837-f9f956bec465/resourceGroups/AI900-LAB/providers/Microsoft.MachineLearningServices/workspaces/LaboratorioAI900/experimentids/160d4587-e88c-460e-956a-7e49730484ed/cancel/dio-aluguel-bike",
        "mltable_data_json": "{\"Type\":\"MLTable\",\"TrainData\":{\"Uri\":\"azureml://locations/eastus/workspaces/a6989bb3-272d-4303-95b5-e92aa39fc8e3/data/Dio-aluguel-bike/versions/1\",\"ResolvedUri\":\"azureml://locations/eastus/workspaces/a6989bb3-272d-4303-95b5-e92aa39fc8e3/data/Dio-aluguel-bike/versions/1\",\"AssetId\":\"azureml://locations/eastus/workspaces/a6989bb3-272d-4303-95b5-e92aa39fc8e3/data/Dio-aluguel-bike/versions/1\"},\"TestData\":null,\"ValidData\":null}",
        "ClientSdkVersion": null,
        "snapshotId": "00000000-0000-0000-0000-000000000000",
        "SetupRunId": "dio-aluguel-bike_setup",
        "SetupRunContainerId": "dcid.dio-aluguel-bike_setup",
        "FeaturizationRunJsonPath": "featurizer_container.json",
        "FeaturizationRunId": "dio-aluguel-bike_featurize",
        "ProblemInfoJsonString": "{\"dataset_num_categorical\": 0, \"is_sparse\": true, \"subsampling\": false, \"has_extra_col\": true, \"dataset_classes\": 552, \"dataset_features\": 64, \"dataset_samples\": 657, \"single_frequency_class_detected\": false}"
    },
    "parameters": {},
    "services": {},
    "inputDatasets": null,
    "outputDatasets": [],
    "runDefinition": null,
    "logFiles": {},
    "jobCost": {
        "chargedCpuCoreSeconds": null,
        "chargedCpuMemoryMegabyteSeconds": null,
        "chargedGpuSeconds": null,
        "chargedNodeUtilizationSeconds": null
    },
    "revision": 13,
    "runTypeV2": {
        "orchestrator": "AutoML",
        "traits": [
            "automl",
            "Remote.Parent"
        ],
        "attribution": null,
        "computeType": null
    },
    "settings": {},
    "computeRequest": null,
    "compute": {
        "target": "Serverless",
        "targetType": "AmlCompute",
        "vmSize": "Standard_DS3_v2",
        "instanceType": "Standard_DS3_v2",
        "instanceCount": 1,
        "gpuCount": null,
        "priority": "Dedicated",
        "region": null,
        "armId": null,
        "properties": null
    },
    "createdBy": {
        "userObjectId": "6f8b71e8-7649-4a99-9ac4-cbda3f45d202",
        "userPuId": "100320015910457F",
        "userIdp": "live.com",
        "userAltSecId": "1:live.com:00037FFED49CBEDC",
        "userIss": "https://sts.windows.net/7c579a76-b1d2-4540-8565-24971bfc06d3/",
        "userTenantId": "7c579a76-b1d2-4540-8565-24971bfc06d3",
        "userName": "wesley pires",
        "upn": null
    },
    "computeDuration": "00:10:53.6438246",
    "effectiveStartTimeUtc": null,
    "runNumber": 1733431493,
    "rootRunId": "dio-aluguel-bike",
    "experimentId": "160d4587-e88c-460e-956a-7e49730484ed",
    "userId": "6f8b71e8-7649-4a99-9ac4-cbda3f45d202",
    "statusRevision": 3,
    "currentComputeTime": null,
    "lastStartTimeUtc": null,
    "lastModifiedBy": {
        "userObjectId": "6f8b71e8-7649-4a99-9ac4-cbda3f45d202",
        "userPuId": "100320015910457F",
        "userIdp": "live.com",
        "userAltSecId": "1:live.com:00037FFED49CBEDC",
        "userIss": "https://sts.windows.net/7c579a76-b1d2-4540-8565-24971bfc06d3/",
        "userTenantId": "7c579a76-b1d2-4540-8565-24971bfc06d3",
        "userName": "wesley pires",
        "upn": null
    },
    "lastModifiedUtc": "2024-12-05T20:56:03.2399617+00:00",
    "duration": "00:10:53.6438246",
    "inputs": {
        "training_data": {
            "assetId": "azureml://locations/eastus/workspaces/a6989bb3-272d-4303-95b5-e92aa39fc8e3/data/Dio-aluguel-bike/versions/1",
            "type": "MLTable"
        }
    },
    "outputs": {
        "best_model": {
            "assetId": "azureml://locations/eastus/workspaces/a6989bb3-272d-4303-95b5-e92aa39fc8e3/models/azureml_dio-aluguel-bike_2_output_mlflow_log_model_247586834/versions/1",
            "type": "MLFlowModel"
        }
    },
    "currentAttemptId": 1
}
