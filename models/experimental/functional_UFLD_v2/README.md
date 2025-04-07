# Ultra-Fast-Lane-Detection-v2

### Platforms:
    WH N300

### Introduction

The Ultra-Fast-Lane-Detection-v2 is a PyTorch-based implementation designed for fast and efficient deep lane detection using hybrid anchor-driven ordinal classification. It enhances the speed and accuracy of lane detection models with a focus on real-time applications.

Resource link - [source](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)

Dataset Link - [source](https://github.com/TuSimple/tusimple-benchmark/issues/3), [kaggle](https://www.kaggle.com/datasets/manideep1108/tusimple?resource=download)


The repository includes models trained on various datasets such as [CULane](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/master/model/model_culane.py), [CurveLanes](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/master/model/model_curvelanes.py), and [Tusimple](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/master/model/model_tusimple.py)



### Model Details

- The entry point to the UFLD_v2 is located at:`models/experimental/functional_UFLD_v2/ttnn/ttnn_UFLD_v2.py`
- The model picks up trained weights from the **tusimple_res34.pth** file located at:`models/experimental/functional_UFLD_v2/reference/tusimple_res34.pth`
- Batch Size :2

### UFLD_v2 Demo

- This demo setup runs with 30 images taken from [Tu_Simple]((https://www.kaggle.com/datasets/manideep1108/tusimple?resource=download)) Dataset with input resolution of (320 x 800)
- Command to Run Demo: `pytest --disable-warnings models/experimental/functional_UFLD_v2/demo/demo.py`
- This setup takes images from this folder(`models/experimental/functional_UFLD_v2/demo/images`) and validates the model results with Ground_table values stored here(`models/experimental/functional_UFLD_v2/demo/GT_test_labels.json`) with F1 score as Output Metric.
- To test the model on different input data, simply add new image files to this images directory and gt values to json.
- The output results for both torch and ttnn model will be stored in this directory(.txt files). - `models/experimental/functional_UFLD_v2/demo/results_txt`

#### Note: The post-processing is performed using PyTorch.


- Use the following command to run the UFLD_v2 performant implementation:
  `pytest tests/ttnn/integration_tests/UFLD_v2/test_UFLD_v2_performant.py::test_run_ufld_v2_trace_2cq_inference`

### Owner: [Venkatesh](https://github.com/vguduruTT)
