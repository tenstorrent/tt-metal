# Yolov evaluation

- Using `coco-2017` validation dataset.
- Loading the dataset using `fiftyone` package.

### The below observations are for ttnn_model vs dataset(ground truth data):

The following model is evaluated(mAPval 50-95) for 500 samples.:-
-   YOLOv4(320x320 resolution) - 0.7547

Currently, The number of samples is set to 500.

To run the test of ttnn vs ground truth, please follow the following commands:

- To run YOLOv4 - `pytest models/experimental/yolo_evaluation/yolo_common_evaluation.py::test_yolov4[res0-device_params0-tt_model]`


### The below observations are for torch_model vs dataset(ground truth data):

The following model is evaluated(mAPval 50-95) for 500 samples.:-
-   YOLOv4(320x320 resolution) - 0.7610

To run the test of ttnn vs ground truth, please follow the following commands:

- To run YOLOv4 - `pytest models/experimental/yolo_evaluation/yolo_common_evaluation.py::test_yolov4[res0-device_params0-torch_model]`
