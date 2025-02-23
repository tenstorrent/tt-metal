# Yolov evaluation

- Using `coco-2017` validation dataset.
- Loading the dataset using `fiftyone` package. Our tt-metal dosen't have fiftyone, Download the package using `pip install fiftyone`.

The following model is evaluated(mAPval 50-95):-
-   YOLOv8x(640x640 resolution) - 0.5994
-   YOLO11n(224x244 resolution) - 0.4289
-   YOLOv4(320x320 resolution) - 0.5643

The above results are for 50 samples.

To run the test please follow the following commands:

- To run YOLOv8x - `pytest models/experimental/yolo_evaluation/yolo_common_evaluation.py::test_yolov8x`
- To run YOLO11n - `pytest models/experimental/yolo_evaluation/yolo_common_evaluation.py::test_yolo11n`
- To run YOLOv4 - `pytest models/experimental/yolo_evaluation/yolo_common_evaluation.py::test_yolov4`
