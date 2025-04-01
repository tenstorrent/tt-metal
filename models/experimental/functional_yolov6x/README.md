## Platforms:
    WH N150

## Introduction:
YOLOv6 is a cutting-edge object detector that offers remarkable balance between speed and accuracy, making it a popular choice for real-time applications. This model introduces several notable enhancements on its architecture and training scheme, including the implementation of a Bi-directional Concatenation (BiC) module, an anchor-aided training (AAT) strategy, and an improved backbone and neck design for state-of-the-art accuracy on the COCO dataset.

## Details
The entry point to functional_yolov6x model is YoloV10 in `models/experimental/functional_yolov6x/tt/ttnn_yolov6x.py`. The
model runs with random weights.

## How to Run:
Use the following command to run the Yolov6x model:
```
pytest --disable-warnings tests/ttnn/integration_tests/yolov6x/test_ttnn_yolov6x.py
```

## To run submodules:
To test SPPF submodule, run the command: `pytest tests/ttnn/integration_tests/yolov6x/test_ttnn_sppf.py`
To test Detect submodule, run the command: `pytest tests/ttnn/integration_tests/yolov6x/test_ttnn_detect.py`
