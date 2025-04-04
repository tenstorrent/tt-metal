# Yolov12x Demo
Demo showcasing Yolov12x running on Wormhole - n150, n300 using ttnn.

## Platforms:
    WH N150, N300

## Introduction:
Yolov12 has an attention-centric architecture that moves away from the traditional CNN-based approaches of previous YOLO models while preserving the real-time inference speed crucial for many applications. This model leverages innovative attention mechanisms and a redesigned network architecture to achieve state-of-the-art object detection accuracy without compromising real-time performance.

## Details:
The entry point to functional_yolov12x model is YoloV12x in `models/experimental/functional_yolov12x/tt/ttnn_yolov12x.py`. The model picks up certain configs and weights from Ultralytics pretrained model. We've used weights available [here](https://docs.ultralytics.com/models/yolo12/#performance-metrics) under YOLO12x.


## How to Run:
Use the following command to run the Yolo12x model :
```
pytest --disable-warnings tests/ttnn/integration_tests/yolov12x/test_ttnn_yolov12x.py::test_yolov12x
```
