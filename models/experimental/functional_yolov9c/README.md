# Yolov9c Demo
Demo showcasing Yolov9c running on Wormhole - n150, n300 using ttnn.

## Platforms:
    WH N150, N300

## Introduction:
Yolov9 marks a significant advancement in real-time object detection, introducing groundbreaking techniques such as Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). Yolov9c is a compact and optimized variant of Yolov9, designed for efficient object detection with reduced computational overhead. It balances speed and accuracy.

## Details:
The entry point to functional_yolov9c model is YoloV9 in `models/experimental/functional_yolov9c/tt/ttnn_yolov9c.py`. The model picks up certain configs and weights from Ultralytics pretrained model. We've used weights available [here](https://docs.ultralytics.com/models/yolov9/#performance-on-ms-coco-dataset) under YOLOv9c.


## How to Run:
Use the following command to run the Yolov9c model :
```
pytest --disable-warnings tests/ttnn/integration_tests/yolov9c/test_ttnn_yolov9c.py::test_yolov9c
```


Torch ops: avg_pool2d
