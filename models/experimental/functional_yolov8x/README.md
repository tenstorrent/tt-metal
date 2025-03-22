# Yolov8x Model

## Platforms:
    WH N300

## Introduction
YOLOv8 is the latest iteration in the YOLO series of real-time object detectors, offering cutting-edge performance in terms of accuracy and speed. Building upon the advancements of previous YOLO versions, YOLOv8 introduces new features and optimizations that make it an ideal choice for various object detection tasks in a wide range of applications.

## Details
The entry point to yolov8x model is YOLOv8x in `models/experimental/functional_yolov8x/tt/ttnn_yolov8x.py`.

Use the following command to run the model :
`pytest -k "pretrained_weight_true" tests/ttnn/integration_tests/yolov8x/test_yolov8x.py::test_yolov8x_640`

## Additional Information:
The tests can be run with  randomly initialized weights and pre-trained real weights.  To run only for the pre-trained weights, specify pretrained_weight_true when running the tests.

### Owner: [Sabira](https://github.com/sabira-mcw), [Saichand](https://github.com/tenstorrent/tt-metal/pulls/saichandax)
