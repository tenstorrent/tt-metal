# Yolov8x Model

## Platforms:
    WH N300

## Introduction
YOLOv8 is the latest iteration in the YOLO series of real-time object detectors, offering cutting-edge performance in terms of accuracy and speed. Building upon the advancements of previous YOLO versions, YOLOv8 introduces new features and optimizations that make it an ideal choice for various object detection tasks in a wide range of applications.

## Details
The entry point to yolov8x model is YOLOv8x in `models/experimental/functional_yolov8x/tt/ttnn_yolov8x.py`.

Use the following command to run the model :
`pytest --disable-warnings tests/ttnn/integration_tests/yolov8x/test_ttnn_yolov8x.py::test_yolov8x_640`

## Batch size: 1

Batch Size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage. It's recommended to set the `batch_size` to 1.

Use the following command to run the yolov8x demo:
`pytest --disable-warnings models/experimental/functional_yolov8x/demo/demo.py`

#### Note: The post-processing is performed using PyTorch.

Use the following command to run the yolov8x performant impelementation:
  `pytest tests/ttnn/integration_tests/yolov8x/test_yolov8x_performant.py::test_run_yolov8x_trace_2cqs_inference`


## Inputs

The demo receives inputs from `models/experimental/functional_yolov8x/demo/images` dir by default. To test the model on different input data, it is recommended to add new image file to this directory.

## Outputs

The runs folder will be created inside the `models/experimental/functional_yolov8x/demo/` dir. For reference, the model output will be stored in `torch_model` dir, while the ttnn model output will be stored in `tt_model` dir.

### Owner: [Sabira](https://github.com/sabira-mcw), [Sudhanshu](https://github.com/ssinghalTT), [Saichand](https://github.com/tenstorrent/tt-metal/pulls/saichandax)
