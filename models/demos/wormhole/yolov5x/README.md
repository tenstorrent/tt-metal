# Yolov5x

## Platforms:

    Wormhole (n150, n300)

## Introduction
**YOLOv5x** is the largest variant in the YOLOv5 series, delivering top-tier accuracy and performance for advanced object detection tasks. It features a deeper and wider architecture, making it ideal for high-precision applications where accuracy is prioritized over model size or inference speed.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run:
Use the following command to run the model:

```
pytest --disable-warnings models/demos/wormhole/yolov5x/tests/pcc/test_ttnn_yolov5x.py::test_yolov5x
```

### Model performant running with Trace+2CQ

#### Single Device (BS=1) :

- For `640x640`, end-2-end perf is `67` FPS (**On N150**), _On N300 single device, the FPS will be low as it uses ethernet dispatch_

  ```
  pytest --disable-warnings models/demos/wormhole/yolov5x/tests/perf/test_e2e_performant.py::test_e2e_performant
  ```

#### Multi Device (DP=2, N300) :

- For `640x640`, end-2-end perf is `126` FPS.

  ```
  pytest --disable-warnings models/demos/wormhole/yolov5x/tests/perf/test_e2e_performant.py::test_e2e_performant_dp
  ```

### Demo:

#### Single Device (BS=1):

##### Custom Images:

- Use the following command to run demo for `640x640` resolution :

    ```
    pytest --disable-warnings models/demos/wormhole/yolov5x/demo/demo.py::test_demo
    ```

##### Coco-2017 dataset:

- Use the following command to run demo for `640x640` resolution :

  ```
  pytest --disable-warnings models/demos/wormhole/yolov5x/demo/demo.py::test_demo_dataset
  ```

#### Multi Device (DP=2, N300):

##### Custom Images:

- Use the following command to run demo for `640x640` resolution :

  ```bash
  pytest --disable-warnings models/demos/wormhole/yolov5x/demo/demo.py::test_demo_dp
  ```

##### Coco-2017 dataset:

- Use the following command to run demo for `640x640` resolution :

  ```
  pytest --disable-warnings models/demos/wormhole/yolov5x/demo/demo.py::test_demo_dataset_dp
  ```

## Details
- The entry point to the yolov5x is located at:`models/demos/yolov5x/tt/yolov5x.py`
- Batch Size :1
- Supported Input Resolution - (640,640) (Height,Width)
- Dataset used for evaluation : **COCO-2017**
- Note: The post-processing is performed using PyTorch.
