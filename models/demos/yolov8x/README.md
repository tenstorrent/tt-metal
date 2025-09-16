# Yolov8x

## Platforms:
Wormhole (n150, n300)

## Introduction
YOLOv8 is one of the recent iterations in the YOLO series of real-time object detectors, offering cutting-edge performance in terms of accuracy and speed.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`

## How to Run
- Use the following command to run the model:
```
pytest --disable-warnings models/demos/yolov8x/tests/pcc/test_yolov8x.py::test_yolov8x_640
```

## Model performant running with Trace+2CQ
### Single Device (BS=1):
- For `640x640`, end-2-end perf is `66` FPS (**On N150**), _On N300 single device, the FPS will be low as it uses ethernet dispatch_
  ```
  pytest --disable-warnings models/demos/yolov8x/tests/perf/test_e2e_performant.py::test_run_yolov8x_performant
  ```

### Multi Device (DP=2, n300):
- For `640x640`, end-2-end perf is `124` FPS:
  ```
  pytest --disable-warnings models/demos/yolov8x/tests/perf/test_e2e_performant.py::test_run_yolov8x_performant_dp
  ```

### Demo
Note: Output images will be saved in the `models/demos/yolov8x/demo/runs` folder.

### Single Device (BS=1)
#### Custom Images:
- Use the following command to run demo for `640x640` resolution:
  ```bash
  pytest --disable-warnings models/demos/yolov8x/demo/demo.py::test_demo
  ```
  - To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov8x/demo/images` and run the same command.

#### Coco-2017 dataset:
- Use the following command to run demo for `640x640` resolution:
  ```
  pytest --disable-warnings models/demos/yolov8x/demo/demo.py::test_demo_dataset
  ```

### Multi Device (DP=2, N300)
#### Custom Images:
- Use the following command to run demo for `640x640` resolution :
  ```bash
  pytest --disable-warnings models/demos/yolov8x/demo/demo.py::test_demo_dp
  ```
  - To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov8x/demo/images` and run the same command.

#### Coco-2017 dataset:
- Use the following command to run demo for `640x640` resolution :
  ```
  pytest --disable-warnings models/demos/yolov8x/demo/demo.py::test_demo_dataset_dp
  ```

### Performant evaluation with Trace+2CQ
- Use the following command to run the performant evaluation with Trace+2CQs:

  ```
  pytest models/demos/yolo_eval/evaluate.py::test_yolov8x[res0-device_params0-tt_model]
  ```
Note: The model is evaluated with 500 samples.


### Web Demo
- Try the interactive web demo at [yolov8x/web_demo](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov8x/web_demo/README.md)

## Details
- The entry point to the `yolov8x` is located at : `models/demos/yolov8x/tt/ttnn_yolov8x.py`.
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution - `(640, 640)` - (Height, Width).
- The post-processing is performed using PyTorch.

### Inputs
The demo receives inputs from `models/demos/yolov8x/demo/images` dir by default. To test the model on different input data, it is recommended to add a new image file to this directory.

### Outputs
A runs folder will be created inside the `models/demos/yolov8x/demo/runs` directory. For reference, the model output will be stored in the torch_model directory, while the TTNN model output will be stored in the tt_model directory.

### Weights:
The tests can be run with  randomly initialized weights and pre-trained real weights.  To run only for the pre-trained weights, specify pretrained_weight_true when running the tests.
