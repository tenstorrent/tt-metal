# Yolov8x Model

### Platforms:
    WH - N150, N300

### Note:

- On N300, Make sure to use `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` with the pytest.

- Or, make sure to set the following environment variable in the terminal:
  ```
  export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
  ```
- To obtain the perf reports through profiler, please build with following command:
  ```
  ./build_metal.sh -p
  ```

### Introduction
YOLOv8 is one of the recent iterations in the YOLO series of real-time object detectors, offering cutting-edge performance in terms of accuracy and speed.

## Details

- The entry point to the `yolov8x` is located at : `models/demos/yolov8x/tt/ttnn_yolov8x.py`.
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution - `(640, 640)` - (Height, Width).


## How to run
Use the following command(s) to run the model :

## Model performant running with Trace+2CQ

### Single Device (BS=1):
- For `640x640`, end-2-end perf is `40` FPS :

  ```
  pytest models/demos/yolov8x/tests/test_e2e_performant.py::test_run_yolov8x_performant[1-device_params0]
  ```

### Multi Device (DP=2, N300):

- For `640x640`, end-2-end perf is `77` FPS :

  ```
  pytest models/demos/yolov8x/tests/test_e2e_performant.py::test_run_yolov8x_performant_dp[wormhole_b0-1-device_params0]
  ```

## Demo

#### Note: Output images will be saved in the `models/demos/yolov8x/demo/runs` folder.

### Single Device (BS=1):

#### Custom Images:

- Use the following command to run demo for `640x640` resolution :

    ```bash
    pytest --disable-warnings models/demos/yolov8x/demo/demo.py::test_demo
    ```

- To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov8x/demo/images` and run :

  ```
  pytest --disable-warnings models/demos/yolov8x/demo/demo.py::test_demo
  ```

#### Coco-2017 dataset:

- Use the following command to run demo for `640x640` resolution :

  ```
  pytest --disable-warnings models/demos/yolov8x/demo/demo.py::test_demo_dataset
  ```

### Multi Device (DP=2, N300):

#### Custom Images:

- Use the following command to run demo for `640x640` resolution :

  ```bash
  pytest --disable-warnings models/demos/yolov8x/demo/demo.py::test_demo_dp
  ```

- To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov8x/demo/images` and run :

  ```
  pytest --disable-warnings models/demos/yolov8x/demo/demo.py::test_demo_dp
  ```

#### Coco-2017 dataset:

- Use the following command to run demo for `640x640` resolution :

  ```
  pytest --disable-warnings models/demos/yolov8x/demo/demo.py::test_demo_dataset_dp
  ```


#### Note: The post-processing is performed using PyTorch.

## Inputs
The demo receives inputs from `models/demos/yolov8x/demo/images` dir by default. To test the model on different input data, it is recommended to add a new image file to this directory.

## Outputs
A runs folder will be created inside the `models/demos/yolov8x/demo/runs` directory. For reference, the model output will be stored in the torch_model directory, while the TTNN model output will be stored in the tt_model directory.

## Additional Information:
The tests can be run with  randomly initialized weights and pre-trained real weights.  To run only for the pre-trained weights, specify pretrained_weight_true when running the tests.

### Web Demo
- Try the interactive web demo [instructions](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov8x/web_demo/README.md).
