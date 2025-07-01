# Yolov8s

### Platforms:
    WH - N150, N300

### Note:

- On N300 ,Make sure to use `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` with the pytest.

- Or, make sure to set the following environment variable in the terminal:
  ```
  export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
  ```
- To obtain the perf reports through profiler, please build with following command:
  ```
  ./build_metal.sh -p
  ```


## Introduction

YOLOv8 is one of the recent iterations in the YOLO series of real-time object detectors, offering cutting-edge performance in terms of accuracy and speed. Resource link - [source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py)

### Details

- The entry point to the yolov8s is located at:`models/demos/yolov8s/tt/ttnn_yolov8s.py`
- Batch Size :1
- Supported Input Resolution - (640,640) (Height,Width)
- Dataset used for evaluation - **coco-2017**

### How to Run:

Use the following command to run the model :
- The entry point to the `yolov8s` is located at : `models/demos/yolov8s/tt/ttnn_yolov8s.py`.
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution - `(640, 640)` - (Height, Width).


## How to Run:
Use the following command(s) to run the model :

```
pytest --disable-warnings tests/ttnn/integration_tests/yolov8s/test_yolov8s.py::test_yolov8s_640
```

### Model performant running with Trace+2CQ

#### Single Device (BS=1):

- end-2-end perf is `175` FPS

```
pytest --disable-warnings models/demos/yolov8s/tests/test_e2e_performant.py::test_run_yolov8s_trace_2cqs_inference[1-device_params0]
```

#### Multi Device (DP=2, N300):

- end-2-end perf is ` ` FPS

```
pytest --disable-warnings models/demos/yolov8s/tests/test_e2e_performant.py::test_run_yolov8s_trace_2cqs_dp_inference[wormhole_b0-1-device_params0]
```

### Demo

#### Note: Output images will be saved in the `models/demos/yolov8s/demo/runs/<model_type>` folder.

### Single Device (BS=1):

- Use the following command to run the demo with Trace+2CQs :
    ```
    pytest models/demos/yolov8s/demo/demo.py::test_demo[res0-True-tt_model-1-models/demos/yolov8s/demo/images-device_params0]
    ```

### Multi Device (DP=2, N300):

- Use the following command to run the demo with Trace+2CQs on Multi Device :
    ```
    pytest models/demos/yolov8s/demo/demo.py::test_demo_dp[wormhole_b0-res0-True-tt_model-1-models/demos/yolov8s/demo/images-device_params0]
    ```

- To use a other image(s) for demo, replace your image(s) in the image path `models/demos/yolov8s/demo/images` and run the following to use demo running on multi-device:
  ```
  pytest pytest models/demos/yolov8s/demo/demo.py::test_demo_dp[wormhole_b0-res0-True-tt_model-1-models/demos/yolov8s/demo/images-device_params0]
  ```

#### Note: The post-processing is performed using PyTorch.

### Inputs
The demo receives inputs from `models/demos/yolov8s/demo/images` dir by default. To test the model on different input data, it is recommended to add a new image file to this directory.

### Outputs
A runs folder will be created inside the `models/demos/yolov8s/demo/` directory. For reference, the model output will be stored in the torch_model directory, while the TTNN model output will be stored in the tt_model directory.

### Web Demo
- Try the interactive web demo [instructions](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov8s/README.md)
