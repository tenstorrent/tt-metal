# Yolov8s

## Platforms:
Wormhole (n150, n300), Blackhole (p150)

## Introduction
YOLOv8 is one of the recent iterations in the YOLO series of real-time object detectors, offering cutting-edge performance in terms of accuracy and speed.

Resource link - [source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

### Accuracy (PCC vs Torch)
Runs TT-NN against the Ultralytics reference on a random **640x640** input (needs a device); asserts PCC ≥ 0.99:
```
pytest --disable-warnings models/demos/yolov8s/tests/pcc/test_yolov8s.py::test_yolov8s_640
```

### Model performant running with Trace+2CQ

Measured end-to-end throughput (single device, batch=1, trace + 2 CQ):

| Device | Resolution | FPS |
|---|---|---|
| Wormhole N150 | 640×640 | 215 |
| Blackhole P150 | 640×640 | **372** |

#### Single Device (BS=1):
- end-2-end perf is `215` FPS (**On N150**), _On N300 single device, the FPS will be low as it uses ethernet dispatch_
```
pytest --disable-warnings models/demos/yolov8s/tests/perf/test_e2e_performant.py
```

#### Multi Device (DP=2, n300):
- end-2-end perf is `368` FPS
```
pytest --disable-warnings models/demos/yolov8s/tests/perf/test_e2e_performant.py::test_run_yolov8s_trace_2cqs_dp_inference[wormhole_b0-1-device_params0]
```

#### Device (kernel) perf:
- Reports on-device kernel throughput (`AVG DEVICE KERNEL SAMPLES/S`) via the Tracy profiler.
```
pytest --disable-warnings models/demos/yolov8s/tests/perf/test_perf_yolov8s.py::test_perf_device_yolov8s
```
> Needs a profiler-enabled build (the default for `build_metal.sh`; use `--disable-profiler` to turn off). The expected-perf threshold is Wormhole-calibrated with a hard assert, so on Blackhole the measured samples/s is still reported but the assertion may not match.

### Demo
Note: Output images will be saved in the `models/demos/yolov8s/demo/runs/<model_type>` folder.

### Single Device (BS=1):
- Use the following command to run the Demo with Trace+2CQs:
    ```
    pytest models/demos/yolov8s/demo/demo.py::test_demo[res0-True-tt_model-1-models/demos/yolov8s/demo/images-device_params0]
    ```

### Multi Device (DP=2, n300):
- Use the following command to run the Demo with Trace+2CQs on Multi Device:
    ```
    pytest models/demos/yolov8s/demo/demo.py::test_demo_dp[wormhole_b0-res0-True-tt_model-1-models/demos/yolov8s/demo/images-device_params0]
    ```

- To use a other image(s) for demo, replace your image(s) in the image path `models/demos/yolov8s/demo/images` and run the following to use demo running on multi-device:
  ```
  pytest pytest models/demos/yolov8s/demo/demo.py::test_demo_dp[wormhole_b0-res0-True-tt_model-1-models/demos/yolov8s/demo/images-device_params0]
  ```

### Performant evaluation with Trace+2CQ

- Use the following command to run the performant evaluation with Trace+2CQs:

  ```
  pytest models/demos/yolo_eval/evaluate.py::test_yolov8s[res0-device_params0-tt_model]
  ```

Note: The model is evaluated with 500 samples.

### Web Demo
- Try the interactive web demo at [yolov8s/web_demo](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov8s/web_demo/README.md)

## Details
- The entry point to the yolov8s is located at:`models/demos/yolov8s/tt/ttnn_yolov8s.py`
- Batch Size :1
- Supported Input Resolution - (640,640) (Height,Width)
- Dataset used for evaluation - **COCO-2017**
- The post-processing is performed using PyTorch.

### Inputs
The demo receives inputs from `models/demos/yolov8s/demo/images` dir by default. To test the model on different input data, it is recommended to add a new image file to this directory.

### Outputs
A runs folder will be created inside the `models/demos/yolov8s/demo/` directory. For reference, the model output will be stored in the torch_model directory, while the TTNN model output will be stored in the tt_model directory.
