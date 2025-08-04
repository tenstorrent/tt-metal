# Yolov8s

## Platforms:
    Wormhole (n150, n300)

## Introduction
YOLOv8 is one of the recent iterations in the YOLO series of real-time object detectors, offering cutting-edge performance in terms of accuracy and speed.

Resource link - [source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`

## How to Run
- Use the following command to run the model:
```
pytest --disable-warnings models/demos/yolov8s/tests/pcc/test_yolov8s.py::test_yolov8s_640
```

### Performant Model with Trace+2CQ
```
pytest --disable-warnings models/demos/yolov8s/tests/perf/test_e2e_performant.py
```

### Demo with Trace+2CQ
```
pytest models/demos/yolov8s/demo/demo.py
```

### Web Demo
- Try the interactive web demo at [yolov8x/web_demo](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov8s/web_demo/README.md)

## Testing
### Performant evaluation with Trace+2CQ
```
pytest models/experimental/yolo_eval/evaluate.py::test_yolov8s[res0-device_params0-tt_model]
```
Note: The model is evaluated with 500 samples.

## Details
- The entry point to the yolov8s is located at:`models/demos/yolov8s/tt/ttnn_yolov8s.py`
- Batch Size :1
- Supported Input Resolution - (640,640) (Height,Width)
- Dataset used for evaluation - **coco-2017**
- End-2-end perf is 175 FPS
- Note: The post-processing is performed using PyTorch.

### Inputs
The demo receives inputs from `models/demos/yolov8s/demo/images` dir by default. To test the model on different input data, it is recommended to add a new image file to this directory.

### Outputs
A runs folder will be created inside the `models/demos/yolov8s/demo/` directory. For reference, the model output will be stored in the torch_model directory, while the TTNN model output will be stored in the tt_model directory.
