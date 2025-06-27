# Yolov8s Model

## Platforms:

    WH N150/N300
**Note:** On N300 ,Make sure to use `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` with the pytest.

Or, make sure to set the following environment variable in the terminal:
```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

## Introduction

YOLOv8 is one of the recent iterations in the YOLO series of real-time object detectors, offering cutting-edge performance in terms of accuracy and speed.

Resource link - [source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py)

### Details

- The entry point to the yolov8s is located at:`models/demos/yolov8s/tt/ttnn_yolov8s.py`
- Batch Size :1
- Supported Input Resolution - (640,640) (Height,Width)

### How to Run:

Use the following command to run the model :

```
pytest --disable-warnings tests/ttnn/integration_tests/yolov8s/test_yolov8s.py::test_yolov8s_640
```

### Performant Model with Trace+2CQ

- end-2-end perf is 143 FPS

Use the following command to run the performant Model with Trace+2CQs:

```
pytest --disable-warnings models/demos/yolov8s/tests/test_e2e_performant.py
```
