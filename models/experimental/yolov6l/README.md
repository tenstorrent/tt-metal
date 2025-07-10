# Yolov6l

### Platforms:

Wormhole N150, N300

**Note:** On N300, make sure to use `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` with the pytest.

Or, make sure to set the following environment variable in the terminal:
```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

### Introduction:
YOLOv6-L is a large variant of the YOLOv6 family—an advanced real-time object detection model developed by Meituan. YOLOv6 is designed to offer high performance in both accuracy and speed, making it suitable for industrial applications like autonomous driving, surveillance, and robotics.

Resource link - [source](https://github.com/meituan/YOLOv6)

### Details:
- The entry point to yolov6l model is TtYolov6l in `models/experimental/yolov6l/tt/ttnn_yolov6l.py`.
- Batch size :1
- Supported Input Resolution - (640,480) (Height,Width)

### How to Run:

Use the following command to run the model :
```
pytest --disable-warnings tests/ttnn/integration_tests/yolov6l/test_ttnn_yolov6l.py
```

### Performant Model with Trace+2CQ
- end-2-end perf is 59 FPS

Use the following command to run the performant Model with Trace+2CQs:

```
pytest --disable-warnings models/experimental/yolov6l/tests/perf/test_perf_yolov6l.py::test_perf_yolov6l
```
