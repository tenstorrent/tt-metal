# Yolov11n - Model

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

### Introduction

**YOLOv11n** is the smallest variant in the YOLOV11 series, it offers improvements in accuracy, speed, and efficiency for real-time object detection. It features enhanced architecture and optimized training methods, suitable for various computer vision tasks.


### Details

- The entry point to the yolov11 is located at:`models/demos/yolov11/tt/ttnn_yolov11.py`
- Batch Size :1
- Supported Input Resolution - (640,640) (Height,Width)
- Dataset used for evaluation - **coco-2017**

### How to Run:

Use the following command to run the model :

```
pytest --disable-warnings tests/ttnn/integration_tests/yolov11/test_ttnn_yolov11.py::test_yolov11
```

### Performant Model with Trace+2CQ
- end-2-end perf is 190 FPS

Use the following command to run the performant Model with Trace+2CQs:

```
pytest --disable-warnings models/demos/yolov11/tests/test_e2e_performant.py
```
### Performant Demo with Trace+2CQ

Use the following command to run the performant Demo with Trace+2CQs:

```
pytest --disable-warnings models/demos/yolov11/demo/demo.py::test_demo
```

### Performant evaluation with Trace+2CQ
Use the following command to run the performant evaluation with Trace+2CQs:

```
pytest models/experimental/yolo_eval/evaluate.py::test_yolov11n[res0-device_params0-tt_model]
```
Note: The model is evaluated with 500 samples.
