# Yolov11n

## Platforms:
    Wormhole (n150, n300)

## Introduction

**YOLOv11n** is the smallest variant in the YOLOV11 series, it offers improvements in accuracy, speed, and efficiency for real-time object detection. It features enhanced architecture and optimized training methods, suitable for various computer vision tasks.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with following command:
  ```sh
  ./build_metal.sh -p
  ```

## How to Run
- **Note:** On N300, make sure to set the following environment variable in the terminal:
    ```
    export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
    ```

- Use the following command to run the model:
```
pytest --disable-warnings tests/ttnn/integration_tests/yolov11/test_ttnn_yolov11.py::test_yolov11
```

### Performant Model with Trace+2CQ
```
pytest --disable-warnings models/demos/yolov11/tests/test_e2e_performant.py
```
### Performant Demo with Trace+2CQ
```
pytest --disable-warnings models/demos/yolov11/demo/demo.py::test_demo
```

### Web Demo:
Try the interactive web demo at [yolov11/web_demo](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov11/web_demo/README.md)

## Testing
### Performant evaluation with Trace+2CQ
```
pytest models/experimental/yolo_eval/evaluate.py::test_yolov11n[res0-device_params0-tt_model]
```
Note: The model is evaluated with 500 samples.

## Details
- The entry point to the yolov11 is located at:`models/demos/yolov11/tt/ttnn_yolov11.py`
- Batch Size :1
- Supported Input Resolution - (640,640) (Height,Width)
- Dataset used for evaluation - **coco-2017**
- End-2-end perf is 190 FPS
