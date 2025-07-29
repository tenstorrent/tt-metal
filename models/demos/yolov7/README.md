# Yolov7

## Platforms:
    Wormhole (n150, n300)

## Introduction
YOLOv7 is a state-of-the-art real-time object detector that surpasses all known object detectors in both speed and accuracy. It builds on the YOLO family of detectors and introduces significant architectural improvements for enhanced speed and accuracy. YOLOv7 supports advanced features such as model reparameterization, extended model scaling, and multi-task capabilities including object detection, instance segmentation, and pose estimation.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with following command:
  ```sh
  ./build_metal.sh -p
  ```

## How to Run
- On N300 ,make sure to set the following environment variable in the terminal:
    ```
    export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
    ```

- Use the following command to run the yolov7 model
```python
pytest --disable-warnings tests/ttnn/integration_tests/yolov7/test_ttnn_yolov7.py
```

### Performant Model With Trace+2CQ:
- end-2-end perf is 89 FPS.
Use the following command to run the performant Model with Trace+2CQ
```python
pytest --disable-warnings models/demos/yolov7/tests/perf/test_e2e_performant.py::test_e2e_performant
 ```

### Performant Demo with Trace+2CQs:
Use the following command to run the performant demo with Trace+2CQs:
```
pytest --disable-warnings models/demos/yolov7/demo/demo.py
```

## Testing

### Performant evaluation with Trace+2CQ
Use the following command to run the performant evaluation with Trace+2CQs:
```
pytest models/experimental/yolo_eval/evaluate.py::test_yolov7[res0-device_params0-tt_model]
```
Note: The model is evaluated with 500 samples.

## Details
- The entry point of the model is located at ```models/demos/yolov7/tt/ttnn_yolov7.py```
- The model picks up weights available [here](https://github.com/WongKinYiu/yolov7?tab=readme-ov-file#performance) under YOLOv7
- Batch Size: 1
- Resolution: 640x640
- Dataset used for evaluation - **coco-2017**
