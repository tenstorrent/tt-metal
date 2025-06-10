## Yolov7 Model

# Platforms:
    WH N300, N150

# Introduction:
YOLOv7 is a state-of-the-art real-time object detector that surpasses all known object detectors in both speed and accuracy. It builds on the YOLO family of detectors and introduces significant architectural improvements for enhanced speed and accuracy. YOLOv7 supports advanced features such as model reparameterization, extended model scaling, and multi-task capabilities including object detection, instance segmentation, and pose estimation.

# Details:
The entry point to functional_yolov7 model is ttnn_yolov7 in models/experimental/yolov7/tt/ttnn_yolov7.py. The model picks up weights available [here](https://github.com/WongKinYiu/yolov7?tab=readme-ov-file#performance) under YOLOv7.

### Batch Size: 1

### Resolution: 640x640

## How to run
If running on Wormhole N300 (not required for N150 or Blackhole), the following environment variable needs to be set as the model requires at least 8x8 core grid size:

```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

Use the following command to run the yolov7 model
```
pytest --disable-warnings tests/ttnn/integration_tests/yolov7/test_ttnn_yolov7.py
```

## Demo:
Details will be included in the upcoming PRs.

## Model Performance:
Details will be included in the upcoming PRs.
