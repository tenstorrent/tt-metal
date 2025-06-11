# Yolov8s-World Demo
Demo showcasing Yolov8s-World running on Wormhole - n150, n300 using ttnn.

## Platforms:
    WH N150, N300

## Introduction:
The YOLO-World Model introduces an advanced, real-time Ultralytics YOLOv8-based approach for Open-Vocabulary Detection tasks. This innovation enables the detection of any object within an image based on descriptive texts. By significantly lowering computational demands while preserving competitive performance, YOLO-World emerges as a versatile tool for numerous vision-based applications.

## Details:
The entry point to yolov8s_world model is TtYOLOWorld in `models/experimental/yolov8s_world/tt/ttnn_yolov8s_world.py`. The model picks up certain configs and weights from Ultralytics pretrained model. We've used weights available [here](https://docs.ultralytics.com/models/yolo-world/#available-models-supported-tasks-and-operating-modes) in YOLOv8s-world row.

## How to Run:
If running on Wormhole N300 (not required for N150 or Blackhole), the following environment variable needs to be set as the model requires at least 8x8 core grid size:
```sh
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

Use the following command to run the Yolov8s-World model :
```
pytest --disable-warnings tests/ttnn/integration_tests/yolov8s_world/test_ttnn_yolov8s_world.py::test_YoloModel
```
### Demo
Use the following command to run the demo :
```
pytest --disable-warnings models/experimental/yolov8s_world/demo/demo.py
```

## Outputs
The Demo outputs are saved inside this directory: `models/experimental/yolov8s_world/demo/runs`
