## Platforms:
    WH N150, N300

## Introduction:
YOLOv10 introduces a new approach to real-time object detection, addressing both the post-processing and model architecture deficiencies found in previous YOLO versions. By eliminating non-maximum suppression (NMS) and optimizing various model components, YOLOv10 achieves state-of-the-art performance with significantly reduced computational overhead. Extensive experiments demonstrate its superior accuracy-latency trade-offs across multiple model scales.

## Details
The entry point to yolov10x model is YoloV10 in `models/demos/yolov10/tt/ttnn_yolov10x.py`. The
model picks up certain configs and weights from Ultralytics pretrained model. We've used weights available [here](https://docs.ultralytics.com/models/yolov10/#performance) under YOLOV10x

## How to Run:
If running on Wormhole N300 (not required for N150 or Blackhole), the following environment variable needs to be set as the model requires at least 8x8 core grid size:
```sh
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

Use the following command to run the Yolov10x model :

```
pytest --disable-warnings tests/ttnn/integration_tests/yolov10/test_ttnn_yolov10.py::test_yolov10x
```
### Demo
- Use the following command to run the demo :

For 640x640:
```
pytest --disable-warnings models/demos/yolov10/demo/demo.py
```

## Outputs
The Demo outputs are saved inside this directory: `models/demos/yolov10/demo/runs/`

## Model e2e device perf
For 640x640:

- end-2-end perf: 1.16 fps (without trace)
- device perf is 37.2 FPS
```
pytest models/experimental/yolov10/tests/perf/test_perf_yolov10.py::test_perf_device_bare_metal_yolov10x
```

## Model performant running with Trace+2CQ
- To be added soon
