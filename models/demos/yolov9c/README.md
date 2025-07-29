# Yolov9c

## Platforms:
    Wormhole (n150, n300)

## Introduction:
Yolov9 marks a significant advancement in real-time object detection, introducing groundbreaking techniques such as Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). Yolov9c is a compact and optimized variant of Yolov9, designed for efficient object detection with reduced computational overhead. It balances speed and accuracy.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with following command:
  ```sh
  ./build_metal.sh -p
  ```

## How to Run:
If running on Wormhole n300 (not required for n150 or Blackhole), the following environment variable needs to be set as the model requires at least 8x8 core grid size:
```sh
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

### Model
- Use the following command to run the Yolov9c model :
```
pytest --disable-warnings tests/ttnn/integration_tests/yolov9c/test_ttnn_yolov9c.py::test_yolov9c
```

**Note:**
- Use `yolov9c-seg.pt` pre-trained weights for segmentation tasks and `yolov9c.pt` pre-trained weights for detection in Tests and Demos.
- Set the `enable_segment` flag accordingly when initializing the TTNN model in tests and demos. Segmentation task is set as default in model.

### Demo
#### Instance Segmentation
- Use the following command to run the demo with Trace and CQs:
```bash
pytest --disable-warnings models/demos/yolov9c/demo/demo.py::test_demo[tt_model-segment-True-models/demos/yolov9c/demo/image.png-device_params0]
```

#### Object Detection
- Use the following command to run the demo with Trace and CQs:
```bash
pytest --disable-warnings models/demos/yolov9c/demo/demo.py::test_demo[tt_model-detect-True-models/demos/yolov9c/demo/image.png-device_params0]
```

**Outputs:** The Demo outputs are saved inside this directory: `models/demos/yolov9c/demo/runs`

### Model performant
#### For 640x640 - Segmentation:
- end-2-end perf with Trace+2CQ for Segmentation is 43 FPS.
  ```bash
  pytest models/demos/yolov9c/tests/perf/test_e2e_performant_segment.py::test_e2e_performant
  ```

#### For 640x640 - Detection:
- end-2-end perf with Trace+2CQ for Detection is 52 FPS.
  ```bash
  pytest models/demos/yolov9c/tests/perf/test_e2e_performant_detect.py::test_e2e_performant
  ```

### Web Demo
- Try the interactive web demo at [yolov9c/web_demo](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov9c/web_demo/README.md)

## Testing
### Performant evaluation with Trace+2CQ for Detection task
```
pytest models/experimental/yolo_eval/evaluate.py::test_yolov9c[res0-device_params0-tt_model]
```
Note: The model is evaluated with 500 samples.

## Details
The entry point to functional_yolov9c model is YoloV9 in `models/demos/yolov9c/tt/ttnn_yolov9c.py`. The model picks up certain configs and weights from Ultralytics pretrained model. We've used weights available [here](https://docs.ultralytics.com/models/yolov9/#performance-on-ms-coco-dataset) under YOLOv9c.
