# Yolov9c

## Platforms:
    Wormhole (n150, n300)

## Introduction:
Yolov9 marks a significant advancement in real-time object detection, introducing groundbreaking techniques such as Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN). Yolov9c is a compact and optimized variant of Yolov9, designed for efficient object detection with reduced computational overhead. It balances speed and accuracy.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`

## How to Run:
**Note:**
- Use `yolov9c-seg.pt` pre-trained weights for segmentation tasks and `yolov9c.pt` pre-trained weights for detection in Tests and Demos.
- Set the `enable_segment` flag accordingly when initializing the TTNN model in tests and demos. Segmentation task is set as default in model.

### Model
Use the following command to run the Yolov9c model:
  ```
  pytest --disable-warnings tests/ttnn/integration_tests/yolov9c/test_ttnn_yolov9c.py::test_yolov9c
  ```

### Demo with Trace+2CQs
#### Single Device (BS=1)
##### Custom Images:
Note: To test the demo with your own images, replace images with `models/demos/yolov9c/demo/images`.

- Use the following command to run the demo for `Instance Segmentation`:
  ```bash
  pytest --disable-warnings models/demos/yolov9c/demo/demo.py::test_demo_segment
  ```

- Use the following command to run the demo for `Object Detection`:
  ```bash
  pytest --disable-warnings models/demos/yolov9c/demo/demo.py::test_demo_detect
  ```

##### Dataset Images - Coco-2017:
- Use the following command to run the demo for `Instance Segmentation`:
  ```bash
  pytest --disable-warnings models/demos/yolov9c/demo/demo.py::test_demo_segment_dataset
  ```

- Use the following command to run the demo for `Object Detection`:
  ```bash
  pytest --disable-warnings models/demos/yolov9c/demo/demo.py::test_demo_detect_dataset
  ```

#### Multi Device (DP=2, n300)
##### Custom Images:
Note: To test the demo with your own images, replace images with `models/demos/yolov9c/demo/images`.

- Use the following command to run the demo for `Instance Segmentation`:
  ```bash
  pytest --disable-warnings models/demos/yolov9c/demo/demo.py::test_demo_segment_dp
  ```

- Use the following command to run the demo for `Object Detection`:
  ```bash
  pytest --disable-warnings models/demos/yolov9c/demo/demo.py::test_demo_detect_dp
  ```

##### Dataset Images - Coco-2017:
- Use the following command to run the demo for `Instance Segmentation`:
  ```bash
  pytest --disable-warnings models/demos/yolov9c/demo/demo.py::test_demo_segment_dataset_dp
  ```

- Use the following command to run the demo for `Object Detection`:
  ```bash
  pytest --disable-warnings models/demos/yolov9c/demo/demo.py::test_demo_detect_dataset_dp
  ```

#### Outputs
- The Demo outputs are saved inside this directory: `models/demos/yolov9c/demo/runs`

### Model performant running with Trace+2CQ
#### Single Device (BS=1):
- For `640x640` - `Segmentation`, end-2-end perf is `45` FPS.
  ```bash
  pytest models/demos/yolov9c/tests/perf/test_e2e_performant_segment.py::test_e2e_performant
  ```

- For `640x640` - `Detection`, end-2-end perf is `65` FPS.
  ```bash
  pytest models/demos/yolov9c/tests/perf/test_e2e_performant_detect.py::test_e2e_performant
  ```

#### Multi Device (DP=2, n300):
- For `640x640` - `Segmentation`, end-2-end perf is `87` FPS.
  ```bash
  pytest models/demos/yolov9c/tests/perf/test_e2e_performant_segment.py::test_e2e_performant_dp
  ```

- For `640x640` - `Detection`, end-2-end perf is `120` FPS.
  ```bash
  pytest models/demos/yolov9c/tests/perf/test_e2e_performant_detect.py::test_e2e_performant_dp
  ```

### Web Demo
- Try the interactive web demo at [yolov9c/web_demo](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov9c/web_demo/README.md)

## Testing
### Performant evaluation with Trace+2CQ for Detection task
- Use the following command to run the performant evaluation with Trace+2CQs:
  ```
  pytest models/experimental/yolo_eval/evaluate.py::test_yolov9c[res0-device_params0-tt_model]
  ```
Note: The model is evaluated with 500 samples.

## Details
The model picks up certain configs and weights from Ultralytics pretrained model. We've used weights available [here](https://docs.ultralytics.com/models/yolov9/#performance-on-ms-coco-dataset) under YOLOv9c.

- The entry point to the `functional_yolov9c` is `YoloV9` in - `models/demos/yolov9c/tt/ttnn_yolov9c.py`.
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution - `(640, 640)` - ( Height, Width ).
