# Yolov11n

## Platforms:
    Wormhole (n150, n300)

## Introduction
**YOLOv11n** is the smallest variant in the YOLOV11 series, it offers improvements in accuracy, speed, and efficiency for real-time object detection. It features enhanced architecture and optimized training methods, suitable for various computer vision tasks.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`

## How to Run
Use the following command to run the model:
```
pytest --disable-warnings models/demos/yolov11/tests/pcc/test_ttnn_yolov11.py::test_yolov11
```

### Model performant running with Trace+2CQ
#### Single Device (BS=1):
- For `640x640`, end-2-end perf is `179` FPS (**On N150**), _On N300 single device, the FPS will be low as it uses ethernet dispatch_
```
pytest --disable-warnings models/demos/yolov11/tests/perf/test_e2e_performant.py::test_e2e_performant
```

### Performant Demo with Trace+2CQ
#### Multi Device (DP=2, N300):
- For `640x640`, end-2-end perf is `290` FPS :
  ```
  pytest --disable-warnings models/demos/yolov11/tests/test_e2e_performant.py::test_e2e_performant_dp
  ```

### Demo with Trace+2CQ

#### Single Device (BS=1)
##### Custom Images:
- Use the following command to run demo for `640x640` resolution :
  ```bash
  pytest --disable-warnings models/demos/yolov11/demo/demo.py::test_demo
  ```
  - To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov11/demo/images` and run the same command.

#### COCO-2017 dataset:
- Use the following command to run demo for `640x640` resolution :
  ```
  pytest --disable-warnings models/demos/yolov11/demo/demo.py::test_demo_dataset
  ```

### Multi Device (DP=2, n300)
#### Custom Images:
- Use the following command to run demo for `640x640` resolution:
  ```bash
  pytest --disable-warnings models/demos/yolov11/demo/demo.py::test_demo_dp
  ```
  - To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov8x/demo/images` and run the same command.

#### Coco-2017 dataset:
- Use the following command to run demo for `640x640` resolution :
  ```
  pytest --disable-warnings models/demos/yolov11/demo/demo.py::test_demo_dataset_dp
  ```

Note: Output images will be saved in the `models/demos/yolov11/demo/runs` folder.

## Testing
### Performant evaluation with Trace+2CQ

Use the following command to run the performant evaluation with Trace+2CQs:
```
pytest models/demos/yolo_eval/evaluate.py::test_yolov11n[res0-device_params0-tt_model]
```
Note: The model is evaluated with 500 samples.

## Details
- The entry point to the `yolov11` is located at : `models/demos/yolov11/tt/ttnn_yolov11.py`
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution : `(640, 640)` - (Height, Width).
- Dataset used for evaluation : **COCO-2017**
