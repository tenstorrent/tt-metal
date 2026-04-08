# YOLOv11l

## Platforms:
    Wormhole (n150, n300, multi-device / T3K)

## Introduction
**YOLOv11l** is the large variant in the YOLO11 series (Ultralytics `yolo11l.pt`), ported on TT-NN with trace + 2 CQ. Architecture matches YOLO11l depth/width (dual `C3k` stacks per `C3k2`, dual PSABlocks in `C2PSA`).

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run
Use the following command to run the model:
```
pytest --disable-warnings models/demos/yolov11l/tests/pcc/test_ttnn_yolov11.py::test_yolov11
```

### Model performant running with Trace+2CQ
#### Single Device (BS=1):
- For `640x640`, end-2-end perf is `95` FPS :
```
pytest --disable-warnings models/demos/yolov11l/tests/perf/test_e2e_performant.py::test_e2e_performant
```

### Performant Demo with Trace+2CQ
#### Multi Device (DP=2, N300):
- For `640x640`, end-2-end perf is `157` FPS :
  ```
  pytest --disable-warnings models/demos/yolov11l/tests/perf/test_e2e_performant.py::test_e2e_performant_dp
  ```

### Demo with Trace+2CQ
Note: Output images will be saved in the `models/demos/yolov11l/demo/runs` folder.

#### Single Device (BS=1)
##### Custom Images:
- Use the following command to run demo (`640x640` and `1280x1280` are both parametrized):
  ```bash
  pytest --disable-warnings models/demos/yolov11l/demo/demo.py::test_demo
  ```
  - To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov11l/demo/images` and run the same command.

#### COCO-2017 dataset:
- Use the following command to run demo (`640x640` and `1280x1280` are both parametrized):
  ```
  pytest --disable-warnings models/demos/yolov11l/demo/demo.py::test_demo_dataset
  ```

### Multi Device (DP=2, n300)
#### Custom Images:
- Use the following command to run demo (`640x640` and `1280x1280` are both parametrized):
  ```bash
  pytest --disable-warnings models/demos/yolov11l/demo/demo.py::test_demo_dp
  ```
  - To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov11l/demo/images` and run the same command.

#### Coco-2017 dataset:
- Use the following command to run demo (`640x640` and `1280x1280` are both parametrized):
  ```
  pytest --disable-warnings models/demos/yolov11l/demo/demo.py::test_demo_dataset_dp
  ```

### Large images (SAHI)
YOLOv11l SAHI supports TT input size selection via `--tt-input-size {640,1280}`. For larger scenes (for example `--pre-resize-to 1280 1280`), combine SAHI slicing with optional `--tt-slice-dp-batch` / `--tt-slice-parallel-devices`. Script: `models/demos/yolov11l/sahi_ultralytics_eval.py` (`--tt-model yolov11l`, Ultralytics default weights `yolo11l.pt`).

## Details
- The entry point to the `yolov11l` is located at : `models/demos/yolov11l/tt/ttnn_yolov11.py`
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolutions : `(640, 640)` and `(1280, 1280)` - (Height, Width).
- Dataset used for evaluation : **COCO-2017**
