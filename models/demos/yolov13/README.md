# Yolov11n

## Platforms:
    Wormhole (n150, n300)

## Introduction
**YOLOv13n**

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`

## How to Run
Use the following command to run the model:
```
pytest --disable-warnings models/demos/yolov13/tests/pcc/test_ttnn_yolov13.py::test_yolov13
```

### Model performant running with Trace+2CQ
#### Single Device (BS=1):
- For `640x640`, end-2-end perf is `X` FPS :
```
pytest --disable-warnings models/demos/yolov13/tests/perf/test_e2e_performant.py::test_e2e_performant
```

### Performant Demo with Trace+2CQ
#### Multi Device (DP=2, N300):
- For `640x640`, end-2-end perf is `X` FPS :
  ```
  pytest --disable-warnings models/demos/yolov13/tests/test_e2e_performant.py::test_e2e_performant_dp
  ```

## Details
- The entry point to the `yolov13` is located at : `models/demos/yolov13/tt/ttnn_yolov13.py`
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution : `(640, 640)` - (Height, Width).
- Dataset used for evaluation : **COCO-2017**
