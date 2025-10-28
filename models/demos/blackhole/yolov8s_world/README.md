# Yolov8s-World

## Platforms:
    Blackhole (p150)

## Introduction
Demo showcasing Yolov8s-World running on `Blackhole - p150` using ttnn.

The YOLO-World Model introduces an advanced, real-time Ultralytics YOLOv8-based approach for Open-Vocabulary Detection tasks. This innovation enables the detection of any object within an image based on descriptive texts. By significantly lowering computational demands while preserving competitive performance, YOLO-World emerges as a versatile tool for numerous vision-based applications. Resource link - [source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run
- Use the following command to run the model:
```
pytest --disable-warnings models/demos/blackhole/yolov8s_world/tests/pcc/test_ttnn_yolov8s_world.py::test_yolo_model
```

### Model Performant with Trace+2CQ
#### Single Device (BS=1):
- For `640x640`, end-2-end perf is `105` FPS.
  ```bash
  pytest --disable-warnings models/demos/blackhole/yolov8s_world/tests/perf/test_e2e_performant.py::test_perf_yolov8s_world
  ```

### Demo with Trace+2CQs
Note: To test the demo with your own images, replace images with `models/demos/blackhole/yolov8s_world/demo/images`.

#### Single Device (BS=1)
##### Custom Images:
- Use the following command to run the performant Demo:
  ```
  pytest --disable-warnings models/demos/blackhole/yolov8s_world/demo/demo.py::test_demo
  ```

##### Dataset Images:
- Use the following command to run the performant Demo:
  ```
  pytest --disable-warnings models/demos/blackhole/yolov8s_world/demo/demo.py::test_demo_dataset
  ```

### Web Demo
- Try the interactive web demo at [demos/yolov8s_world/web_demo](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov8s_world/web_demo/README.md)

## Testing
### Performant evaluation with Trace+2CQ
- Use the following command to run the performant evaluation with Trace+2CQs:
  ```
  pytest models/demos/yolo_eval/evaluate.py::test_yolov8s_world[res0-device_params0-tt_model]
  ```
Note: The model is evaluated with 500 samples.

## Details
The model picks up certain configs and weights from Ultralytics pretrained model. We've used weights available [here](https://docs.ultralytics.com/models/yolo-world/#available-models-supported-tasks-and-operating-modes) in YOLOv8s-world row.

- The entry point to the `yolov8s_world` is `TtYOLOWorld` in - `models/demos/yolov8s_world/tt/ttnn_yolov8s_world.py`.
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution : `(640, 640)` - (Height, Width).
- Dataset used for evaluation : **COCO-2017**
