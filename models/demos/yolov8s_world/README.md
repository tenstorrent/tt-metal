# Yolov8s-World
Demo showcasing Yolov8s-World running on `Wormhole - N150, N300` using ttnn.

### Note:

- On N300, Make sure to use `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` with the pytest.

- Or, make sure to set the following environment variable in the terminal:
  ```
  export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
  ```
- To obtain the perf reports through profiler, please build with following command:
  ```
  ./build_metal.sh -p
  ```

## Introduction:

The YOLO-World Model introduces an advanced, real-time Ultralytics YOLOv8-based approach for Open-Vocabulary Detection tasks. This innovation enables the detection of any object within an image based on descriptive texts. By significantly lowering computational demands while preserving competitive performance, YOLO-World emerges as a versatile tool for numerous vision-based applications. Resource link - [source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py)

## Details:
The model picks up certain configs and weights from Ultralytics pretrained model. We've used weights available [here](https://docs.ultralytics.com/models/yolo-world/#available-models-supported-tasks-and-operating-modes) in YOLOv8s-world row.

- The entry point to the `yolov8s_world` is `TtYOLOWorld` in - `models/demos/yolov8s_world/tt/ttnn_yolov8s_world.py`.
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution : `(640, 640)` - (Height, Width).
- Dataset used for evaluation : **COCO-2017**


## How to Run:

Use the following command to run the model :
```
pytest --disable-warnings tests/ttnn/integration_tests/yolov8s_world/test_ttnn_yolov8s_world.py::test_YoloModel
```

### Model Performant with Trace+2CQ

#### Single Device (BS=1) :

- For `640x640`, end-2-end perf is `90` FPS.

  ```bash
  pytest --disable-warnings models/demos/yolov8s_world/tests/perf/test_perf_yolov8s_world.py::test_perf_yolov8s_world
  ```

#### Multi Device (DP=2, N300) :

- For `640x640`, end-2-end perf is `155` FPS.

  ```bash
  pytest --disable-warnings models/demos/yolov8s_world/tests/perf/test_perf_yolov8s_world.py::test_perf_yolov8s_world_dp
  ```

### Demo with Trace+2CQs

##### Note: To test the demo with your own images, replace images with `models/demos/yolov8s_world/demo/images`.

#### Single Device (BS=1) :

#### Custom Images :

- Use the following command to run the performant Demo:

  ```
  pytest --disable-warnings models/demos/yolov8s_world/demo/demo.py::test_demo
  ```

#### Dataset Images :

- Use the following command to run the performant Demo:

  ```
  pytest --disable-warnings models/demos/yolov8s_world/demo/demo.py::test_demo_dataset
  ```


#### Multi Device (DP=2, N300) :

#### Custom Images :

- Use the following command to run the performant Demo:

  ```
  pytest --disable-warnings models/demos/yolov8s_world/demo/demo.py::test_demo_dp
  ```

#### Dataset Images :

- Use the following command to run the performant Demo:

  ```
  pytest --disable-warnings models/demos/yolov8s_world/demo/demo.py::test_demo_dataset_dp
  ```

### Performant evaluation with Trace+2CQ

- Use the following command to run the performant evaluation with Trace+2CQs:

  ```
  pytest models/experimental/yolo_eval/evaluate.py::test_yolov8s_world[res0-device_params0-tt_model]
  ```
Note: The model is evaluated with 500 samples.

### Web Demo
- Try the interactive web demo [instructions](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov8s_world/web_demo/README.md).
