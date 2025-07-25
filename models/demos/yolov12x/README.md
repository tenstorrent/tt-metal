# Yolov12x Model

### Platforms:

Wormhole N150, N300

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

## Introduction

**Yolov12** has an attention-centric architecture that moves away from the traditional CNN-based approaches of previous YOLO models while preserving the real-time inference speed crucial for many applications. This model leverages innovative attention mechanisms and a redesigned network architecture to achieve state-of-the-art object detection accuracy without compromising real-time performance.


### Details

- The entry point to the `yolov12` is located at : `models/demos/yolov12x/tt/yolov12x.py`.
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution : `(640, 640)` - (Height, Width).
- Dataset used for evaluation : **COCO-2017**

## How to Run:

Use the following command to run the model :

```
pytest --disable-warnings models/demos/yolov12x/tests/pcc/test_ttnn_yolov12x.py::test_yolov12x
```

#
## Model performant running with Trace+2CQ

### Single Device (BS=1):
- For `640x640`, end-2-end perf is `14` FPS :

  ```
  pytest --disable-warnings models/demos/yolov12x/tests/perf/test_e2e_performant.py::test_e2e_performant
  ```

### Multi Device (DP=2, N300):

- For `640x640`, end-2-end perf is `28` FPS :

  ```
  pytest --disable-warnings models/demos/yolov12x/tests/perf/test_e2e_performant.py::test_e2e_performant_dp
  ```

### Demo with Trace+2CQ

#### Note: Output images will be saved in the `models/demos/yolov12x/demo/runs` folder.

### Single Device (BS=1):

#### Custom Images:

- Use the following command to run demo for `640x640` resolution :

    ```bash
    pytest --disable-warnings models/demos/yolov12x/demo/demo.py::test_demo
    ```

- To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov12x/demo/images` and run :

    ```bash
    pytest --disable-warnings models/demos/yolov12x/demo/demo.py::test_demo
    ```

#### COCO-2017 dataset:

- Use the following command to run demo for `640x640` resolution :

  ```
  pytest --disable-warnings models/demos/yolov12x/demo/demo.py::test_demo_dataset
  ```

### Multi Device (DP=2, N300):

#### Custom Images:

- Use the following command to run demo for `640x640` resolution :

  ```bash
  pytest --disable-warnings models/demos/yolov11/demo/demo.py::test_demo_dp
  ```

- To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov12x/demo/images` and run :

  ```
  pytest --disable-warnings models/demos/yolov12x/demo/demo.py::test_demo_dp
  ```

#### Coco-2017 dataset:

- Use the following command to run demo for `640x640` resolution :

  ```
  pytest --disable-warnings models/demos/yolov12x/demo/demo.py::test_demo_dataset_dp
  ```
