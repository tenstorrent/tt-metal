# Yolo12X

## Platforms:
Wormhole (n150, n300)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
   - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`

## Introduction:

Yolov12 has an attention-centric architecture that moves away from the traditional CNN-based approaches of previous YOLO models while preserving the real-time inference speed crucial for many applications. This model leverages innovative attention mechanisms and a redesigned network architecture to achieve state-of-the-art object detection accuracy without compromising real-time performance.

## How to Run:

Use the following command to run the Yolo12x model with pre-trained weights :
```sh
pytest --disable-warnings models/demos/yolov12x/tests/pcc/test_ttnn_yolov12x.py::test_yolov12x[pretrained_weight_true-0]
```

### Model performant running with Trace+2CQ

#### Single Device (BS=1):

- For `640x640`, end-2-end perf is `14` FPS :

  ```
  pytest --disable-warnings models/demos/yolov12x/tests/perf/test_e2e_performant.py::test_e2e_performant
  ```

#### Multi Device (DP=2, N300):

- For `640x640`, end-2-end perf is `28` FPS :

  ```
  pytest --disable-warnings models/demos/yolov12x/tests/perf/test_e2e_performant.py::test_e2e_performant_dp
  ```

### Demo with Trace+2CQ:

##### Note: Output images will be saved in the `models/demos/yolov12x/demo/runs` folder.

#### Single Device (BS=1):

#### Custom Images:

- Use the following command to run demo for `640x640` resolution :

    ```bash
    pytest --disable-warnings models/demos/yolov12x/demo/demo.py::test_demo
    ```

- To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov12x/demo/images`.

#### COCO-2017 dataset:

- Use the following command to run demo for `640x640` resolution :

  ```
  pytest --disable-warnings models/demos/yolov12x/demo/demo.py::test_demo_dataset
  ```

#### Multi Device (DP=2, N300):

#### Custom Images:

- Use the following command to run demo for `640x640` resolution :

  ```bash
  pytest --disable-warnings models/demos/yolov11/demo/demo.py::test_demo_dp
  ```

- To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov12x/demo/images`.

#### COCO-2017 dataset:

- Use the following command to run demo for `640x640` resolution :

  ```
  pytest --disable-warnings models/demos/yolov12x/demo/demo.py::test_demo_dataset_dp
  ```


### Performant evaluation with Trace+2CQ

- Use the following command to run the performant evaluation with Trace+2CQs:

  ```
  pytest models/demos/yolo_eval/evaluate.py::test_yolov12x[device_params0-tt_model]
  ```
Note: The model is evaluated with 500 samples.

### Details:

The model picks up certain configs and weights from Ultralytics pretrained model. We've used weights available [here](https://docs.ultralytics.com/models/yolo12/#performance-metrics) under YOLO12x.

- The entry point to `yolov12x` model is `YoloV12x` in `models/demos/yolov12x/tt/yolov12x.py`.
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution - `(640, 640)` - (Height, Width).
