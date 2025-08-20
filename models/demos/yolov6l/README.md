# Yolov6l

### Platforms:
Wormhole (n150, n300)

### Introduction:
YOLOv6-L is a large variant of the YOLOv6 family—an advanced real-time object detection model developed by Meituan. YOLOv6 is designed to offer high performance in both accuracy and speed, making it suitable for industrial applications like autonomous driving, surveillance, and robotics. Resource link - [source](https://github.com/meituan/YOLOv6)

### Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`


### How to Run:

Use the following command to run the model :
```
pytest --disable-warnings models/demos/yolov6l/tests/pcc/test_ttnn_yolov6l.py
```

### Model Performant with Trace+2CQ

#### Single Device (BS=1) :

- For `640x640`, end-2-end perf is `65` FPS.

  ```
  pytest --disable-warnings models/demos/yolov6l/tests/perf/test_e2e_performant.py::test_perf_yolov6l
  ```

#### Multi Device (DP=2, N300) :

- For `640x640`, end-2-end perf is `130` FPS.

  ```
  pytest --disable-warnings models/demos/yolov6l/tests/perf/test_e2e_performant.py::test_perf_yolov6l_dp
  ```

### Demo:

#### Single Device (BS=1):

##### Custom Images:

- Use the following command to run demo for `640x640` resolution :

    ```
    pytest --disable-warnings models/demos/yolov6l/demo/demo.py::test_demo
    ```


##### Coco-2017 dataset:

- Use the following command to run demo for `640x640` resolution :

  ```
  pytest --disable-warnings models/demos/yolov6l/demo/demo.py::test_demo_dataset
  ```

#### Multi Device (DP=2, N300):

##### Custom Images:

- Use the following command to run demo for `640x640` resolution :

  ```bash
  pytest --disable-warnings models/demos/yolov6l/demo/demo.py::test_demo_dp
  ```

##### Coco-2017 dataset:

- Use the following command to run demo for `640x640` resolution :

  ```
  pytest --disable-warnings models/demos/yolov6l/demo/demo.py::test_demo_dataset_dp
  ```


### Performant evaluation with Trace+2CQ

- Use the following command to run the performant evaluation with Trace+2CQs:

  ```
  pytest models/demos/yolo_eval/evaluate.py::test_yolov6l[res0-device_params0-tt_model]
  ```

Note: The model is evaluated with 500 samples.


### Details
- The entry point to yolov6l model is TtYolov6l in `models/demos/yolov6l/tt/ttnn_yolov6l.py`.
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution : `(640, 640)` - (Height, Width).
- Dataset used for evaluation : **COCO-2017**
- Note: The post-processing is performed using PyTorch.

### Inputs
The demo receives inputs from `models/demos/yolov6l/demo/images` dir by default. To test the model on different input data, it is recommended to add a new image file to this directory.

### Outputs
A runs folder will be created inside the `models/demos/yolov6l/demo` directory. For reference, the model output will be stored in the `torch_model` directory, while the TTNN model output will be stored in the `tt_model` directory.
