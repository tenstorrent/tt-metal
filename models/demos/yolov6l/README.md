# Yolov6l
Demo showcasing Yolov6l running on `Wormhole - N150, N300` using ttnn.

To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

### Introduction:
YOLOv6-L is a large variant of the YOLOv6 family—an advanced real-time object detection model developed by Meituan. YOLOv6 is designed to offer high performance in both accuracy and speed, making it suitable for industrial applications like autonomous driving, surveillance, and robotics. Resource link - [source](https://github.com/meituan/YOLOv6)

### Details
- The entry point to yolov6l model is TtYolov6l in `models/demos/yolov6l/tt/ttnn_yolov6l.py`.
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution : `(640, 640)` - (Height, Width).
- Dataset used for evaluation : **COCO-2017**

### How to Run:

Use the following command to run the model :
```
pytest --disable-warnings models/demos/yolov6l/tests/pcc/test_ttnn_yolov6l.py
```

### Model Performant with Trace+2CQ

#### Single Device (BS=1) :

- For `640x640`, end-2-end perf is `65` FPS.

  ```bash
  pytest --disable-warnings models/demos/yolov6l/tests/perf/test_e2e_performant.py::test_perf_yolov6l
  ```

#### Multi Device (DP=2, N300) :

- For `640x640`, end-2-end perf is `130` FPS.

  ```bash
  pytest --disable-warnings models/demos/yolov6l/tests/perf/test_e2e_performant.py::test_perf_yolov6l_dp
  ```

## Demo:

#### Note: Output images will be saved in the `models/demos/yolov6l/demo/runs` folder.

### Single Device (BS=1):

#### Custom Images:

- Use the following command to run demo for `640x640` resolution :

    ```bash
    pytest --disable-warnings models/demos/yolov6l/demo/demo.py::test_demo
    ```

- To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov6l/demo/images`

#### Coco-2017 dataset:

- Use the following command to run demo for `640x640` resolution :

  ```
  pytest --disable-warnings models/demos/yolov6l/demo/demo.py::test_demo_dataset
  ```

### Multi Device (DP=2, N300):

#### Custom Images:

- Use the following command to run demo for `640x640` resolution :

  ```bash
  pytest --disable-warnings models/demos/yolov6l/demo/demo.py::test_demo_dp
  ```

#### Coco-2017 dataset:

- Use the following command to run demo for `640x640` resolution :

  ```
  pytest --disable-warnings models/demos/yolov6l/demo/demo.py::test_demo_dataset_dp
  ```
