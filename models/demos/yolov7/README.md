# Yolov7 Model
Demo showcasing Yolov7 running on `Wormhole - N150, N300` using ttnn.

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

YOLOv7 is a state-of-the-art real-time object detector that surpasses all known object detectors in both speed and accuracy. It builds on the YOLO family of detectors and introduces significant architectural improvements for enhanced speed and accuracy. YOLOv7 supports advanced features such as model reparameterization, extended model scaling, and multi-task capabilities including object detection, instance segmentation, and pose estimation. The model picks up weights available [here](https://github.com/WongKinYiu/yolov7?tab=readme-ov-file#performance) under YOLOv7

### Details
- The entry point of the model is located at ```models/demos/yolov7/tt/ttnn_yolov7.py```
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution : `(640, 640)` - (Height, Width).
- Dataset used for evaluation : **COCO-2017**


## How to run :

Use the following command to run the yolov7 model

```python
pytest --disable-warnings models/demos/yolov7/tests/pcc/test_ttnn_yolov7.py
```

### Model Performant with Trace+2CQ

#### Single Device (BS=1) :

- For `640x640`, end-2-end perf is `70` FPS.

  ```bash
  pytest --disable-warnings models/demos/yolov7/tests/perf/test_e2e_performant.py::test_e2e_performant
  ```

#### Multi Device (DP=2, N300) :

- For `640x640`, end-2-end perf is `134` FPS.

  ```bash
  pytest --disable-warnings models/demos/yolov7/tests/perf/test_e2e_performant.py::test_e2e_performant_dp
  ```

## Demo:

#### Note: Output images will be saved in the `models/demos/yolov7/demo/runs` folder.

### Single Device (BS=1):

#### Custom Images:

- Use the following command to run demo for `640x640` resolution :

    ```bash
    pytest --disable-warnings models/demos/yolov7/demo/demo.py::test_demo
    ```

- To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov7/demo/images`


#### Coco-2017 dataset:

- Use the following command to run demo for `640x640` resolution :

  ```
  pytest --disable-warnings models/demos/yolov7/demo/demo.py::test_demo_dataset
  ```

### Multi Device (DP=2, N300):

#### Custom Images:

- Use the following command to run demo for `640x640` resolution :

  ```bash
  pytest --disable-warnings models/demos/yolov7/demo/demo.py::test_demo_dp
  ```

#### Coco-2017 dataset:

- Use the following command to run demo for `640x640` resolution :

  ```
  pytest --disable-warnings models/demos/yolov7/demo/demo.py::test_demo_dataset_dp
  ```

### Performant evaluation with Trace+2CQ
Use the following command to run the performant evaluation with Trace+2CQs:

```
pytest models/experimental/yolo_eval/evaluate.py::test_yolov7[res0-device_params0-tt_model]
```
Note: The model is evaluated with 500 samples.
