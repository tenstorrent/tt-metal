# YOLOV10x
Demo showcasing Yolov10x running on `Wormhole - N150, N300` using ttnn.

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

YOLOv10x introduces a new approach to real-time object detection, addressing both the post-processing and model architecture deficiencies found in previous YOLO versions. By eliminating non-maximum suppression (NMS) and optimizing various model components, YOLOv10x achieves state-of-the-art performance with significantly reduced computational overhead. Extensive experiments demonstrate its superior accuracy-latency trade-offs across multiple model scales. We've used weights available [here](https://docs.ultralytics.com/models/yolov10x/#performance) under YOLOV10x

### Details
- The entry point to yolov10x model is YoloV10x in `models/demos/yolov10x/tt/ttnn_yolov10x.py`.
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution : `(640, 640)` - (Height, Width).
- Dataset used for evaluation : **COCO-2017**

## How to Run:

Use the following command to run the Yolov10x model :

```
pytest --disable-warnings models/demos/yolov10x/tests/pcc/test_ttnn_yolov10x.py::test_yolov10x
```

### Model Performant with Trace+2CQ

#### Single Device (BS=1) :

- For `640x640`, end-2-end perf is `42` FPS.

  ```bash
  pytest --disable-warnings models/demos/yolov10x/tests/test_e2e_performant.py::test_e2e_performant
  ```

#### Multi Device (DP=2, N300) :

- For `640x640`, end-2-end perf is `83` FPS.

  ```bash
  pytest --disable-warnings models/demos/yolov10x/tests/test_e2e_performant.py::test_e2e_performant_dp
  ```

## Demo:

#### Note: Output images will be saved in the `models/demos/yolov10x/demo/runs` folder.

### Single Device (BS=1):

#### Custom Images:

- Use the following command to run demo for `640x640` resolution :

    ```bash
    pytest --disable-warnings models/demos/yolov10x/demo/demo.py::test_demo
    ```

- To use a different image(s) for demo, replace your image(s) in the image path `models/demos/yolov10x/demo/images`

#### Coco-2017 dataset:

- Use the following command to run demo for `640x640` resolution :

  ```
  pytest --disable-warnings models/demos/yolov10x/demo/demo.py::test_demo_dataset
  ```

### Multi Device (DP=2, N300):

#### Custom Images:

- Use the following command to run demo for `640x640` resolution :

  ```bash
  pytest --disable-warnings models/demos/yolov10x/demo/demo.py::test_demo_dp
  ```

#### Coco-2017 dataset:

- Use the following command to run demo for `640x640` resolution :

  ```
  pytest --disable-warnings models/demos/yolov10x/demo/demo.py::test_demo_dataset_dp
  ```

#### Note: The post-processing is performed using PyTorch.

### Inputs
The demo receives inputs from `models/demos/yolov10x/demo/images` dir by default. To test the model on different input data, it is recommended to add a new image file to this directory.

### Outputs
A runs folder will be created inside the `models/demos/yolov10x/demo/` directory. For reference, the model output will be stored in the torch_model directory, while the TTNN model output will be stored in the tt_model directory.

### Additional Information:
The tests can be run with  randomly initialized weights and pre-trained real weights.  To run only for the pre-trained weights, specify pretrained_weight_true when running the tests.

## Web Demo:
Try the interactive web demo following the [instructions](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov10x/web_demo/README.md)
