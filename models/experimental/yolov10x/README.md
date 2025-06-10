# YOLOV10x

### Platforms:

Wormhole N150, N300

**Note:** On N300, make sure to use `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` with the pytest.

Or, make sure to set the following environment variable in the terminal:
```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

### Introduction:
YOLOv10x introduces a new approach to real-time object detection, addressing both the post-processing and model architecture deficiencies found in previous YOLO versions. By eliminating non-maximum suppression (NMS) and optimizing various model components, YOLOv10x achieves state-of-the-art performance with significantly reduced computational overhead. Extensive experiments demonstrate its superior accuracy-latency trade-offs across multiple model scales.

### Details
- The entry point to yolov10x model is YoloV10x in `models/experimental/yolov10x/tt/ttnn_yolov10x.py`.
- The model picks up certain configs and weights from Ultralytics pretrained model. We've used weights available [here](https://docs.ultralytics.com/models/yolov10/#performance) under YOLOV10x
- Batch Size: 1
- Support Input Resolution: 640x640 (Height, Width)


### How to Run: (640x640 resolution)

Use the following command to run the Yolov10x model :

```
pytest --disable-warnings tests/ttnn/integration_tests/yolov10x/test_ttnn_yolov10.py::test_yolov10x
```

### Performant Model with Trace+2CQ
- end-2-end perf is 4 FPS <br>

Use the following command to run Model performant running with Trace+2CQ

```
pytest models/experimental/yolov10x/tests/perf/test_e2e_performant.py::test_e2e_performant
```

### Performant Demo with Trace+2CQ

Use the following command to run Model demo running with Trace+2CQ

```
pytest models/experimental/yolov10x/demo/demo.py::test_demo
```

#### Inputs:
The demo receives inputs from `models/experimental/yolov10x/demo/images` dir by default. To test the model on different input data, it is recommended to add a new image file to this directory.
#### Outputs:
A runs folder will be created inside the `models/experimental/yolov10x/demo/` directory. For reference, the model output will be stored in the torch_model directory, while the TTNN model output will be stored in the tt_model directory.

#### Additional Information:
- The post-processing is performed using PyTorch.
- The tests can be run with randomly initialized weights and pre-trained real weights. To run only for the pre-trained weights, specify pretrained_weight_true when running the tests.
