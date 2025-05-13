# Yolov8x Model

## Platforms:
    WH N300
**Note:** On N300 ,Make sure to use `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` with the pytest.

Or, make sure to set the following environment variable in the terminal:
```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

## Introduction
YOLOv8 is the latest iteration in the YOLO series of real-time object detectors, offering cutting-edge performance in terms of accuracy and speed. Building upon the advancements of previous YOLO versions, YOLOv8 introduces new features and optimizations that make it an ideal choice for various object detection tasks in a wide range of applications.

## Details
The entry point to yolov8x model is YOLOv8x in
`models/demos/yolov8x/tt/ttnn_yolov8x.py`.

### Use the following commands for 640x640 to run the :

- model :
```bash
pytest tests/ttnn/integration_tests/yolov8x/test_yolov8x.py::test_yolov8x_640
```

- e2e perf(1.86 fps) :
```bash
pytest models/demos/yolov8x/tests/test_perf_yolov8x.py::test_yolov8x
```

- e2e perf with trace(40 fps):
```bash
pytest models/demos/yolov8x/tests/test_e2e_performant.py
```

- device perf (46 fps):
```bash
pytest models/demos/yolov8x/tests/test_perf_yolov8x.py::test_perf_device_bare_metal_yolov8x`
```

- yolov8x demo :
```bash
pytest models/demos/yolov8x/demo/demo.py
```

#### Note: The post-processing is performed using PyTorch.

## Inputs
The demo receives inputs from `models/demos/yolov8x/demo/images` dir by default. To test the model on different input data, it is recommended to add a new image file to this directory.

## Outputs
A runs folder will be created inside the `models/demos/yolov8x/demo/` directory. For reference, the model output will be stored in the torch_model directory, while the TTNN model output will be stored in the tt_model directory.

## Additional Information:
The tests can be run with  randomly initialized weights and pre-trained real weights.  To run only for the pre-trained weights, specify pretrained_weight_true when running the tests.

### Owner: [Sabira](https://github.com/sabira-mcw), [Saichand](https://github.com/tenstorrent/tt-metal/pulls/saichandax)
