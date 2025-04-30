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
The entry point to yolov8x model is YOLOv8x in `models/experimental/yolov8x/tt/ttnn_yolov8x.py`.

Use the following command to run the model :
`pytest tests/ttnn/integration_tests/yolov8x/test_yolov8x.py::test_yolov8x_640`

se the following command to run the e2e perf(1.86 fps) :
`pytest models/experimental/yolov8x/tests/test_perf_yolov8x.py::test_yolov8x`

Use the following command to run the device perf :
`pytest models/experimental/yolov8x/tests/test_yolov8x.py::test_perf_device_bare_metal_yolov8x`

Use the following command to run the e2e perf with trace(40 fps):
`pytest models/experimental/yolov8x/tests/test_e2e_performant.py`

## Additional Information:
The tests can be run with  randomly initialized weights and pre-trained real weights.  To run only for the pre-trained weights, specify pretrained_weight_true when running the tests.

### Owner: [Sabira](https://github.com/sabira-mcw), [Saichand](https://github.com/tenstorrent/tt-metal/pulls/saichandax)
