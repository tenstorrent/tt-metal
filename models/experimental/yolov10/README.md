# YOLOV10x

## Platforms:
    WH N150, N300

#### Resolution: 640x640
#### Batch size: 1

## Introduction:
YOLOv10 introduces a new approach to real-time object detection, addressing both the post-processing and model architecture deficiencies found in previous YOLO versions. By eliminating non-maximum suppression (NMS) and optimizing various model components, YOLOv10 achieves state-of-the-art performance with significantly reduced computational overhead. Extensive experiments demonstrate its superior accuracy-latency trade-offs across multiple model scales.

## Details
The entry point to yolov10x model is YoloV10 in `models/experimental/yolov10/tt/ttnn_yolov10x.py`. The
model picks up certain configs and weights from Ultralytics pretrained model. We've used weights available [here](https://docs.ultralytics.com/models/yolov10/#performance) under YOLOV10x

## How to Run: (640x640 resolution)
If running on Wormhole N300 (not required for N150 or Blackhole), the following environment variable needs to be set as the model requires at least 8x8 core grid size:
```sh
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```
To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

Use the following command to run the Yolov10x model :

```
pytest --disable-warnings tests/ttnn/integration_tests/yolov10/test_ttnn_yolov10.py::test_yolov10x
```

## Model performant running with Trace+2CQ
- end-2-end perf is 4 FPS <br>

Use the following command to run Model performant running with Trace+2CQ

```
pytest models/experimental/yolov10/tests/perf/test_e2e_performant.py::test_e2e_performant
```

## Model demo running with Trace+2CQ

Use the following command to run Model demo running with Trace+2CQ

```
pytest models/experimental/yolov10/demo/demo.py::test_demo_ttnn
```

#### Note: The post-processing is performed using PyTorch.

## Inputs
The demo receives inputs from `models/experimental/yolov10/demo/images` dir by default. To test the model on different input data, it is recommended to add a new image file to this directory.
## Outputs
A runs folder will be created inside the `models/experimental/yolov10/demo/` directory. For reference, the model output will be stored in the torch_model directory, while the TTNN model output will be stored in the tt_model directory.

## Additional Information:
The tests can be run with  randomly initialized weights and pre-trained real weights.  To run only for the pre-trained weights, specify pretrained_weight_true when running the tests.
