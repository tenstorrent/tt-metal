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

Yolov12 has an attention-centric architecture that moves away from the traditional CNN-based approaches of previous YOLO models while preserving the real-time inference speed crucial for many applications. This model leverages innovative attention mechanisms and a redesigned network architecture to achieve state-of-the-art object detection accuracy without compromising real-time performance.

### Details:
The entry point to yolov12x model is YoloV12x in `models/experimental/yolov12x/tt/ttnn_yolov12x.py`. The model picks up certain configs and weights from Ultralytics pretrained model. We've used weights available [here](https://docs.ultralytics.com/models/yolo12/#performance-metrics) under YOLO12x.

Use the following command to run the Yolo12x model with pre-trained weights :
```sh
pytest tests/ttnn/integration_tests/yolov12x/test_ttnn_yolov12x.py::test_yolov12x[pretrained_weight_true-0]
```
