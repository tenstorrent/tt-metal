### Platforms:

Wormhole N150, N300

To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

### Introduction:

Yolov12 has an attention-centric architecture that moves away from the traditional CNN-based approaches of previous YOLO models while preserving the real-time inference speed crucial for many applications. This model leverages innovative attention mechanisms and a redesigned network architecture to achieve state-of-the-art object detection accuracy without compromising real-time performance.

### Details:
The entry point to yolov12x model is YoloV12x in `models/experimental/yolov12x/tt/ttnn_yolov12x.py`. The model picks up certain configs and weights from Ultralytics pretrained model. We've used weights available [here](https://docs.ultralytics.com/models/yolo12/#performance-metrics) under YOLO12x.

## How to Run:

Use the following command to run the Yolo12x model with pre-trained weights :
```sh
pytest models/experimental/yolov12x/tests/pcc/test_ttnn_yolov12x.py::test_yolov12x[pretrained_weight_true-0]
```

### Model performant running with Trace+2CQ

- For `640x640`, end-2-end perf is `14` FPS :
    ```sh
    pytest models/experimental/yolov12x/tests/perf/test_e2e_performant.py::test_e2e_performant
    ```
