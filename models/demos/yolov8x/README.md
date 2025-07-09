# Yolov8x Model

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

### Introduction
YOLOv8 is one of the recent iterations in the YOLO series of real-time object detectors, offering cutting-edge performance in terms of accuracy and speed.

## Details
-   The entry point to yolov8x model is YOLOv8x in
`models/demos/yolov8x/tt/ttnn_yolov8x.py`.
-   Dataset used for evaluation - **coco-2017**

### How to Run:

Use the following command to run the model :
```
pytest --disable-warnings /home/ubuntu/punith/tt-metal/tests/ttnn/integration_tests/yolov8x/test_yolov8x.py::test_yolov8x_640
```

### Performant Model with Trace+2CQ
- end-2-end perf is 47 FPS

Use the following command to run the performant Model with Trace+2CQs:

```
pytest --disable-warnings models/demos/yolov8x/tests/perf/test_e2e_performant.py
```

### Performant Demo with Trace+2CQ
Use the following command to run the performant Demo with Trace+2CQs:

```
pytest --disable-warnings models/demos/yolov8x/demo/demo.py
```

### Performant evaluation with Trace+2CQ
Use the following command to run the performant evaluation with Trace+2CQs:

```
pytest models/experimental/yolo_eval/evaluate.py::test_yolov8x[res0-device_params0-tt_model]
```
Note: The model is evaluated with 500 samples.

### Additional Information:
The tests can be run with  randomly initialized weights and pre-trained real weights.  To run only for the pre-trained weights, specify pretrained_weight_true when running the tests.

### Web Demo
- Try the interactive web demo [instructions](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov8x/README.md)
