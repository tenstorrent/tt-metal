# Yolov8s Model

## Platforms:
    WH N150/N300
**Note:** On N300 ,Make sure to use `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` with the pytest.

Or, make sure to set the following environment variable in the terminal:
```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

## Introduction
YOLOv8 is one of the recent iterations in the YOLO series of real-time object detectors, offering cutting-edge performance in terms of accuracy and speed.

## Details
The entry point to yolov8s model is YOLOv8s in
`models/experimental/yolov8s/tt/ttnn_yolov8s.py`.

### Use the following commands for 640x640 to run the :

- yolov8s demo :
```bash
pytest models/experimental/yolov8s/demo/demo.py
```

- e2e perf with trace(100 fps):
```bash
pytest models/experimental/yolov8s/tests/test_e2e_performant.py
```

#### Note: The post-processing is performed using PyTorch.

## Inputs
The demo receives inputs from `models/experimental/yolov8s/demo/images` dir by default. To test the model on different input data, it is recommended to add a new image file to this directory.

## Outputs
A runs folder will be created inside the `models/experimenatl/yolov8s/demo/` directory. For reference, the model output will be stored in the torch_model directory, while the TTNN model output will be stored in the tt_model directory.

## Additional Information:
The tests can be run with  randomly initialized weights and pre-trained real weights.  To run only for the pre-trained weights, specify pretrained_weight_true when running the tests.
