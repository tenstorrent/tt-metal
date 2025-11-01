# VGG Unet

## Platforms:
    Blackhole (p150)

## Introduction
The VGG-UNet model performs brain tumor segmentation on MRI images. It takes an MRI scan as input and outputs a pixel-wise mask that highlights the regions where a tumor is present. In simple terms, it automatically identifies and outlines brain tumors in medical images to assist doctors in diagnosis and treatment planning.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run
### Inference pipeline with random weights and random tensor:
```sh
pytest models/demos/blackhole/vgg_unet/tests/pcc/test_vgg_unet.py::test_vgg_unet[0-pretrained_weight_false]
```

### Inference pipeline with trained weights:
```sh
pytest models/demos/blackhole/vgg_unet/tests/pcc/test_vgg_unet.py::test_vgg_unet[0-pretrained_weight_true]
```

### Performant Model with Trace+2CQ
#### Single Device (BS=1):
Use the following command to run the e2e perf with trace 2cq:
```sh
pytest models/demos/blackhole/vgg_unet/tests/perf/test_e2e_performant.py::test_vgg_unet_e2e
```
- end-2-end perf with Trace+2CQs is 320 FPS (**On P150**)


### Performant Demo with Trace+2CQ
#### Single Device (BS=1):
Use the following command to run performant model demo (supports single and multiple images):
```sh
pytest models/demos/blackhole/vgg_unet/demo/demo.py::test_demo
```

### Performant Data evaluation with Trace+2CQ:
#### Single Device (BS=1):
Use the following command to run the performant evaluation with Trace+2CQs:
```sh
pytest models/demos/segmentation_evaluation/test_segmentation_eval.py::test_vgg_unet
```

## Details
- Entry point for the model is `models/demos/vgg_unet/ttnn/ttnn_vgg_unet.py`
- Batch Size: 1
- Support Input Resolution: 256x256 (Height, Width)
