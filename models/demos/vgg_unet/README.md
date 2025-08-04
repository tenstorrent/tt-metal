# VGG Unet
### Platforms:

Wormhole N150, N300

To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

### Introduction
The VGG-UNet model performs brain tumor segmentation on MRI images. It takes an MRI scan as input and outputs a pixel-wise mask that highlights the regions where a tumor is present. In simple terms, it automatically identifies and outlines brain tumors in medical images to assist doctors in diagnosis and treatment planning.
### Details
- Entry point for the model is `models/demos/vgg_unet/ttnn/ttnn_vgg_unet.py`
- Batch Size: 1
- Support Input Resolution: 256x256 (Height, Width)

### How to run

To run the inference, make sure to build the project, activate the environment, and set the appropriate environment variables.
For more information, refer to the [installation and build guide](https://docs.tenstorrent.com/tt-metalium/latest/get_started/get_started.html#install-and-build).

Command to run the inference pipeline with random weights and random tensor:

```sh
pytest models/demos/vgg_unet/tests/pcc/test_vgg_unet.py::test_vgg_unet[0-pretrained_weight_false]
```

Command to run the inference pipeline with trained weights:

```sh
pytest models/demos/vgg_unet/tests/pcc/test_vgg_unet.py::test_vgg_unet[0-pretrained_weight_true]
```

### Performant Model with Trace+2CQ

#### Single Device (BS=1):

Use the following command to run the e2e perf with trace 2cq:
```sh
pytest --disable-warnings models/demos/vgg_unet/tests/perf/test_e2e_performant.py::test_vgg_unet_e2e
```
- end-2-end perf with Trace+2CQs is 90 FPS

#### Multi Device (DP=2, N300):

Use the following command to run the e2e perf with trace 2cq:
```sh
pytest --disable-warnings models/demos/vgg_unet/tests/perf/test_e2e_performant.py::test_vgg_unet_e2e_dp
```
- end-2-end perf with Trace+2CQs is 158 FPS


### Performant Demo with Trace+2CQ

#### Single Device (BS=1):

Use the following command to run performant model demo (supports single and multiple images):

```sh
pytest --disable-warnings models/demos/vgg_unet/demo/demo.py::test_demo
```

#### Multi Device (DP=2, N300):


Use the following command to run performant model demo:

```sh
pytest --disable-warnings models/demos/vgg_unet/demo/demo.py::test_demo_dp
```

### Performant Data evaluation with Trace+2CQ:

#### Single Device (BS=1):

Use the following command to run the performant evaluation with Trace+2CQs:

```sh
pytest models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vgg_unet
```

#### Multi Device (DP=2, N300):

Use the following command to run the performant evaluation with Trace+2CQs:

```sh
pytest models/experimental/segmentation_evaluation/test_segmentation_eval.py::test_vgg_unet_dp
```
