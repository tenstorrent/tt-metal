# VGG Unet

## Platforms:
    Wormhole (n150, n300)

## Introduction
The VGG-UNet model performs brain tumor segmentation on MRI images. It takes an MRI scan as input and outputs a pixel-wise mask that highlights the regions where a tumor is present. In simple terms, it automatically identifies and outlines brain tumors in medical images to assist doctors in diagnosis and treatment planning.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with following command:
  ```sh
  ./build_metal.sh -p
  ```

## How to Run
**Note:** On N300, make sure to set the following environment variable in the terminal:
```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

### Inference pipeline with random weights and random tensor:
```sh
pytest tests/ttnn/integration_tests/vgg_unet/test_vgg_unet.py::test_vgg_unet[0-pretrained_weight_false]
```

### Inference pipeline with trained weights:
```sh
pytest tests/ttnn/integration_tests/vgg_unet/test_vgg_unet.py::test_vgg_unet[0-pretrained_weight_true]
```

### Performant Model with Trace+2CQ
```sh
pytest models/demos/vgg_unet/tests/test_e2e_performant.py
```

### Performant Demo with Trace+2CQ
#### Single image
- Use the following command to run torch model demo:
```sh
pytest models/demos/vgg_unet/demo/demo.py::test_demo[device_params0-pretrained_weight_true-torch_model-single]
```

- Use the following command to run ttnn model demo:
```sh
pytest models/demos/vgg_unet/demo/demo.py::test_demo[device_params0-pretrained_weight_true-ttnn_model-single]
```

#### Multiple images
- Use the following command to run torch model demo:
```sh
pytest models/demos/vgg_unet/demo/demo.py::test_demo[device_params0-pretrained_weight_true-torch_model-multi]
```

- Use the following command to run ttnn model demo:
```sh
pytest models/demos/vgg_unet/demo/demo.py::test_demo[device_params0-pretrained_weight_true-ttnn_model-multi]
```

## Details
- Entry point for the model is models/demos/vgg_unet/ttnn/ttnn_vgg_unet.py
- Batch Size: 1
- Support Input Resolution: 256x256 (Height, Width)
- End-2-end perf with Trace+2CQs is 80 FPS
