# VGG

## Platforms:
    Grayskull (e150), Wormhole (n150, n300)

## Introduction
The VGG model is a popular convolutional neural network architecture introduced by the Visual Geometry Group at Oxford in their paper "Very Deep Convolutional Networks for Large-Scale Image Recognition" (2014). It is widely used for image classification and feature extraction tasks.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

# How to Run
To run the demo for image classification of the VGG model using ImageNet-1k Validation Dataset, use the following command:
```
pytest models/demos/vgg/demo/demo.py::test_demo_imagenet_vgg
```

## Details
### Model Architectures
- VGG11
- VGG16

  NOTE: VGG11 and VGG16 currently supports BATCH_SIZE = 1.
