# Introduction

The VGG model is a popular convolutional neural network architecture introduced by the Visual Geometry Group at Oxford in their paper "Very Deep Convolutional Networks for Large-Scale Image Recognition" (2014). It is widely used for image classification and feature extraction tasks.

# Platforms:
    GS E150, WH N150, WH N300

# Model Architectures
- VGG11
- VGG16
VGG11 and VGG16 currently supports BATCH_SIZE = 1.

# How to Run
To run the demo for image classification of the VGG model using ImageNet-1k Validation Dataset, follow these instructions

- Use the following command to run the model using ttnn_vgg

```
pytest models/demos/vgg/demo/demo.py::test_demo_imagenet_vgg
```

NOTE: one ttnn.reshape in VGG11 and VGG16 is on host.
