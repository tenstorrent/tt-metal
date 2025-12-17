# Mobilenetv2

## Platforms:
    Wormhole (n150, n300), Blackhole (p150)

### Introduction
The MobileNetV2 model is a convolutional neural network (CNN) architecture designed for efficient mobile and embedded vision applications. It was introduced in the paper ["MobileNetV2: Inverted Residuals and Linear Bottlenecks"](https://arxiv.org/abs/1801.04381). </br>
The MobileNetV2 model has been pre-trained on the ImageNet dataset and can be used for various tasks such as image classification, object detection, and semantic segmentation. It has achieved state-of-the-art performance on several benchmarks 1 for mobile and embedded vision applications.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- login to huggingface with: `huggingface-cli login` or by setting the token with the command `export HF_TOKEN=<token>`
   - To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens

## How to Run
Find MobileNetV2 instructions for the following device implementations:

- Wormhole: [demos/wormhole/mobilenetv2/README](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/wormhole/mobilenetv2/README.md)

- Blackhole:[demos/blackhole/mobilenetv2/README](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/blackhole/mobilenetv2/README.md)
