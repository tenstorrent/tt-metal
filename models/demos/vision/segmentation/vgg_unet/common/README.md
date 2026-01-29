# VGG Unet

## Platforms:
    Wormhole (n150, n300), Blackhole (p150)

## Introduction
The VGG-UNet model performs brain tumor segmentation on MRI images. It takes an MRI scan as input and outputs a pixel-wise mask that highlights the regions where a tumor is present. In simple terms, it automatically identifies and outlines brain tumors in medical images to assist doctors in diagnosis and treatment planning.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run
Find the model instructions for each device below:

### Wormhole N150, N300
[models/demos/vision/segmentation/vgg_unet/wormhole](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/vision/segmentation/vgg_unet/wormhole)

### Blackhole P150
[models/demos/vision/segmentation/vgg_unet/blackhole](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/vision/segmentation/vgg_unet/blackhole)
