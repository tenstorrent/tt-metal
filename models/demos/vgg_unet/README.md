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
[models/demos/wormhole/vgg_unet](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/wormhole/vgg_unet)

### Blackhole P150
[models/demos/blackhole/vgg_unet](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/blackhole/vgg_unet)
