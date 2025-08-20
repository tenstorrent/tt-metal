# Convnet Mnist

## Platforms:
    Grayskull (e150), Wormhole (n150, n300)

# Introduction
Convnet Mnist implements a Convolutions to classify handwritten digits from the MNIST dataset. The MNIST dataset contains grayscale images of handwritten digits (0-9), each of size 32x32 pixels.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run
To run the demo for digit classification using the MNIST model, use the following command:
```
pytest models/demos/convnet_mnist/demo/demo.py
```

## Details
[Maxpool](https://github.com/tenstorrent/tt-metal/issues/12642) and [softmax](https://github.com/tenstorrent/tt-metal/issues/12664) are used in torch inside the model.

**Model Owner**: [vigneshkumarkeerthivasan](https://github.com/vigneshkeerthivasanx)
