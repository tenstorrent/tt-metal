# Introduction

Convnet Mnist implements a Convolutions to classify handwritten digits from the MNIST dataset. The MNIST dataset contains grayscale images of handwritten digits (0-9), each of size 32x32 pixels.

# Platforms:
    GS E150, WH N150, WH N300

## How to Run

To run the demo for digit classification using the MNIST model, follow these instructions:

- Use the following command to run the MNIST model.

```
pytest models/demos/convnet_mnist/demo/demo.py
```

Maxpool and Softmax are used in torch inside the model.
ISSUES:
 #12664 - [softmax](https://github.com/tenstorrent/tt-metal/issues/12664)
 #12642 - [maxpool](https://github.com/tenstorrent/tt-metal/issues/12642)


### Owner: [vigneshkumarkeerthivasan](https://github.com/vigneshkeerthivasanx)
