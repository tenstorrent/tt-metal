# Alexnet ttnn Implementation


# Platforms:
    GS E150, WH N150, WH N300


## Introduction
AlexNet is a deep learning model designed for image classification tasks. It uses convolutional layers to extract features from images and classify them accurately. Known for introducing ReLU activations, dropout, and GPU acceleration, AlexNet played a key role in advancing deep learning and remains a foundational model in computer vision.


# Details
The entry point to alexnet model is ttnn_alexnet in `models/experimental/functional_alexnet/tt/ttnn_alexnet.py`.

Use `pytest --disable-warnings models/experimental/functional_alexnet/demo/demo.py` to run the ttnn_alexnet demo.

## Inputs

The demo receives input from `huggingface/cats-image` by default.
