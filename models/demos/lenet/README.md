# LENET

## Introduction

The LeNet model is a pioneering convolutional neural network architecture designed for handwritten digit recognition on the MNIST dataset. This model comprises multiple convolutional and pooling layers, followed by fully connected layers. By leveraging convolutional layers, LeNet effectively captures spatial hierarchies in images, resulting in improved performance compared to simpler architectures.

### Batch size: 8

Batch Size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage. It's recommended to set the batch_size to 8

## How to Run

To run the demo for digit classification using the MNIST model, follow these instructions:

-  Use the following command to run the LENET model.
  ```
  pytest models/demos/lenet/demo/demo.py::test_demo_dataset
  ```

## Inputs

The demo receives inputs from respective dataset MNIST.

### Owner: [sabira-mcw](https://github.com/sabira-mcw)
