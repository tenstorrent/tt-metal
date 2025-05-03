# MNIST

## Platforms

GS E150, WH N150, WH N300

## Introduction

The MNIST model uses only fully connected linear layers to classify handwritten digits from the MNIST dataset. Despite the absence of convolutional layers, the model efficiently processes the 28x28 pixel images by flattening them into a 1D vector and passing them through multiple linear layers to predict the corresponding digit (0-9). This approach demonstrates how even simpler architectures can be applied for image classification tasks.

### Batch size: 128

Batch Size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage. It's recommended to set the batch_size to 128

## How to Run

To run the demo for digit classification using the MNIST model, follow these instructions:

-  Use the following command to run the MNIST model.
  ```
  pytest models/demos/mnist/demo/demo.py::test_demo_dataset
  ```

## Inputs

The demo receives inputs from respective dataset MNIST.

## Additional Information

If you encounter issues when running the model, ensure that device has support for all required operations.
