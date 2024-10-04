# LENET

# Platforms:
   E150, WH N300, N150

## Introduction

The LeNet model is a foundational convolutional neural network (CNN) architecture that was specifically developed for handwritten digit recognition on the MNIST dataset. This pioneering model consists of several convolutional layers interspersed with pooling layers, followed by fully connected layers that output the final classification. By utilizing convolutional layers, LeNet effectively captures spatial hierarchies and local patterns in images, leading to significantly enhanced performance compared to traditional, simpler architectures. Its design laid the groundwork for many modern deep learning models used in image classification tasks today.

### Batch size: 8

Batch Size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage. It's recommended to set the batch_size to 8

## How to Run

To run the demo for digit classification using the LeNet model, follow these instructions:

Ensure you have the necessary dependencies installed and that your environment is set up correctly for running the model.

Use the following command to execute the LeNet demo
  ```
  pytest models/demos/lenet/demo/demo.py::test_demo_dataset
  ```
This command will initiate the test for the demo dataset, allowing you to observe the model's performance in classifying handwritten digits


## Inputs

The demo accepts inputs from the MNIST dataset, which consists of a large collection of labeled handwritten digits. The dataset provides a diverse range of examples, enabling the model to learn and generalize effectively. Each input consists of a grayscale image of a handwritten digit, which is processed through the model to produce a predicted classification.

### Owner: [sabira-mcw](https://github.com/sabira-mcw)
