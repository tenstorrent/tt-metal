## Squeezenet Demo

## Introduction
SqueezeNet is a lightweight and efficient convolutional neural network (CNN) architecture designed for image classification. The key innovation of SqueezeNet is its ability to achieve AlexNet-level accuracy on ImageNet with far fewer parameters, making it an excellent choice for resource-constrained environments like mobile devices or embedded systems.

# Platforms:
    GS 150, WH N300, N150

## How to Run

Use `pytest --disable-warnings models/demos/squeezenet/demo/demo.py::test_demo_dataset[1-1-device_params0]` to run the demo for imagenet dataset.


If you wish to run for `n_iterations` samples, use 'pytest --disable-warnings models/demos/squeezenet/demo/demo.py::test_demo_dataset[1-n_iterations-device_params0]

# Inputs



Inputs for performance default are provided from `models/demos/squeezenet/demo/dog_image.jpeg`. If you wish you to change the inputs, provide a different path to test_perf.


# Details
The entry point to squeezenet model is image_classification in `models/demos/squeezenet/tt/tt_squeezenet.py`. The model picks up weights from torchvision-models package.

## Batch size: 1

Batch Size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage. It's recommended to set the `batch_size` to 1
