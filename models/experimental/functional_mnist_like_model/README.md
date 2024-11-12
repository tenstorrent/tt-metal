# MNIST-like
The MNIST-like model is a Convolutional Neural Network (CNN) designed for image classification. It takes an image input, passes it through several convolutional layers with pooling, then flattens the output and passes it through fully connected layers designed to classify images of handwritten digits (0-9). </br>


## How to Run

To run the demo, make sure to build the project, activate the environment, and set the appropriate environment variables.
For more information, refer [installation and build guide](https://docs.tenstorrent.com/tt-metalium/latest/get_started/get_started.html#install-and-build).

To run the functional Mnist-like model on a single-chip:
```sh
pytest models/experimental/functional_mnist_like_model/test/test_ttnn_mnist_like_model.py
```

## Supported Hardware

- N150
## Other Details

- Inputs by default are random data.
