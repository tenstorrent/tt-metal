# OpenPDN Mnist
The OpenPDN Mnist model is a Convolutional Neural Network (CNN) designed for image classification. It takes an image input, passes it through several convolutional layers with pooling, then flattens the output and passes it through fully connected layers designed to classify images of handwritten digits (0-9). </br>

## How to Run

To run the demo, make sure to build the project, activate the environment, and set the appropriate environment variables.
For more information, refer [installation and build guide](https://docs.tenstorrent.com/tt-metalium/latest/get_started/get_started.html#install-and-build).

If running on Wormhole N300, the following environment variable needs to be set:

```sh
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

To run the functional OpenPDN Mnist model on a single-chip with random weights:
```sh
pytest tests/ttnn/integration_tests/openpdn_mnist/test_ttnn_openpdn_mnist.py
```
## Supported Hardware
- N150
## Other Details
- Inputs by default are random data.
