# MobilenetV2
The MobileNetV2 model is a convolutional neural network (CNN) architecture designed for efficient mobile and embedded vision applications. It was introduced in the paper ["MobileNetV2: Inverted Residuals and Linear Bottlenecks"](https://arxiv.org/abs/1801.04381). </br>
The MobileNetV2 model has been pre-trained on the ImageNet dataset and can be used for various tasks such as image classification, object detection, and semantic segmentation. It has achieved state-of-the-art performance on several benchmarks 1 for mobile and embedded vision applications.

## How to Run

To run the demo, make sure to build the project, activate the environment, and set the appropriate environment variables.
For more information, refer [installation and build guide](https://docs.tenstorrent.com/tt-metalium/latest/get_started/get_started.html#install-and-build).

To run the functional Mobilenetv2 model on a single-chip:
```sh
pytest --disable-warnings models/experimental/functional_mobilenetv2/test/test_ttnn_mobilenetv2.py
```

To run the functional Mobilenetv2 model on a single-chip:
```sh
pytest --disable-warnings models/experimental/functional_mobilenetv2/demo/demo.py
```

## Supported Hardware
- N150

## Other Details

- Inputs by default are random data in test_ttnn_mobilenetv2 and images can be fed as input in demo.py.
- The model weights will be automatically downloaded from Google Drive using wget implemented in weights_download.sh.
