# Mobilenetv2 Model

## Platforms:
    WH N300

## Introduction
The MobileNetV2 model is a convolutional neural network (CNN) architecture designed for efficient mobile and embedded vision applications. It was introduced in the paper ["MobileNetV2: Inverted Residuals and Linear Bottlenecks"](https://arxiv.org/abs/1801.04381). </br>
The MobileNetV2 model has been pre-trained on the ImageNet dataset and can be used for various tasks such as image classification, object detection, and semantic segmentation. It has achieved state-of-the-art performance on several benchmarks 1 for mobile and embedded vision applications.

## Details
The entry point to mobilenetv2 model is MobileNetV2 in `models/experimental/functional_mobilenetv2/tt/ttnn_monilenetv2.py`.

Use the following command to run the model :
`pytest tests/ttnn/integration_tests/mobilenetv2/test_mobilenetv2.py`

Note : The model supports a batch size of 8 for a resolution of 224. If you prefer to use a different batch size, it is recommended to modify the batch_size accordingly in the test file.

### Owner: [Sabira](https://github.com/sabira-mcw)
