# Mobilenetv2 Model

## Platforms:
    WH N300

## Introduction
The MobileNetV2 model is a convolutional neural network (CNN) architecture designed for efficient mobile and embedded vision applications. It was introduced in the paper ["MobileNetV2: Inverted Residuals and Linear Bottlenecks"](https://arxiv.org/abs/1801.04381). </br>
The MobileNetV2 model has been pre-trained on the ImageNet dataset and can be used for various tasks such as image classification, object detection, and semantic segmentation. It has achieved state-of-the-art performance on several benchmarks 1 for mobile and embedded vision applications.

## Details
The entry point to mobilenetv2 model is MobileNetV2 in `models/experimental/functional_mobilenetv2/tt/ttnn_monilenetv2.py`.

Use the following command to run the model :
` pytest -k "pretrained_weight_true" tests/ttnn/integration_tests/mobilenetv2/test_ttnn_mobilenetv2.py`

## Batch size: 1

Batch Size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage. It's recommended to set the `batch_size` to 1.

Use the following command to run the mobilenetv2 demo:
`pytest -k "pretrained_weight_true" models/experimental/functional_mobilenetv2/demo/demo.py`

#### Note: The post-processing is performed using PyTorch. and also note that the first time the model is run, you need to login to huggingface using your token: `huggingface-cli login` or by setting the token with the command `export HF_TOKEN=<token>`
- To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens

## Inputs

The demo receives inputs from Imagenet dataset by default.


## Additional Information:

The tests can be run with  randomly initialized weights and pre-trained real weights.  To use the pre-trained weights, specify pretrained_weight_true when running the tests.

### Owner: [Sabira](https://github.com/sabira-mcw)
