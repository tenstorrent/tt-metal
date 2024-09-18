#  Effective Squeeze-Excitation Variety of View Network (ESE-VoVNet) model

## Introduction

VoVNet is a convolutional neural network that seeks to make DenseNet more efficient by concatenating all features only once in the last feature map, which makes input size constant and enables enlarging new output channel.

To solve the inefficiency of DenseNet, VoVNet model is proposed with energy and computation efficient architecture comprised of One-Shot Aggregation (OSA). The OSA not only adopts the strength of DenseNet that represents diversified features with multi receptive fields but also overcomes the inefficiency of dense connection by aggregating all features only once in the last feature maps.
https://arxiv.org/abs/1904.09730


## Details

The entry point to the Functional VoVNet model is the vovnet function located in ttnn_functional_vovnet.py. The `hf_hub:timm/ese_vovnet19b_dw.ra_in1k` version from Hugging Face is used as the reference model.


## How to Run

To run the demo for image classification using the VoVNet model, follow these instructions:

-  Use the following command to run the VoVNet model using ttnn_functional_vovnet:
  ```
  pytest --disable-warnings --input-path="models/experimental/functional_vovnet/demo/dataset" models/experimental/functional_vovnet/demo/demo.py::test_demo
  ```
-  If you wish to run the demo with different input samples, replace <address_to_your_inputs_folder> with the path for your inputs folder in the following command:

  ```
  pytest --disable-warnings --input-path="<address_to_your_inputs_folder>" models/experimental/functional_vovnet/demo/demo.py::test_demo

  ```
-  Our second demo is designed to run VoVNet using ImageNet-1k Validation Dataset, run this with the following command:
  ```
  pytest --disable-warnings models/experimental/functional_vovnet/demo/demo.py::test_demo_imagenet_1k
  ```
