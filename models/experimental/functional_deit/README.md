# Data-efficient image Transformers (DeiT) model

## Introduction

The  Data-efficient image Transformers (DeiT) model was proposed in "Training data-efficient image transformers & distillation through attention".
The Vision Transformer (ViT) introduced in Dosovitskiy et al., 2020 has shown that one can match or even outperform existing convolutional neural networks using a Transformer encoder (BERT-like). However, the ViT models introduced in that paper required training on expensive infrastructure for multiple weeks, using external data. DeiT (data-efficient image transformers) are more efficiently trained transformers for image classification, requiring far less data and far less computing resources compared to the original ViT models.
https://huggingface.co/docs/transformers/en/model_doc/deit


## Details

The entry point to the Functional DeiT model is the deit function located in ttnn_optimized_sharded_deit.py. The "facebook/deit-base-distilled-patch16-224" version from Hugging Face is utilized as the reference model.


## How to Run

To run the demo for image classification with teacher using the DeiT model, follow these instructions:

-  Use the following command to run the DeiTForImageClassificationWithTeacher using ttnn_optimized_sharded_deit:
  ```
  pytest --disable-warnings --input-path="models/experimental/functional_deit/demo/input_samples" models/experimental/functional_deit/demo/demo_with_teacher.py::test_demo
  ```
-  If you wish to run the demo with different input samples, replace <address_to_your_inputs_folder> with the path for your inputs folder in the following command:

  ```
  pytest --disable-warnings --input-path="<address_to_your_inputs_folder>" models/experimental/functional_deit/demo/demo_with_teacher.py::test_demo

  ```
-  Our second demo is designed to run DeiTForImageClassificationWithTeacher using ImageNet-1k Validation Dataset, run this with the following command:
  ```
  pytest --disable-warnings models/experimental/functional_deit/demo/demo_with_teacher.py::test_demo_imagenet_1k
  ```


## Results

- The Imagenet-1K inference accuracy is 80%
