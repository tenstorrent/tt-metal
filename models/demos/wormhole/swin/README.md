## Swin Model

# Platforms:
    GS WH N150, WH N300

## Introduction
The Swin Transformer is a variant of the Vision Transformer that generates hierarchical feature maps by progressively merging image patches in its deeper layers. It achieves linear computational complexity relative to the input image size by restricting self-attention calculations to local windows. This design allows it to function as a versatile backbone for tasks like image classification and dense recognition. In contrast, earlier Vision Transformers generate feature maps at a single low resolution and have quadratic computational complexity, as they compute self-attention across the entire image.

# Details
The entry point to  swin model is swin_for_image_classification in `models/demos/swin/tt/tt/ttnn_optimized_swin.py`. The model picks up certain configs and weights from huggingface pretrained model. We have used `microsoft/swin-tiny-patch4-window7-224` version from huggingface as our reference.


## Batch size: 8

Batch Size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage. It's recommended to set the `batch_size` to 8

Use `pytest --disable-warnings models/demos/wormhole/swin/demo/demo.py::test_demo_imagenet[wormhole_b0-True-8-5-device_params0]` to run the ttnn_optimized_swin demo.


If you wish to run for `n_iterations` samples, use `pytest --disable-warnings models/demos/swin/demo/demo.py::test_demo_imagenet[8-<n_iterations>-device_params0]`
