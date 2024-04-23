---

# Vision Transformer (ViT) model

## Introduction

The Vision Transformer (ViT) model was proposed in "An Image is Worth 16x16 Words, Transformers for Image Recognition at Scale".
It’s the first paper that successfully trains a Transformer encoder on ImageNet, attaining very good results compared to familiar convolutional architectures.
https://huggingface.co/docs/transformers/en/model_doc/vit


## How to Run

To run the demo for question answering using the Bloom model, follow these instructions:

-  For Imagenet-21K to test inference accuracy, use the following command to run the demo:

  ```sh
  pytest --disable-warnings  models/demos/grayskull/vit/demo/demo_vit_ttnn_imagenet_inference.py
  ```

-  For the inference overall rutime (end-2-end), use the following command to run the demo:

  ```sh
  pytest --disable-warnings  models/demos/grayskull/vit/demo/demo_vit_ttnn_inference_throughput.py
  ```

-  For running the inference device OPs analysis, use the following command to run the demo:

  ```sh
scripts/build_scripts/build_with_profiler_opt.sh # need build to enable the profiler
./tt_metal/tools/profiler/profile_this.py -n vit -c "pytest --disable-warnings  models/demos/grayskull/vit/demo/demo_vit_ttnn_inference_device_OPs.py"
  ```



## Results

- The Imagenet-21K inference accuracy is 79%
- Model runtime (host end-2-end) is 430 FPS
- Device OPs runtime summation will is 640 FPS

---
