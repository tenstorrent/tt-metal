---

# Vision Transformer (ViT) model

## Introduction

The Vision Transformer (ViT) model was proposed in "An Image is Worth 16x16 Words, Transformers for Image Recognition at Scale".
It’s the first paper that successfully trains a Transformer encoder on ImageNet, attaining very good results compared to familiar convolutional architectures.
https://huggingface.co/docs/transformers/en/model_doc/vit


## How to Run

To run the demo for question answering using the Bloom model, follow these instructions:

-  For the inference overall rutime (end-2-end), use the following command to run the demo:

  ```sh
  pytest --disable-warnings models/demos/wormhole/vit/demo/demo_vit_ttnn_inference_perf_e2e_2cq_trace.py
  ```

-  For running the inference device OPs analysis, use the following command to run the demo:

  ```sh
build_metal.sh --enable-profiler # need build to enable the profiler
./tt_metal/tools/profiler/profile_this.py -n vit -c "pytest --disable-warnings models/demos/wormhole/vit/demo/test_vit_device_perf.py::test_vit_device_ops" # to manually inspect ops
pytest models/demos/wormhole/vit/demo/test_vit_device_perf.py::test_vit_perf_device # to get an automated report of samples/s
  ```

-  For Imagenet-21K to test inference accuracy, use the following command to run the demo:

  ```sh
  pytest --disable-warnings models/demos/wormhole/vit/demo/demo_vit_performant_imagenet_inference.py::test_run_vit_trace_2cqs_inference
  ```


## Results

- Model runtime (host end-2-end) is ~1370 FPS
- The Imagenet-21K inference accuracy is 80%

---
