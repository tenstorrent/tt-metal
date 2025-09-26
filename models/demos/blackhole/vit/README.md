# ViT

## Platforms:
    Blackhole (p150)

## Introduction
This demo shows how Vision Transformer Base patch16-224 runs on Blackhole devices.

The Vision Transformer (ViT) model was proposed in "An Image is Worth 16x16 Words, Transformers for Image Recognition at Scale".
It’s the first paper that successfully trains a Transformer encoder on ImageNet, attaining very good results compared to familiar convolutional architectures.
https://huggingface.co/docs/transformers/en/model_doc/vit

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`
- login to huggingface with your token: `huggingface-cli login` or by setting the token with the command `export HF_TOKEN=<token>`
  - To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens

## How to Run
- Use the following command to run the ViT model:
```sh
pytest --disable-warnings models/demos/vit/tests/pcc/test_ttnn_optimized_sharded_vit_bh.py::test_vit
```

### Demo
To run the demo for ViT model, follow these instructions:
-  For overall rutime inference (end-2-end), use the following command to run the demo:
```sh
pytest --disable-warnings models/demos/blackhole/vit/demo/demo_vit_ttnn_inference_perf_e2e_2cq_trace.py
```

-  For inference device OPs analysis, use the following command to run the demo:
```sh
# Need to enable the profiler by building with ./build_metal.sh -p

# To manually inspect ops
./tools/tracy/profile_this.py -n vit -c "pytest --disable-warnings models/demos/blackhole/vit/demo/test_vit_device_perf.py::test_vit_device_ops"

# For an automated device perf report(samples/s)
pytest models/demos/blackhole/vit/demo/test_vit_device_perf.py::test_vit_perf_device
```

## Testing
- Use the following command to run the demo and to test inference accuracy for Imagenet-21K:
```sh
pytest --disable-warnings models/demos/blackhole/vit/demo/demo_vit_performant_imagenet_inference.py::test_run_vit_trace_2cqs_inference
```

## Details
- Entry point for the model is `vit` in `models/demos/vit/tt/ttnn_optimized_sharded_vit_bh.py`
- Batch Size: 10
- Sequence size: 224
- Dataset Used: `ImageNet-21k dataset`.

## Results
- Model runtime (host end-2-end) is `3760` FPS
- The Imagenet-21K inference accuracy is `80%`
