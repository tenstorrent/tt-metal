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
- login to huggingface with your token: `huggingface-cli login` or by setting the token with the command `export HF_TOKEN=<token>`
  - To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens

## How to Run
- Use the following command to run the ViT model:
```sh
pytest --disable-warnings models/demos/blackhole/vit/tests/test_ttnn_optimized_sharded_vit_bh.py::test_vit[224-3-224-10-google/vit-base-patch16-224]
```

### Demo
- Model runtime (end-2-end) is `~3700` FPS (**On P150**):
```sh
pytest --disable-warnings models/demos/blackhole/vit/tests/test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py
```

-  For inference device OPs analysis, use the following command to run the demo:
```sh
# To manually inspect ops
./tools/tracy/profile_this.py -n vit -c "pytest --disable-warnings models/demos/blackhole/vit/tests/test_vit_device_perf.py::test_vit_device_ops"

# For an automated device perf report(samples/s)
pytest models/demos/blackhole/vit/tests/test_vit_device_perf.py::test_vit_perf_device
```

## Testing
- Use the following command to run the demo and to test inference accuracy for Imagenet-21K:
```sh
pytest --disable-warnings models/demos/blackhole/vit/demo/demo_vit_performant_imagenet_inference.py::test_run_vit_trace_2cqs_inference
```

## Details
- Entry point for the model is `vit` in `models/demos/blackhole/vit/tt/ttnn_optimized_sharded_vit_bh.py`
- Batch Size: 10
- Sequence size: 224
- Dataset Used: `ImageNet-21k dataset`.
- The Imagenet-21K inference accuracy is `80%`

---

## High-Resolution ViT (Batch 1)

This folder also contains `ttnn_optimized_sharded_vit_hiRes_bh.py`, which targets a different use case than the standard ViT above.

### When to use which?

| | Standard ViT | High-Res ViT |
|---|---|---|
| **Use case** | Classification at scale | Detailed image analysis |
| **Batch size** | 10+ (throughput-optimized) | 1 (latency-optimized) |
| **Sequence length** | 196 patches (fixed, 14×14 from 224px) | 1024 / 2048 / 3072 (variable) |
| **Hidden dimension** | 768 (fixed) | 512 / 1024 / 1536 / 2304 (variable) |

The standard ViT processes many images in parallel for maximum throughput (images/sec). The high-res variant handles larger sequence lengths from higher resolution inputs, optimized for batch=1 inference where you need more detail per image rather than more images per second.

### Running the High-Res ViT tests

```sh
pytest --disable-warnings models/demos/blackhole/vit/tests/test_ttnn_optimized_sharded_vit_hiRes_bh.py
```

This runs a sweep over sequence sizes (1024, 2048, 3072) and hidden dimensions (512, 1024, 1536, 2304) at batch=1.

### Demo (2CQ Trace Performance)

This Performance test measures e2e model runtime,  measures samples/sec for all 12 configurations:

```sh
pytest --disable-warnings models/demos/blackhole/vit/tests/test_demo_vit_hiRes_ttnn_inference_perf_e2e_2cq_trace.py
```


For a deeper dive into ViT implementation details (sharding strategies, matmul configs, encoder layer breakdown), see the [ViT Tech Report](../../../../tech_reports/ViT-TTNN/vit.md).
