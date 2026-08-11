# HunyuanImage-3.0

## Introduction

[HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0) is a state-of-the-art Mixture-of-Experts (MoE) model for text-to-image synthesis (`HunyuanImage3ForCausalMM`, `model_type=hunyuan_image_3_moe`).

This is a native-TTNN implementation tuned for inference on Blackhole Galaxy systems — the full transformer and the VAE decode run on-device.

## Details

HunyuanImage-3.0 is an ~80B-total / ~13B-active MoE: a 32-layer transformer (64 routed experts, top-8, plus 1 shared expert; hidden size 4096; vocab 133,120) paired with a Conv3D VAE. A prompt is rendered to a 1024×1024 image by driving the transformer with a FlowMatch (Euler) denoising loop and classifier-free guidance (CFG), a timestep-conditioned velocity head, and a final VAE decode — entirely on the mesh (the hidden state never leaves the device across the loop).

The render is a deterministic feed-forward: `(prompt, seed, steps, size)` reproduces the same image.

## Performance

Performance is measured in seconds per image at 1024×1024px, warm (resident server; the one-time step-1 kernel compile is amortized across images).

| System        | CFG | SP | TP | Steps | Current Performance |
|---------------|-----|----|----|-------|---------------------|
| Galaxy (8×4)  | 2   | 4  | 8  | 50    | ~29.8 s             |

CFG is batched into a single `bsz=2` forward per step (CFG-parallel). The ~29.8 s is a ~25.8 s denoising loop (~511 ms/step) plus a 4.0 s on-device VAE decode. Correctness: image-step velocity PCC 1.0 vs the host reference; on-device VAE PCC 0.99995.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal)
- Installed [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

1. Visit [HuggingFace](https://huggingface.co/tencent/HunyuanImage-3.0) to access the model weights.
2. Login with your HuggingFace token: `huggingface-cli login`

```bash
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)

# Warm resident render server — builds the 32-layer stack + conv heads once, then renders prompts warm (~29.8 s/image)
HUNYUAN_SP=1 HUNYUAN_CFG_PARALLEL=1 HUNYUAN_ONDEVICE_VAE=1 HUNYUAN_VAE_WARMUP=1 \
HUNYUAN_VAE_AUTOCAST=bf16 HUNYUAN_CCL_LINKS=2 ./python_env/bin/python -m \
  models.demos.vision.generative.hunyuanimage_3_0.demo.warm_render_server

# Steady per-step performance (drops the one-time step-1 compile)
pytest -svq -o timeout=0 models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_host_glue_stage3_perf.py

# Correctness / PCC
pytest -svq -o timeout=0 models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_image3_t2i_e2e_pcc.py
pytest -svq -o timeout=0 models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_vae_decode_pcc.py
```

## Scalability

Runs on the full 32-chip Galaxy (8×4 mesh, `FABRIC_1D`). The transformer is parallelized on three axes:

1. **SP** (sequence parallel) — the token sequence is fractured across the DP mesh axis; the FeedForward/MoE layers execute per shard, and K,V are all-gathered for attention.
2. **TP** (tensor parallel, factor 8) — weights are fractured across the TP axis; AllReduce collectives combine the partial activations.
3. **EP** (expert parallel, factor 8) — the 64 routed experts are distributed across the TP axis.

The VAE decoder runs on-device with a mesh-parallel Conv3D and a distributed reduce-moments GroupNorm (an O(num_groups) scalar all-reduce instead of a full-spatial gather).
