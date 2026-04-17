# Wan2.2-A14B

## Introduction

[Wan2.2](https://huggingface.co/Wan-AI) is a state-of-the-art open-weights video generative model supporting both text-to-video (T2V) and image-to-video (I2V) generation.

This model is implemented in the TT-DiT library to enable inference on Wormhole and Blackhole multi-chip systems.

## Details

The architecture is described in the paper [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314).

The model consists of a text encoder ([UMT5-XXL](https://huggingface.co/google/umt5-xxl)), a scheduler, two WanTransformer3DModel transformers, and a VAE. Each transformer has 40 blocks of self-attention, cross-attention, and feedforward layers, with RoPE positional embeddings.

Wan2.2 uses a Mixture-of-Experts (MoE) architecture with two-stage denoising. A high-noise expert handles the first 87.5% of timesteps and a low-noise expert handles the remainder, giving 27B total parameters with 14B active per step. Classifier-free guidance (CFG) is used with separate guidance scales per stage.

## Performance

Current T2V (text-to-video) performance for supported systems is detailed below. Performance is measured in total seconds per video, with 81 frames and 40 denoising steps.

### 480p (832x480)

| System           | Arch | SP | TP | Current Performance |
|------------------|------|----|----|---------------------|
| Loud Box (2x4)   | WH   | 2  | 4  | 735s                |
| Quiet Box (2x2)  | BH   | 2  | 2  | 466s                |
| Loud Box (2x4)   | BH   | 4  | 2  | 207s                |

### 720p (1280x720)

| System           | Arch | SP | TP | Current Performance |
|------------------|------|----|----|---------------------|
| Galaxy (4x8)     | WH   | 8  | 4  | 354s                |
| Galaxy (4x8)     | BH   | 8  | 4  | 168s                |

Performance work is ongoing to improve these numbers:
- increased matmul utilization
- increased SDPA utilization
- overlapped AllGather-Matmul and Matmul-ReduceScatter
- fused binary ops
- overlapped weight AllGather with compute

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

```bash
# [Install tt-metal](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

# Set the directory to cache the weights to speed up future runs
export TT_DIT_CACHE_DIR=/your/cache/path

# Text-to-Video (T2V)

# Run T2V on Blackhole Quiet Box (2x2 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py -k "2x2sp0tp1"

# Run T2V on Wormhole Loud Box (2x4 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py -k "2x4sp0tp1"

# Run T2V on Blackhole Loud Box (2x4 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_performance_wan.py -k "bh_2x4sp1tp0"

# Run T2V on Wormhole Galaxy (4x8 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py -k "wh_4x8sp1tp0"

# Run T2V on Blackhole Galaxy (4x8 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py -k "bh_4x8sp1tp0"

# Image-to-Video (I2V)

# Run I2V on Blackhole Quiet Box (2x2 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_i2v.py -k "2x2sp0tp1"

# Run I2V on Wormhole Loud Box (2x4 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_i2v.py -k "2x4sp0tp1"

# Run I2V on Blackhole Loud Box (2x4 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_i2v.py -k "bh_2x4sp1tp0"

# Run I2V on Wormhole Galaxy (4x8 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_i2v.py -k "wh_4x8sp1tp0"

# Run I2V on Blackhole Galaxy (4x8 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_i2v.py -k "bh_4x8sp1tp0"
```

## Scalability

Wan2.2 has been implemented to support execution on the following systems:

**Wormhole:**
- 8-chip (Loud Box with 2x4 mesh topology)
- 32-chip (Galaxy with 4x8 mesh topology)

**Blackhole:**
- 4-chip (Quiet Box with 2x2 mesh topology)
- 8-chip (Loud Box with 2x4 mesh topology)
- 32-chip (Galaxy with 4x8 mesh topology)

The DiT model can be parallelized on 2 main axes:
1. `sp` (sequence parallel) - the input sequence is fractured across a mesh axis. Self-attention is implemented with ring attention, overlapping KV all-gather with computation.
2. `tp` (tensor parallel) - weights are fractured across a mesh axis. CCLs such as AllGather and ReduceScatter are used to gather and scatter activations.

A parallel config is defined by the mesh shape and the axis assignments for `sp` and `tp`. For example, on a 2x4 mesh with `sp_axis=0, tp_axis=1`, we get `sp` parallelism with factor 2 on axis 0 and `tp` factor 4 on axis 1. On a 4x8 mesh with `sp_axis=1, tp_axis=0`, we get `sp` factor 8 on axis 1 and `tp` factor 4 on axis 0.

The text encoder (UMT5) is parallelized with tensor parallelism. The VAE is parallelized with height/width spatial parallelism across the mesh.

## Model Variants

### Text-to-Video (T2V)
- Generates video from a text prompt and an optional negative prompt
- Uses [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) checkpoint
- Supports 480p (832x480) and 720p (1280x720) resolutions
- Default: 81 frames, 40 denoising steps

### Image-to-Video (I2V)
- Generates video conditioned on one or more input images and a text prompt
- Uses [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) checkpoint
- Input images are encoded through the VAE encoder and concatenated with the latent noise
- Supports the same resolutions and frame counts as T2V

Both variants use the same MoE two-stage denoising architecture with separate high-noise and low-noise expert transformers.

## Limitations

While output videos look good, we have many items of work in progress to improve correctness.
Performance optimization is in progress.
