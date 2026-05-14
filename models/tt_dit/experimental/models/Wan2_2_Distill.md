# Wan2.2-Distill (lightx2v, 4-step I2V)

## Introduction

[lightx2v/Wan2.2-Distill-Models](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)
is a 4-step distillation of [Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers).
It produces image-to-video outputs in roughly 1/10 the denoising steps of the
base model, with classifier-free guidance baked in.

This page documents the tt_dit integration. See [Wan2_2.md](Wan2_2.md) for the
underlying architecture — the distill is weight-only; the model graph is identical.

## Details

The distill ships only the DiT weights as flat `.safetensors` files. The
tokenizer, UMT5 text encoder, VAE, and scheduler are loaded from the base
`Wan-AI/Wan2.2-I2V-A14B-Diffusers` repo.

The two-stage MoE structure is preserved: a high-noise expert handles the first
half of the denoising trajectory and a low-noise expert handles the rest. With
4 inference steps and `boundary_ratio=0.5`, that's 2 steps per expert.

## Files used

From [lightx2v/Wan2.2-Distill-Models](https://huggingface.co/lightx2v/Wan2.2-Distill-Models):

- `wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors` (BF16, 28.6 GB) — high-noise expert
- `wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors` (BF16, 28.6 GB) — low-noise expert

From [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers):
tokenizer, text encoder, VAE, scheduler config.

## Inference defaults

| Parameter | Value | Note |
|---|---|---|
| `num_inference_steps` | 4 | Distill target |
| `boundary_ratio` | 0.5 | 2 high-noise + 2 low-noise |
| `guidance_scale` | 1.0 | CFG baked in — disables negative-prompt path |
| `guidance_scale_2` | 1.0 | Same as above for the low-noise stage |

## Supported configurations (PR1)

| System | Mesh | SP | TP | Topology | Test ID |
|---|---|---|---|---|---|
| BH Galaxy | 4x8 | 8 (axis 1) | 4 (axis 0) | Ring | `bh_4x8sp1tp0_ring` |

Additional mesh shapes will be added in follow-up work.

## How to run

```bash
# Required environment
export TT_DIT_CACHE_DIR=/your/cache/path
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)

# Allow first-time download of the lightx2v safetensors (~57 GB total).
# Skip if you've already downloaded them and put them in the HF cache or
# pointed LIGHTX2V_LOCAL_DIR at a directory that contains both files.
export TT_DIT_ALLOW_HF_DOWNLOAD=1

# Place a seed image at ./prompt_image.png in the working directory.

NO_PROMPT=1 pytest \
  models/tt_dit/experimental/tests/test_pipeline_wan_distill_i2v.py \
  -v -k "bh_4x8sp1tp0_ring and resolution_480p" --timeout 1500
```

To use weights from a custom location:

```bash
export LIGHTX2V_LOCAL_DIR=/path/to/lightx2v/weights
```

## Performance

BH Galaxy 4×8 Ring, 81 frames, single timed iteration (2026-05-08):

| Stage | Base 480p (40 steps, CFG) | Base 720p (40 steps, CFG) | Distill 480p (4 steps, no CFG) | Distill 720p (4 steps, no CFG) |
|---|---:|---:|---:|---:|
| Text encoding (UMT5) | 0.136 s | 0.133 s | 0.134 s | 0.138 s |
| Image encoding (CLIP+VAE encode) | 5.020 s | 13.024 s | 4.949 s | 12.577 s |
| Denoising | 48.806 s | 133.243 s | 2.436 s | 6.553 s |
| VAE decoding | 0.426 s | 0.711 s | 0.424 s | 0.746 s |
| **Total** | **54.4 s** | **147.1 s** | **7.96 s** | **20.0 s** |
| Speedup vs base | — | — | 6.84× | 7.34× |

Per-forward denoise cost is identical between base and distill at each resolution; distill's win is "10× fewer steps × 2× from no-CFG" with kernel performance unchanged.
