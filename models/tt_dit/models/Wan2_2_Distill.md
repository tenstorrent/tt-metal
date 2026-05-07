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
  models/tt_dit/tests/models/wan2_2/test_pipeline_wan_distill_i2v.py \
  -v -k "bh_4x8sp1tp0_ring and resolution_480p" --timeout 1500
```

To use weights from a custom location:

```bash
export LIGHTX2V_LOCAL_DIR=/path/to/lightx2v/weights
```

## Limitations / open items

- Scheduler: the pipeline currently uses the base UniPC scheduler with the
  config from `Wan-AI/Wan2.2-I2V-A14B-Diffusers`. lightx2v's reference may use
  a different schedule (LCM / flow-match); if output quality lags the
  reference, this is the first place to look.
- `boundary_ratio=0.5` assumes a symmetric 2-high / 2-low split at 4 steps.
  Verify against the lightx2v reference inference example.
- Only `bh_4x8sp1tp0_ring` is exercised in PR1.
- Only the BF16 4-step variant is supported; FP8 / INT8 variants from the same
  HF repo would require a quantization integration in tt_dit.
