# Index-AniSora V3.2 (I2V)

## Introduction

[IndexTeam/Index-anisora](https://huggingface.co/IndexTeam/Index-anisora) V3.2
is an anime-domain image-to-video model derived from
[Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers).
It preserves the two-expert (high-noise / low-noise) MoE structure of Wan2.2
and ships a finetuned weight pair specialized for anime-style motion.

This page documents the tt_dit integration. See [Wan2_2.md](Wan2_2.md) for the
underlying architecture — AniSora is a weight-only swap; the model graph is
identical.

## Details

The HF repo ships only the DiT experts under `V3.2/high_noise_model/` and
`V3.2/low_noise_model/` as flat `diffusion_pytorch_model.safetensors` files.
The tokenizer, UMT5 text encoder, VAE, and scheduler are loaded from the base
`Wan-AI/Wan2.2-I2V-A14B-Diffusers` repo.

The two-stage MoE schedule is preserved: the high-noise expert handles the
first 90% of the denoising trajectory and the low-noise expert handles the
final 10% (`boundary_ratio=0.9`).

The AniSora safetensors use the original-Wan key naming
(`blocks.X.self_attn.q.weight`, etc), identical to the lightx2v distill
checkpoints. The same `wan_lightx2v_to_diffusers_key` rename function in
`models/tt_dit/utils/lightx2v_loader.py` is reused — no new key remap is
needed.

## Files used

From [IndexTeam/Index-anisora](https://huggingface.co/IndexTeam/Index-anisora):

- `V3.2/high_noise_model/diffusion_pytorch_model.safetensors` — high-noise expert
- `V3.2/low_noise_model/diffusion_pytorch_model.safetensors` — low-noise expert

From [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers):
tokenizer, text encoder, VAE, scheduler config.

## Inference defaults

Mirrors `anisoraV3.2/wan/configs/wan_i2v_A14B.py`:

| Parameter | Value | Note |
|---|---|---|
| `num_inference_steps` | 40 | UniPC sampler |
| `boundary_ratio` | 0.9 | High-noise covers first 90% of timesteps |
| `guidance_scale` | 3.5 | High-noise stage CFG |
| `guidance_scale_2` | 3.5 | Low-noise stage CFG |
| `sample_shift` | 5.0 | Inherited from base Wan2.2 scheduler config |

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

# Allow first-time download of the AniSora safetensors (~57 GB total).
# Skip if you've already downloaded them and put them in the HF cache or
# pointed ANISORA_LOCAL_DIR at a directory containing V3.2/{high,low}_noise_model/.
export TT_DIT_ALLOW_HF_DOWNLOAD=1

# Place a seed image at ./prompt_image.png in the working directory.

NO_PROMPT=1 pytest \
  models/tt_dit/tests/models/wan2_2/test_pipeline_anisora.py \
  -v -k "bh_4x8sp1tp0_ring and resolution_480p" --timeout 1500
```

To use weights from a custom location:

```bash
# Layout under this directory must mirror the HF repo:
#   $ANISORA_LOCAL_DIR/V3.2/high_noise_model/diffusion_pytorch_model.safetensors
#   $ANISORA_LOCAL_DIR/V3.2/low_noise_model/diffusion_pytorch_model.safetensors
export ANISORA_LOCAL_DIR=/path/to/index-anisora-weights
```

## Limitations / open items

- Scheduler: the pipeline uses the base `UniPCMultistepScheduler` config from
  `Wan-AI/Wan2.2-I2V-A14B-Diffusers`. AniSora's reference inference uses
  `FlowUniPCMultistepScheduler` with `sample_shift=5.0`; if output quality
  diverges from the upstream sample, this is the first place to look.
- Only `bh_4x8sp1tp0_ring` is exercised in PR1.
- `boundary_ratio=0.9` matches the upstream `boundary=0.900` constant. The
  effective high/low step split at 40 steps is 36 / 4.
