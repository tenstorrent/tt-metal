# Wan2.2 I2V — Stable-Video-Infinity (SVI 2.0 Pro)

## Introduction

`WanPipelineSVI` generates long videos by autoregressively chaining short
Wan2.2 I2V clips. Each clip's first-frame conditioning is the last decoded
frame of the previous clip; the original input image is repeated as an
"anchor" so the model can recover from drift across clips.

SVI is two artefacts together:

1. A **LoRA adapter pair** — one per Wan2.2 MoE expert — trained with
   "Error-Recycling Fine-Tuning" so the model is robust to its own drift
   when chained.
2. A **driver loop** that handles per-clip seeding, anchor frames, motion
   latent splicing across clip boundaries, and overlap-frame trimming on
   the decoded output.

Upstream: [vita-epfl/Stable-Video-Infinity](https://github.com/vita-epfl/Stable-Video-Infinity)
(svi_wan22 branch).

SVI 2.0 Pro extends 2.0 with `prev_last_latent` splicing — the last few
latent frames of the previous clip are written into the next clip's I2V
conditioning at frame positions 1..num_motion_latent, which gives smoother
motion handoff than the last-frame-only approach of 2.0. Setting
`num_motion_latent=0` reduces 2.0 Pro behavior to 2.0.

## Sampling regimes

| Regime | Steps | CFG | Scheduler | Shift | LoRA stack (high / low) |
|---|---|---|---|---|---|
| `python` | 50 | 5.0 | UniPC (flow sigmas) | 5.0 (default) | SVI 1.0 / SVI 1.0 |
| `comfyui` | 6 | 1.5 | **UniPC (flow sigmas) — see note** | 8.0 | LightX2V 1.0 + SVI 0.5 / LightX2V 1.0 + SVI 1.0 |

**Scheduler caveat for `comfyui` regime.** The upstream ComfyUI workflow
uses k-diffusion's `dpm++_sde` sampler with Karras "fixed" sigmas. tt-metal
exposes only `UniPCSolver` and `EulerSolver` in `models/tt_dit/solvers/`
today. We use UniPC with `flow_shift=8` here — same flow-shift, same step
count, same CFG, same LoRA stack, but a different (deterministic) ODE
integrator. Output is functionally equivalent for flow-matching models but
will not bit-match ComfyUI. A native `DPMSolverSDESolver` is tracked as a
follow-up in [../../../../wiki/WAN_SVI.md](../../../../wiki/WAN_SVI.md).

## Weights

### SVI 2.0 Pro LoRA

- **Source:** [vita-video-gen/svi-model](https://huggingface.co/vita-video-gen/svi-model)
- **Files (one per MoE expert):**
  - `SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors`
  - `SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors`
- **Format:** standard diffusers-style LoRA (`lora_A` / `lora_B` pairs).

### Base Wan2.2 components (shared)

- **Source:** [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)

### Download

```bash
python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('/path/to/svi', exist_ok=True)
for f in [
    'SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors',
    'SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors',
]:
    hf_hub_download('vita-video-gen/svi-model', f, local_dir='/path/to/svi')
"
```

### LightX2V LoRA (required for `comfyui` regime only)

- **Source:** [lightx2v/Wan2.2-Distill-Loras](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras)
- **Files:**
  - `wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors`
  - `wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors`

```bash
python -c "
from huggingface_hub import hf_hub_download
for f in [
    'wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors',
    'wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors',
]:
    hf_hub_download('lightx2v/Wan2.2-Distill-Loras', f, local_dir='/path/to/lightx2v')
"
```

## Supported configurations

| System | Mesh | Topology | Test ID |
|---|---|---|---|
| BH Loud Box | 2x4 | Linear | `bh_2x4sp1tp0` |
| BH Galaxy | 4x8 | Ring | `bh_4x8sp1tp0_ring` |

## How to Run

### Python regime (50 steps, slower, no LightX2V needed)

```bash
export TT_DIT_CACHE_DIR=/your/cache/path
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export TT_DIT_ALLOW_HF_DOWNLOAD=1

export SVI_HIGH_PATH=/path/to/svi/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors
export SVI_LOW_PATH=/path/to/svi/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors

export PROMPT_IMAGE=$TT_METAL_HOME/prompt_image.png
export SVI_NUM_CLIPS=4

pytest models/tt_dit/experimental/tests/test_pipeline_wan_svi.py \
  -v -k "bh_2x4sp1tp0 and resolution_480p and python" \
  --timeout 7200 -s
```

### ComfyUI regime (6 steps, LightX2V-accelerated)

```bash
# Same env exports as above, plus the LightX2V LoRA pair:
export LIGHTX2V_HIGH_PATH=/path/to/lightx2v/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors
export LIGHTX2V_LOW_PATH=/path/to/lightx2v/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors

pytest models/tt_dit/experimental/tests/test_pipeline_wan_svi.py \
  -v -k "bh_2x4sp1tp0 and resolution_480p and comfyui" \
  --timeout 3600 -s
```

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `SVI_HIGH_PATH` | Yes | — | Path to high-noise expert SVI LoRA `.safetensors` |
| `SVI_LOW_PATH` | Yes | — | Path to low-noise expert SVI LoRA |
| `LIGHTX2V_HIGH_PATH` | Yes for `comfyui` | — | Path to high-noise expert LightX2V LoRA |
| `LIGHTX2V_LOW_PATH` | Yes for `comfyui` | — | Path to low-noise expert LightX2V LoRA |
| `SVI_NUM_CLIPS` | No | 2 | Number of 81-frame clips to chain |
| `SVI_NUM_MOTION_LATENT` | No | 1 | Latent frames of prev clip to splice into next clip's conditioning |
| `SVI_NUM_OVERLAP_FRAME` | No | 4 | Decoded pixel frames to drop on concat between clips |
| `SVI_SEED` | No | 0 | Base seed; clip N uses `base + 42 × N` |
| `NUM_STEPS` | No | 50 (`python`) / 6 (`comfyui`) | Denoising steps per clip |
| `GUIDANCE_SCALE` | No | 5.0 (`python`) / 1.5 (`comfyui`) | CFG scale for the high-noise expert |
| `GUIDANCE_SCALE_2` | No | matches `GUIDANCE_SCALE` | CFG scale for the low-noise expert |
| `PROMPT_IMAGE` | No | `./prompt_image.png` | Seed image (acts as anchor for every clip) |
| `PROMPT` | No | golden retriever prompt | Text prompt |

## Limitations / open items

- **ComfyUI regime not implemented yet** — requires a new device-side
  `DPMSolverSDESolver` (k-diffusion `sample_dpmpp_sde` semantics with
  Karras `get_sigmas` sampling). Tracked in
  [../../../../wiki/WAN_SVI.md](../../../../wiki/WAN_SVI.md).
- **Anchor handling is pixel-space pre-fill**, not latent-space tiling like
  upstream `WanVideoUnit_ImageEmbedderVAE`. The temporal VAE encodes the
  anchor-everywhere video and frame 0 dominates latent frame 0 due to the
  1+4N temporal pattern. In practice this produces equivalent conditioning
  but is not bit-equivalent to upstream's latent-space tile.
- **CPU-side LoRA fusion** (~5–10 s per expert on first clip) — same
  semantics as static LoRA. Switching between LoRA stacks across runs
  rebuilds the TT cache; runtime LoRA scale tuning or mid-clip LoRA
  switching is not supported (would require a device-side `LoRALinear`).
- **Per-clip latency** scales linearly with `num_clips`. Measured on BH-LB
  (2x4) at 480p: `python` regime ~3:50/clip (50 steps); `comfyui` regime
  ~42 sec/clip (6 steps, LightX2V-accelerated).
- `bh_2x4sp1tp0` (BH-LB) is the primary tested config; `bh_4x8sp1tp0_ring`
  (Galaxy) is parameterized but not yet exercised.
