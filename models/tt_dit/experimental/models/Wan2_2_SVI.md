# Wan2.2 I2V — Stable-Video-Infinity (SVI 2.0 Pro)

## Introduction

`WanPipelineSVI` generates long videos by autoregressively chaining short
Wan2.2 I2V clips. Per upstream's
[SVI 2.0 Pro doc](https://github.com/vita-epfl/Stable-Video-Infinity/blob/svi_wan22/docs/svi/svi_2.0_pro.md),
the conditioning latent for each clip is built as

```
y = concat([anchor_latent, motion_latent, padding], dim=temporal)
```

- **`anchor_latent`** — the VAE-encoded user-provided first frame, repeated
  across every clip.
- **`motion_latent`** — the last `num_motion_latent` latent frames of the
  previous clip's denoised output (`None` on the first clip).
- **`padding`** — literal-zero latents for the remaining frame slots.

The original SVI 2.0 last-frame handoff is explicitly **not** used in 2.0
Pro — the decoded last frame of the previous clip is never re-encoded.
Setting `num_motion_latent=0` collapses 2.0 Pro back to 2.0 behavior (anchor
only, no latent handoff).

SVI ships as a LoRA adapter pair (one per Wan2.2 MoE expert) trained with
"Error-Recycling Fine-Tuning" so the model is robust to its own drift when
chained. See [Wan2_2_LoRA.md](Wan2_2_LoRA.md) for the underlying LoRA
fusion pipeline this builds on.

Upstream: [vita-epfl/Stable-Video-Infinity](https://github.com/vita-epfl/Stable-Video-Infinity)
(svi_wan22 branch).

## Sampling regimes

| Regime | Steps | CFG | Scheduler | flow_shift | LoRA stack (high / low) |
|---|---|---|---|---|---|
| `python` | 50 | 5.0 | `FlowMatchEulerDiscreteScheduler` → `EulerSolver` | 5.0 | SVI 1.0 / SVI 1.0 |
| `comfyui` | 6 | 1.5 | `UniPCMultistepScheduler(use_flow_sigmas=True)` → `UniPCSolver` *(see note)* | 8.0 | LightX2V 0.5 + SVI 1.0 / LightX2V 1.0 + SVI 1.0 |

The `python` regime matches upstream's `inference_svi_2.0_pro.py` —
diffsynth's `FlowMatchScheduler("Wan")` is a plain Euler step on a
flow-matching schedule, so `FlowMatchEulerDiscreteScheduler` + `EulerSolver`
is semantically equivalent.

**Scheduler caveat for `comfyui` regime.** The upstream ComfyUI workflow
(`comfyui_workflow/SVI-Wan22-1210-{4,10}-Clips.json`) uses k-diffusion's
`dpm++_sde` sampler. tt-metal has no on-device port of that stochastic
2nd-order singlestep solver, so we substitute UniPC at the same
`flow_shift=8`. Both are order-2 flow-aware solvers; output is visually
close but not bit-exact to ComfyUI. The 4-clip variant additionally uses
`cfg=2.0` for clips 1-2 and `cfg=1.5` for clips 3-4 — pass
`GUIDANCE_SCALE="2.0||2.0||1.5||1.5"` (and matching `SVI_NUM_CLIPS=4`) to
match that pattern; the 10-clip variant uses uniform `cfg=1.5` which is the
default.

## Weights

### SVI 2.0 Pro LoRA

- **Source:** [vita-video-gen/svi-model](https://huggingface.co/vita-video-gen/svi-model)
- **Files (one per MoE expert):**
  - `SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors`
  - `SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors`
- **Format:** PEFT-style — keys are `<base>.lora_A.default.weight` /
  `<base>.lora_B.default.weight` (the trailing `.default` adapter-name
  segment is tolerated by the loader; see
  [Wan2_2_LoRA.md](Wan2_2_LoRA.md#supported-adapter-key-formats)).

### LightX2V LoRA (required for `comfyui` regime only)

- **Source:** [lightx2v/Wan2.2-Distill-Loras](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras)
- **Files:**
  - `wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors`
  - `wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors`

### Download

```bash
python -c "
from huggingface_hub import hf_hub_download
for f in [
    'SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors',
    'SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors',
]:
    hf_hub_download('vita-video-gen/svi-model', f, local_dir='/path/to/svi')
for f in [
    'wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors',
    'wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors',
]:
    hf_hub_download('lightx2v/Wan2.2-Distill-Loras', f, local_dir='/path/to/lightx2v')
"
```

### Base Wan2.2 components (shared)

- **Source:** [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)

## Supported configurations

| System | Mesh | Topology | Test ID |
|---|---|---|---|
| BH Loud Box | 2x4 | Linear | `bh_2x4sp1tp0` |
| BH Galaxy | 4x8 | Ring | `bh_4x8sp1tp0_ring` |

## How to Run

### ComfyUI regime (6 steps, LightX2V-accelerated — recommended)

```bash
export TT_DIT_CACHE_DIR=/your/cache/path
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export TT_DIT_ALLOW_HF_DOWNLOAD=1

export SVI_HIGH_PATH=/path/to/svi/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors
export SVI_LOW_PATH=/path/to/svi/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors
export LIGHTX2V_HIGH_PATH=/path/to/lightx2v/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors
export LIGHTX2V_LOW_PATH=/path/to/lightx2v/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors

export PROMPT_IMAGE=$TT_METAL_HOME/prompt_image.png
export SVI_NUM_CLIPS=10

pytest models/tt_dit/experimental/tests/test_pipeline_wan_svi.py \
  -v -k "bh_2x4sp1tp0 and resolution_480p and comfyui" \
  --timeout 3600 -s
```

### Python regime (50 steps, no LightX2V needed)

Same `SVI_HIGH_PATH` / `SVI_LOW_PATH` exports as above; omit the LightX2V
ones. Replace the `-k` selector with `... and python`. Expect roughly 6×
the wall time per clip.

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `SVI_HIGH_PATH` | Yes | — | Path to high-noise expert SVI LoRA `.safetensors` |
| `SVI_LOW_PATH` | Yes | — | Path to low-noise expert SVI LoRA |
| `LIGHTX2V_HIGH_PATH` | Yes for `comfyui` | — | High-noise expert LightX2V LoRA |
| `LIGHTX2V_LOW_PATH` | Yes for `comfyui` | — | Low-noise expert LightX2V LoRA |
| `SVI_NUM_CLIPS` | No | 2 | Number of 81-frame clips to chain |
| `SVI_NUM_MOTION_LATENT` | No | 1 | Latent frames of prev clip to splice into next clip's conditioning |
| `SVI_NUM_OVERLAP_FRAME` | No | 4 | Decoded pixel frames to drop on concat between clips |
| `SVI_SEED` | No | 0 | Base seed; clip N uses `base + 42 × N` |
| `NUM_STEPS` | No | 50 (`python`) / 6 (`comfyui`) | Denoising steps per clip |
| `GUIDANCE_SCALE` | No | 5.0 (`python`) / 1.5 (`comfyui`) | CFG for high-noise expert; `\|\|`-separated list for per-clip override |
| `GUIDANCE_SCALE_2` | No | matches `GUIDANCE_SCALE` | CFG for low-noise expert; same `\|\|` syntax |
| `PROMPT_IMAGE` | No | `./prompt_image.png` | Seed image (acts as anchor for every clip) |
| `PROMPT` | No | golden retriever prompt | Text prompt; `\|\|`-separated list for per-clip prompts |

## Limitations / open items

- **`comfyui` regime is not bit-exact to upstream** — see the scheduler
  caveat above. A native `DPMSolverSDESolver` is the follow-up that would
  close the gap.
- **Anchor encoding differs slightly from upstream.** Our implementation
  reuses `WanPipelineI2V.prepare_latents`, which encodes a full anchor-at-
  frame-0 / zero-elsewhere video and overwrites the non-anchor frames with
  the motion-latent splice and zero-padding. Upstream's `wan_video_svi_pro`
  encodes only the single anchor frame and concatenates. Both produce the
  same anchor latent at frame 0 and zeros / motion at the rest, but the
  encode path is not byte-identical.
- **CPU-side LoRA fusion** (~5–10 s per expert at cold cache, scales with
  stack depth). Switching between LoRA stacks across runs rebuilds the TT
  cache; runtime LoRA scale tuning or mid-clip switching is not supported.
- **Per-clip latency** (BH-LB 2x4, 480p, with anchor VAE-encode cache hit
  after clip 1): `comfyui` regime ~63 s/clip; `python` regime not yet
  measured end-to-end on this branch.
- `bh_2x4sp1tp0` (BH-LB) is the primary tested config; `bh_4x8sp1tp0_ring`
  (Galaxy) is parameterized but not yet exercised.
