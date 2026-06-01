# LTX-2.3

## Introduction

[LTX-2](https://github.com/Lightricks/LTX-2) is a joint audio-video diffusion transformer from Lightricks. The TT-DiT port supports text-to-video (video-only) and text-to-audio-video (AV) generation on Wormhole and Blackhole multi-chip systems.

Clone the reference repo beside `tt-metal` (not vendored as a submodule):

```bash
git clone https://github.com/Lightricks/LTX-2.git LTX-2
```

## Details

LTX-2.3 Pro uses a 22B diffusion transformer (48 layers, 4096 dim, 128-channel latents) with Gemma-3 text encoding, full MultiModalGuider guidance (CFG + STG + modality), and a causal 3D VAE. The distilled Fast variant uses a 19B checkpoint with a 2-stage half-res → upsample → full-res flow (~11 denoising steps total).

## Performance

Current AV Pro performance for supported systems. Measured at 512×768, 121 frames, 30 denoising steps unless noted. Denoise time excludes Gemma text encoding (~2–3 min on CPU).

### 512p (512×768)

| System | Arch | SP | TP | Denoise | s/step | Notes |
|--------|------|----|----|---------|--------|-------|
| Loud Box (2×4) | WH | 2 | 4 | ~795s | ~20–29s | 4 guidance passes/step; host sync overhead |
| Galaxy (4×8) | BH | 8 | 4 | ~843s | ~28s | Ring topology, `FABRIC_1D_RING` |

Performance work is ongoing:
- device-resident denoise loop (latents, Euler step, CFG on device)
- batched/fused MultiModalGuider passes
- tuned matmul and SDPA blocking
- on-device Gemma encoding
- Conv3D blocking sweep for VAE decode

See `models/tt_dit/models/transformers/ltx/BRINGUP.md` for bringup history, correctness decisions, and optimization backlog.

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal)
- Cloned [LTX-2](https://github.com/Lightricks/LTX-2) at `LTX-2/` (for reference text encoding and audio decode)
- Installed [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Checkpoints: `ltx-2.3-22b-dev.safetensors` (Pro) and/or `ltx-2-19b-distilled.safetensors` + spatial upsampler (Fast)
- Gemma-3-12B QAT weights for text encoding

## How to Run

```bash
# Set cache directory for compiled model artifacts
export TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-$HOME/.cache/tt-dit}"

# Optional: checkpoint and Gemma paths
export LTX_CHECKPOINT="$HOME/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors"
export GEMMA_PATH="/path/to/gemma-3-12b-it-qat"

# Audio-Video Pro (one-stage, 30 steps) — Wormhole Loud Box 2x4
NO_PROMPT=1 pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_av.py \
  -k "wh_lb_2x4sp0tp1" -s --timeout 3600

# Audio-Video Pro — Blackhole Galaxy 4x8
NO_PROMPT=1 pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_av.py \
  -k "bh_glx_4x8sp1tp0" -s --timeout 3600

# Audio-Video Fast (distilled, 2-stage) — Wormhole Loud Box 2x4
export LTX_CHECKPOINT="$HOME/.cache/ltx-checkpoints/ltx-2-19b-distilled.safetensors"
export LTX_UPSAMPLER="$HOME/.cache/ltx-checkpoints/ltx-2-spatial-upscaler-x2-1.0.safetensors"
NO_PROMPT=1 pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_fast.py \
  -k "wh_lb_2x4sp0tp1" -s --timeout 3600

# Interactive prompt (omit NO_PROMPT)
pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_av.py -k "wh_lb_2x4sp0tp1" -s --timeout 3600

# Video-only smoke (random weights, no checkpoint)
pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx.py -k "test_pipeline_denoising_loop" -v

# Consolidated audio test suites
pytest models/tt_dit/tests/models/ltx/test_audio_components_ltx.py \
  models/tt_dit/tests/models/ltx/test_audio_integration_ltx.py -q
```

Override generation settings with environment variables: `PROMPT`, `NUM_FRAMES`, `HEIGHT`, `WIDTH`, `NUM_STEPS`, `SEED`, `OUTPUT_PATH`.

## Scalability

**Wormhole:**
- 8-chip Loud Box (`2×4` mesh, `sp_axis=0`, `tp_axis=1`, Linear topology)
- 8-chip linear (`1×8` mesh, `sp_axis=1`, `tp_axis=0`)
- 32-chip Galaxy (`4×8` mesh)

**Blackhole:**
- 8-chip Loud Box (`2×4` mesh, `sp_axis=1`, `tp_axis=0`)
- 32-chip Galaxy (`4×8` mesh, Ring topology, `FABRIC_1D_RING`)

The DiT uses sequence parallel (ring attention) and tensor parallel sharding. Text encoding uses reference CPU Gemma via `ltx_pipelines` (`encode_prompts_reference`). The VAE decoder currently runs without spatial mesh sharding (see `vae_ltx.py`).

## Model Variants

### LTXAVPipeline (Pro)
- One-stage AV generation with full guidance (CFG + STG + modality)
- Default: 121 frames, 512×768, 30 steps
- Entry: `models/tt_dit/pipelines/ltx/pipeline_ltx_av.py`

### LTXFastPipeline (Distilled)
- Two-stage: half-resolution denoise → spatial upsample → full-resolution refine
- No CFG/STG (distilled sigmas)
- Entry: `models/tt_dit/pipelines/ltx/pipeline_ltx_fast.py`

### LTXPipeline (video-only)
- Video-only `__call__` for unit tests and video-only paths
- Entry: `models/tt_dit/pipelines/ltx/pipeline_ltx.py`

## Limitations

While output can match the CPU reference at PSNR 21–24 dB for AV video, several items remain in progress:

- AV mode does not yet use FSDP / dynamic_load (unlike Wan2.2)
- LTX VAE has no spatial sharding or temporal chunking; memory scales with resolution and frame count
- Full Gemma on-device encoding is disabled pending numerical fixes; reference CPU encoding is used (~2–3 min)
- Denoise loop keeps latents on host: per-step H2D/D2H, host CFG/Euler, host per-head gate (see `HOST_OPS.md`)
- Pro AV runs up to 4 serial transformer forwards per denoising step (CFG + STG + modality)
- Matmul and Conv3D blocking tables are incomplete for LTX shapes (fallback defaults in logs)
- Audio vocoder sharding tests are not part of the default consolidated audio suite; 2D T-sharding on `2x4` is currently unsupported.
- Performance optimization is ongoing

## Developer Documentation

| Document | Location |
|----------|----------|
| User guide (this file) | `models/tt_dit/models/LTX2.md` |
| Bringup history, bugs, measurements | `models/tt_dit/models/transformers/ltx/BRINGUP.md` |
| Host-side CPU fallbacks | `models/tt_dit/models/transformers/ltx/HOST_OPS.md` |
| Original port plan | `models/tt_dit/models/transformers/ltx/LTX2_DIT_PORT.md` |
| Historical ExecPlans | `models/tt_dit/models/transformers/ltx/plans/` |
| CPU reference runner (debug) | `models/tt_dit/tests/models/ltx/reference_cpu_pipeline.py` |
