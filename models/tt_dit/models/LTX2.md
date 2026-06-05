# LTX-2.3

## Introduction

[LTX-2](https://github.com/Lightricks/LTX-2) is a joint audio-video diffusion transformer from Lightricks. The TT-DiT port supports text-to-video (video-only) and text-to-audio-video (AV) generation on Wormhole and Blackhole multi-chip systems.

Clone the reference repo beside `tt-metal` (not vendored as a submodule):

```bash
git clone https://github.com/Lightricks/LTX-2.git LTX-2
```

## Details

LTX-2.3 Pro uses a 22B diffusion transformer (48 layers, 4096 dim, 128-channel latents) with on-device Gemma-3-12B text encoding, full MultiModalGuider guidance (CFG + STG + modality), and a causal 3D VAE. The distilled Fast variant uses the `ltx-2.3-22b-distilled-1.1` checkpoint with a 2-stage half-res → upsample → full-res flow (~11 denoising steps total).

## Performance

### 1080p AV Fast (distilled 2-stage) — Blackhole Loud Box (2×4)

Warm end-to-end, measured on this branch: 1088×1920, 145 frames (~6s @ 24fps). Gemma text
encode is on-device and disk-cached.

| Stage | Warm |
|-------|------|
| Gemma encode (device; cached prompt) | 0.0s (~0.5s on a fresh prompt) |
| Transformer prepare | 5.7s |
| Stage 1 denoise (half-res) | 11.0s |
| Latent upsample | 1.6s |
| Stage 2 denoise (full-res) | 18.0s |
| VAE decode | 4.3s |
| Audio decode (on-device) | 4.7s |
| Video export | 1.3s |
| **Total** | **47.0s** |

One-time warmup (compile every device program, incl. the encoder): ~176s.

### AV Pro 512p — prior measurements (not re-verified on this branch/HW)

512×768, 121 frames, 30 steps; denoise loop only (excludes text encode):

| System | Arch | SP | TP | Denoise | s/step | Notes |
|--------|------|----|----|---------|--------|-------|
| Loud Box (2×4) | WH | 2 | 4 | ~795s | ~20–29s | 4 guidance passes/step; host sync overhead |
| Galaxy (4×8) | BH | 8 | 4 | ~843s | ~28s | Ring topology, `FABRIC_1D_RING` |

On-device Gemma encoding (TP=4 on 2×4 by default), with embeddings disk-cached.
Remaining performance work:
- batched/fused MultiModalGuider passes
- tuned matmul and SDPA blocking

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal)
- Cloned [LTX-2](https://github.com/Lightricks/LTX-2) at `LTX-2/` (for reference text encoding and audio decode)
- Installed [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Checkpoints: `ltx-2.3-22b-dev.safetensors` (Pro) and/or `ltx-2.3-22b-distilled-1.1.safetensors` (Fast; the spatial upsampler auto-resolves from the LTX-2.3 repo)
- Gemma-3-12B QAT weights for on-device text encoding

## How to Run

```bash
# Set cache directory for compiled model artifacts
export TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-$HOME/.cache/tt-dit}"

# Optional: checkpoint and Gemma paths
export LTX_CHECKPOINT="$HOME/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors"
export GEMMA_PATH="/path/to/gemma-3-12b-it-qat"

# Gemma encodes on-device by default. RUN_WARMUP=1 pre-compiles all device programs
# (incl. the encoder) so the timed generation is warm. Do NOT set TT_METAL_WATCHER on a
# multi-chip fabric mesh — it overflows the active-eth fabric-router kernel-config buffer.

# Audio-Video Pro (one-stage, 30 steps) — Wormhole Loud Box 2x4 (sp0/tp1)
NO_PROMPT=1 pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx.py \
  -k "2x4sp0tp1" -s --timeout 3600

# Audio-Video Pro — Blackhole Galaxy 4x8 (ring)
NO_PROMPT=1 pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx.py \
  -k "bh_4x8sp1tp0_ring" -s --timeout 3600

# Audio-Video distilled (2-stage) — Blackhole Loud Box 2x4
export LTX_CHECKPOINT="$HOME/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors"
RUN_WARMUP=1 NO_PROMPT=1 pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled_av.py \
  -k "bh_2x4sp1tp0" -s --timeout 3600

# Interactive prompt (omit NO_PROMPT)
pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx.py -k "2x4sp0tp1" -s --timeout 3600

# Scheduler primitives (no device)
pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx.py -k "sigma_schedule or euler_step" -v

# Audio test suites (decode components + integration)
pytest models/tt_dit/tests/models/ltx/test_audio_components_ltx.py \
  models/tt_dit/tests/models/ltx/test_audio_ltx.py -q
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

The DiT uses sequence parallel (ring attention) and tensor parallel sharding. Text encoding runs fully on-device — Gemma-3-12B tensor-parallel across the mesh's wide axis (TP=4 on 2×4, TP=8 on 4×8), with embeddings disk-cached so repeated prompts skip the encoder. The audio VAE + vocoder also run on device (weights loaded straight from the checkpoint safetensors); there is no host/CPU reference path in the pipeline anymore. The VAE decoder currently runs without spatial mesh sharding (see `vae_ltx.py`).

## Model Variants

### LTXPipeline (Pro, base)
- One-stage AV generation with full guidance (CFG + STG + modality)
- Default: 144 frames, 512×768, 30 steps
- `generate()` orchestration + `call_av()` denoise loop
- Entry: `models/tt_dit/pipelines/ltx/pipeline_ltx.py`

### LTXDistilledPipeline (Distilled)
- Two-stage: half-resolution denoise → spatial upsample → full-resolution refine
- No CFG/STG (distilled sigmas)
- Entry: `models/tt_dit/pipelines/ltx/pipeline_ltx_distilled.py`

### LTXTwoStagesPipeline (Pro + distilled-LoRA refine)
- Two-stage: full-guidance s1 (base 22B) → distilled-LoRA s2 refine
- Entry: `models/tt_dit/pipelines/ltx/pipeline_ltx_two_stages.py`
