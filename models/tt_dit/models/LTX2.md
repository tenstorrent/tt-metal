# LTX-2.3

## Introduction

[LTX-2](https://github.com/Lightricks/LTX-2) is a joint audio-video diffusion transformer from Lightricks. The TT-DiT port supports text-to-video (video-only) and text-to-audio-video (AV) generation on Blackhole multi-chip systems. Everything — text encoding, the DiT, the video VAE, and the audio VAE + vocoder — runs on device.

## Details

LTX-2.3 Pro uses a 22B diffusion transformer (48 layers, 4096 dim, 128-channel latents) with on-device Gemma-3-12B text encoding, full MultiModalGuider guidance (CFG + STG + modality), and a causal 3D VAE. The distilled Fast variant uses the `ltx-2.3-22b-distilled-1.1` checkpoint with a 2-stage half-res → upsample → full-res flow (~11 denoising steps total).

## Performance

### 1080p AV Fast (distilled 2-stage) — Blackhole Loud Box (2×4)

Warm (`RUN_WARMUP=1`, untraced), measured on this branch: 1088×1920, 145 frames (~6s @
24fps). Gemma text encode is on-device and disk-cached.

| Stage | Warm |
|-------|------|
| Gemma encode (device; cached prompt) | 0.0s (~0.5s on a fresh prompt) |
| Transformer prepare | 5.7s |
| Stage 1 denoise (half-res) | 11.4s |
| Latent upsample | 1.3s |
| Stage 2 denoise (full-res) | 13.1s |
| VAE prepare | 0.3s |
| VAE decode | 3.9s |
| Audio decode (on-device) | 4.8s |
| Video export | 1.5s |
| **Total (compute)** | **34.6s** |


### 1080p AV Fast (distilled 2-stage) — Blackhole Galaxy (4×8), traced

Traced steady-state (`LTX_TRACED=1`, warmup + replay), same shape: 1088×1920, 145 frames.

| Stage | Time |
|-------|------|
| Gemma encode (device; cached prompt) | 0.0s (~0.5s on a fresh prompt) |
| Stage 1 denoise (half-res) | 2.7s |
| Latent upsample | 0.2s |
| Stage 2 denoise (full-res) | 3.1s |
| VAE decode | 1.1s |
| Audio decode (on-device) | 0.5s |
| **Total** | **7.6s** |


Remaining performance work:
- batched/fused MultiModalGuider passes

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal)
- Installed [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Checkpoints: `ltx-2.3-22b-dev.safetensors` (Pro) and/or `ltx-2.3-22b-distilled-1.1.safetensors` (Fast; the spatial upsampler auto-resolves from the `Lightricks/LTX-2.3` repo). Each resolves automatically from `~/.cache/ltx-checkpoints/` or the HF repo when not set explicitly.
- Gemma-3-12B QAT weights (`google/gemma-3-12b-it-qat-q4_0-unquantized`) for on-device text encoding

## How to Run

```bash
# Set cache directory for compiled model artifacts
export TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-$HOME/.cache/tt-dit}"

# Optional overrides — both resolve automatically (env var > ~/.cache/ltx-checkpoints/ >
# HF repo), so only set them to point at a non-default checkpoint or Gemma snapshot.
export LTX_CHECKPOINT="$HOME/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors"
export GEMMA_PATH="google/gemma-3-12b-it-qat-q4_0-unquantized"

# Audio-Video Fast (distilled 2-stage) — Blackhole Loud Box 2x4 (sp1/tp0)
RUN_WARMUP=1 NO_PROMPT=1 pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py \
  -k "bh_2x4sp1tp0" -s --timeout 3600

# Audio-Video Fast (distilled 2-stage) — Blackhole Galaxy 4x8 (ring)
RUN_WARMUP=1 NO_PROMPT=1 pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py \
  -k "bh_4x8sp1tp0_ring" -s --timeout 3600

# Interactive prompt (omit NO_PROMPT)
pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py -k "bh_2x4sp1tp0" -s --timeout 3600
```

Override generation settings with environment variables: `PROMPT`, `NUM_FRAMES`, `HEIGHT`, `WIDTH`, `NUM_STEPS`, `SEED`, `OUTPUT_PATH`.

Trace gates (all accept `1`/`0`/`true`/`false`; require `LTX_TRACED`): `LTX_VOC_TRACE` (vocoder + BWE, default on), `LTX_VAE_TRACE` (mel decoder, default off).

## Scalability

**Blackhole:**
- 8-chip Loud Box (`2×4` mesh, `sp_axis=1`, `tp_axis=0`)
- 32-chip Galaxy (`4×8` mesh, Ring topology, `FABRIC_1D_RING`)

The DiT uses sequence parallel (ring attention) and tensor parallel sharding. Text encoding runs fully on-device — Gemma-3-12B tensor-parallel across the mesh's wide axis (TP=4 on 2×4, TP=8 on 4×8), with embeddings disk-cached so repeated prompts skip the encoder. The video VAE decoder is spatially mesh-sharded (height across `tp_axis`, width across `sp_axis`, with halo exchange on the sharded conv boundaries — see `vae_ltx.py`), and the audio VAE + vocoder also run on device with weights loaded straight from the checkpoint safetensors.

## Model Variants

`LTXPipeline` (`pipeline_ltx.py`) is the shared base — device machinery (loaders, `call_av`,
encode/decode) only; the concrete variants below implement `generate` / `warmup_buffers`.

### LTXOneStagePipeline (Pro, one-stage)
- Single full-guidance (CFG + STG + modality) denoise on the base 22B checkpoint
- Mirrors the reference `ti2vid_one_stage.TI2VidOneStagePipeline`
- Entry: `models/tt_dit/pipelines/ltx/pipeline_ltx_one_stage.py`

### LTXDistilledPipeline (Distilled)
- Two-stage: half-resolution denoise → spatial upsample → full-resolution refine
- No CFG/STG (distilled sigmas)
- Entry: `models/tt_dit/pipelines/ltx/pipeline_ltx_distilled.py`

### LTXTwoStagesPipeline (Pro + distilled-LoRA refine)
- Two-stage: full-guidance s1 (base 22B) → distilled-LoRA s2 refine
- Entry: `models/tt_dit/pipelines/ltx/pipeline_ltx_two_stages.py`
