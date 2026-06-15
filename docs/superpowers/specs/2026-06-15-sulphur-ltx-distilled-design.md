# Sulphur-2 on tt-metal LTX-2.3 distilled pipeline

**Date:** 2026-06-15
**Branch:** `ltx-sulphur` (worktree off `ltx-perf`)
**Goal:** Generate videos with `SulphurAI/Sulphur-2-base` weights through tt-metal's existing
LTX-2.3 distilled AV pipeline. Distilled first; full/dev as fallback.

## Key finding: it's a weight swap, not a port

`SulphurAI/Sulphur-2-base` is a fine-tune of `Lightricks/LTX-2.3` (22B DiT video model).
tt-metal already implements LTX-2.3 22B at `models/tt_dit/pipelines/ltx/` with a working
distilled AV pipeline (`pipeline_ltx_distilled.py`) and an e2e test
(`tests/models/ltx/test_pipeline_ltx_distilled.py`).

The Sulphur distilled checkpoint `sulphur_distil_bf16.safetensors` (46 GB) is the **full
LTX-2.3 distilled bundle** with the Sulphur fine-tune baked in. Verified via safetensors
header (HTTP range read, no full download):

- `model.diffusion_model.*` — 4444 keys (DiT, incl. `audio_adaln_single`, AV layers)
- `vae.*` — 170 keys (video VAE decoder + per_channel_statistics)
- `audio_vae.*` — 102 keys
- `vocoder.*` — 1227 keys
- `text_embedding_projection.*` — 4 keys

These prefixes are **byte-for-byte what the tt-metal loader expects**:
- `_build_transformer_state_dict` strips `model.diffusion_model.`
- VAE loader reads `vae.decoder.` and `vae.per_channel_statistics.`
- `adaln_single.linear.weight` shape `[36864, 4096]` → `36864 > 6*4096` → cross-attn AdaLN
  variant detected correctly.

So `sulphur_distil_bf16.safetensors` is a **drop-in replacement** for
`ltx-2.3-22b-distilled-1.1.safetensors`. The two components Sulphur does NOT ship — the
Gemma text encoder and the spatial upscaler (`Lightricks/LTX-2.3:ltx-2.3-spatial-upscaler-x2-1.1`)
— already load from their original sources, unchanged.

`_resolve_checkpoint_file` accepts a `repo:file` HF ref and auto-downloads, so
`checkpoint_name="SulphurAI/Sulphur-2-base:sulphur_distil_bf16.safetensors"` is sufficient.

## Approach

Minimal, reversible, env-driven — matching the existing test's `PROMPT`/`HEIGHT`/`WIDTH`
pattern and the repo's env-gated style (e.g. `LTX_AUDIO_SUBMESH`):

1. Add a `CHECKPOINT` env override to `test_pipeline_ltx_distilled.py` so the hardcoded
   `default_ltx_checkpoint(...)` can be replaced by any `repo:file` ref. One line, no
   shared pipeline-code changes.
2. Run the girl-singing e2e (`DEFAULT_LTX_PROMPT`) on the **Lightricks** distilled
   checkpoint first — the before-gate proving the pipeline works on `ltx-perf`.
3. Run the same e2e with `CHECKPOINT=SulphurAI/Sulphur-2-base:sulphur_distil_bf16.safetensors`
   — the after-gate. Same prompt, swapped weights.
4. Compare MP4 outputs (ffprobe: video + audio streams, duration, resolution).

## TDD

Fast CPU-only test before any 46 GB device run: assert the Sulphur checkpoint's
transformer/VAE key sets (after prefix stripping) are a superset of what the Lightricks
distilled checkpoint provides — i.e. the loader will find every key it needs. Read both
headers via range requests; no full download. Fails before the override path exists,
passes after.

## Variants (loop, no human input)

- **Sulphur LoRA** (`sulphur_lora_rank_768.safetensors`, 10 GB): the fine-tune delta as a
  LoRA on top of the Lightricks distilled base. The pipeline already supports LoRA fusing
  (`_build_transformer_state_dict(checkpoint, lora_specs)`). Try if the merged base works.
- **Other Sulphur mods**: scan the HF org / community for additional checkpoints or
  resolutions; try any that map onto the same loader.

## Risks

- `ltx-perf` is 28 commits behind `smarton/audio-submesh-e2e`, missing recent audio fixes
  (polyphase upsample, submesh routing). If AV decode misbehaves, cherry-pick those fixes
  or fall back to video-only. (User chose `ltx-perf` as base explicitly.)
- Sulphur is marketed text-to-video; its inherited audio branch may produce low-quality
  audio. AV mode chosen by user regardless — exercises the full pipeline.
- Distilled checkpoint may differ subtly in LoRA-merge recipe from Lightricks distilled;
  if quality is poor, try `sulphur_dev_bf16.safetensors` (full/dev) instead.

## Success criteria

A playable MP4 generated from Sulphur weights via the distilled pipeline on device, with
the girl-singing e2e passing both before (Lightricks) and after (Sulphur) the swap.
