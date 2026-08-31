---
name: diffusion-model-bringup
description: Orchestrate a full Tenstorrent bringup of a video/image/audio DIFFUSION model (a Diffusion Transformer + VAE(s) + text encoder + iterative denoise loop) in the tt_dit framework. Use as the top-level mission/workspace/reporting contract that the diffusion sub-skills defer to — the diffusion counterpart of the autoregressive-LLM model bringup. Use when bringing up a diffusers pipeline (DiT/MMDiT + VAE + scheduler) rather than an autoregressive text LLM; the LLM path (functional-decoder/full-model/vllm/KV-cache) does NOT apply.
---

# Diffusion Model Bringup (orchestration)

## Mission

Bring up a HuggingFace/diffusers **diffusion** model on Tenstorrent hardware end-to-end: reproduce its DiT,
VAE(s), text encoder, scheduler, and denoise loop in TTNN, validated against the diffusers reference, then
parallelize, optimize, and produce a real generated artifact (image / video / audio). Diffusion models
denoise a fixed latent over N scheduler steps — there is **no KV cache, no prefill/decode, no token loop**.
Check whether the model is guidance-distilled (then there is **no CFG**).

## Workspace & file contracts

Put model code in the `tt_dit` framework, mirroring the closest existing model (LTX-2 for video+audio;
SD3.5/Flux for image), NOT the autoregressive `models/autoports/` tree:
- `models/tt_dit/models/transformers/<model>/` — DiT block + attention + rope (`transformer_<model>.py`,
  `attention_<model>.py`, `rope_<model>.py`) and a distributed variant for multichip.
- `models/tt_dit/models/vae/decoder_<model>.py`, `models/tt_dit/models/audio_vae/decoder_<model>.py`.
- `models/tt_dit/encoders/<enc>/` (reuse clip/t5/gemma/qwen where possible).
- `models/tt_dit/pipelines/<model>/` — scheduler + the end-to-end pipeline + a generate script.
- `models/tt_dit/tests/models/<model>/` — one diffusers-reference PCC test per module.
- Evidence under `models/tt_dit/models/transformers/<model>/doc/<stage>/` — `DESIGN.md`, `README.md`,
  `work_log.md`, plus a `capability_contract.json` (resolution, num_frames, fps, audio rate, latent shapes,
  denoise steps, any hard-limit reductions with byte/probe evidence).

## Stage sequence (each = PCC-validated + evidence + a stage-review)

0. **Reference baseline** — resolve a diffusers build that has the model (a `PYTHONPATH` source clone if the
   pinned diffusers lacks it — do NOT upgrade the shared pin), confirm weights/access, extract the exact
   packed-sequence / patchify / (de)normalization contract, stand up a CPU golden harness (per-module; the
   full model may not run on CPU).
1. **Scheduler** (`$denoise-loop-scheduler`) — host math; bit-exact vs reference.
2. **Functional DiT block** (`$functional-dit-block`, `$adaln-conditioning`, `$multiaxis-rope`) — 1x1, PCC>=0.99.
3. **Full DiT assembly** — config-driven; reduced layers on 1x1 if the real model exceeds one device's DRAM.
4. **Text encoder** (`$text-encoder-port`) — often reuse; multichip if large.
5. **VAE decoder(s)** (`$vae-port`) — decoder-first for text-conditioned generation.
6. **Multichip** (`$multichip`) — shard the real full-size DiT (and encoder) across the mesh; this is often the
   FIRST run of the real model. Report peak DRAM/ASIC.
7. **End-to-end pipeline** (`$diffusion-full-pipeline`, `$diffusion-qualitative-check`) — produce a real clip.
8. **Optimize + trace** (`$optimize`, `$tt-enable-tracing`) — warmed latency, tt-perf-report.
9. **Precision sweep** (`$diffusion-datatype-sweep`) — datatype/fidelity policy keyed to latent/reconstruction
   PCC + perceptual quality (respect the reference's fp32-pinned modules).

## Reporting / evidence contract

Every sub-skill leaves a `work_log.md` with the exact commands, PCC/RMSE, precision recipe, op fallbacks, and
peak DRAM. Default PCC bar 0.99 at block level (relative_rmse 0.05) and 0.99 / relative_rmse 0.15 at
full-model level (the tt_dit standard). Where a hard physical limit (device SDPA dtype, DRAM capacity) sets a
fidelity floor below a bar, record the evidence (depth sweep, standalone op check, byte calc) and treat the
end-to-end perceptual result as the real gate — do not fake a pass or silently lower a PCC bar. Local commits
only; never push. Reuse `$multichip`, `$optimize`, `$tt-enable-tracing`, `$tt-device-usage`, `$stage-review`,
`$autodebug`/`$autofix`/`$autotriage` unchanged. `$functional-decoder`, `$full-model`, `$vllm-integration`,
`$datatype-sweep` (as-is), `$qualitative-check` (as-is) do NOT apply to the diffusion track.
