---
name: diffusion-datatype-sweep
description: Sweep and select datatype / compute-fidelity policies for a TTNN diffusion model to get the fastest configuration that still meets latent-reconstruction PCC and perceptual-quality bars. Use after the model is runnable end-to-end, to choose weight/activation/accumulation dtypes per module (respecting the reference's fp32-pinned modules), keyed to diffusion quality metrics rather than LLM token top-1/5. Diffusion analog of $datatype-sweep.
---

# Diffusion Datatype Sweep

## Mission Context

If used as part of `$diffusion-model-bringup`, follow its contract. Diffusion models mix precisions: the
reference often pins specific modules to fp32 (`_keep_in_fp32_modules`: input/output projections, time
embedder, sometimes the VAEs) while the block stack runs bf16 with fp32 accumulation. This skill finds the
fastest per-module policy that holds quality.

## How To Approach It

- Start from the reference's mixed-precision contract: keep the fp32-pinned modules fp32; run the rest bf16 +
  fp32 accumulation (HiFi4, `fp32_dest_acc_en`). Then sweep candidates (bf16 vs bf8_b weights, HiFi2 vs HiFi4,
  fp32 escalation of sensitive modules) per component.
- **Metric = diffusion quality, not token accuracy**: per-step velocity PCC and full-trajectory latent PCC vs
  the diffusers reference, plus perceptual non-degeneracy (`$diffusion-qualitative-check`) and, at real
  res/steps, an image/video quality score (FID/CLIP/VBench-style) where feasible.
- Watch **error accumulation over depth**: a per-block bf16 error that is invisible at 2 layers can drop PCC
  after 50 layers, especially in low-dimensional subspaces (e.g. a few-channel audio velocity amplified by a
  final high-gain block). Some floors are hardware limits (device SDPA is bf16-only; a full-fp32 stack may
  exceed DRAM) — record the floor with evidence rather than forcing an unattainable bar.
- Respect L1/DRAM: fp32 activations roughly double circular-buffer sizes; escalating a module to fp32 can push
  a matmul over L1 (see the matmul L1-aware fallback). Re-check memory after each escalation.

## Evidence To Leave

- A `selected_precision_config.json` (per-module dtype/fidelity) with, for each choice, the quality metric and
  the speed/DRAM delta that justified it. Any quality floor set by a hardware limit is documented with the
  measurement (depth sweep, standalone op check, byte calc). Recorded in `doc/<stage>/work_log.md`.
