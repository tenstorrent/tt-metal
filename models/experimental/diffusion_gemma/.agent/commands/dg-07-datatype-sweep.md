---
description: DiffusionGemma stage 07 — choose the fastest precision config that preserves the diffusion decisions.
---

Load `diffusion-gemma` first. This is the PRECISION stage (#47475 quant/dequant, #47465 perf). It selects the weight / activation / CCL / KV-cache / canvas-scratch datatype config that is fastest while preserving the diffusion DECISIONS (not teacher-forcing top-1/top-5). Requires the decision-fidelity stage (#48291) reference. The accuracy metric is diffusion-aware because bfp8 small-probability drift can FLIP accept/renoise. Do NOT begin vLLM work. Do NOT edit models/demos/gemma4/ — the sweep drives the shared backbone's precision knobs through DiffusionGemma-local config/construction, never by editing the backbone. Work only under models/experimental/diffusion_gemma/.

Goal completion requirements:
- The accuracy source of truth is a diffusion-decision metric, NOT AIME24 top-1/top-5: per-step Gumbel-max argmax agreement, entropy PCC, and end-to-end accept/renoise agreement vs the injected-noise reference trajectory. The bar and its rationale (small-probability sensitivity) are recorded.
- The router / top-k / logits→probability / entropy path is treated as high-sensitivity (kept at BF16/FP32 and explicitly validated), NOT swept down in the coarse pass, because it drives the diffusion decision.
- Candidate results in doc/datatype_sweep/sweep_results.{json,csv}: config id, dtype policy, compute fidelity, the diffusion-decision agreement metrics, per-denoise-step / per-block latency (traced), measurement regime, command, hardware, mesh, pass/fail. Every material BFP4 matmul group has a paired BFP4+LoFi candidate or an exact TTNN blocker + `autofix` evidence.
- selected_precision_config.json includes weight groups, layer exceptions, compute fidelities, activation/residual dtype, CCL dtype, KV-cache dtype, canvas-scratch dtype, and logits→probability/entropy dtype — and evidence proves the measured runtime path actually consumes it (a field ignored by hard-coded code does not count).
- Two Pareto charts (argmax-agreement vs latency, accept-agreement vs latency) mark the selected point and the minimum-allowed-decision-agreement line. Ranking uses TRACED per-step/per-block latency, not eager.
- Recompute doc/context_contract.json for every KV-cache/canvas-scratch dtype candidate that changes capacity.
- doc/datatype_sweep/README.md and work_log.md record thresholds, selected/rejected configs, Pareto interpretation, commands, plots, perf, limitations, exact artifacts.
- `stage-review` returns clean-pass; findings fixed/rereviewed. Locally commit under models/experimental/diffusion_gemma/ (no Co-Authored-By); then push; never edit models/demos/gemma4/; log SHAs.

Unmet requirements, review findings, failed gates: work. Stop only after `autofix` fails.
