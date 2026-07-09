---
description: DiffusionGemma stage 07 — choose the fastest precision config that preserves the diffusion decisions.
---

Read `models/experimental/diffusion_gemma/.agent/skills/diffusion-gemma/SKILL.md` and `models/experimental/diffusion_gemma/.agent/skills/datatype-sweep/SKILL.md` first. This is the PRECISION stage (#47475 / #47465). The completed dg-07 result keeps BF16 experts: BF8 saved 5.44 GiB/chip and improved traced @48 throughput by 9.1%, but failed committed/entropy/accept decision gates. Reopen the sweep only for a new candidate or changed #48291 baseline. Do NOT begin vLLM work or edit models/demos/gemma4/.

Goal completion requirements:
- The accuracy source of truth is a diffusion-decision metric, NOT AIME24 top-1/top-5: per-step Gumbel-max argmax agreement, entropy PCC, and end-to-end accept/renoise agreement vs the injected-noise reference trajectory. The bar and its rationale (small-probability sensitivity) are recorded.
- The router / top-k / logits→probability / entropy path is treated as high-sensitivity (kept at BF16/FP32 and explicitly validated), NOT swept down in the coarse pass, because it drives the diffusion decision.
- Candidate results in `models/experimental/diffusion_gemma/doc/datatype_sweep/sweep_results.{json,csv}` include config, dtype/fidelity policy, decision metrics, traced step/block latency, command, hardware, mesh, and pass/fail. Do not spend device time on BFP4 after a corresponding BF8 group already fails unless new fidelity headroom or a distinct technical rationale exists.
- `selected_precision_config.json` includes weight groups, layer exceptions, compute fidelities, activation/residual dtype, CCL dtype, KV-cache dtype, canvas-scratch dtype, and logits→probability/entropy dtype, with runtime-consumption proof.
- Two Pareto charts (argmax-agreement vs latency, accept-agreement vs latency) mark the selected point and the minimum-allowed-decision-agreement line. Ranking uses TRACED per-step/per-block latency, not eager.
- Recompute `models/experimental/diffusion_gemma/doc/context_contract.json` for every KV-cache/canvas-scratch dtype candidate that changes capacity.
- `models/experimental/diffusion_gemma/doc/datatype_sweep/README.md` and `work_log.md` record thresholds, selected/rejected configs, Pareto interpretation, commands, plots, performance, limitations, and exact artifacts.
- `stage-review` returns clean-pass; findings fixed/rereviewed. Locally commit under models/experimental/diffusion_gemma/ (no Co-Authored-By); then push — invoking this stage command explicitly authorizes both actions; never edit models/demos/gemma4/; log SHAs.

Unmet requirements, review findings, failed gates: work. Stop only after `autofix` fails.
