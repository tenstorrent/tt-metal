---
description: DiffusionGemma stage 05 — measure and decide the #48291 decision-fidelity bar.
---

Load `diffusion-gemma` first. This is the CORRECTNESS stage (#48291). Diffusion commits the CLEAN ARGMAX, so there is no temperature/top-p cushion — the bf16/MoE/TP=4 backbone's ≈50% argmax agreement vs HF maps almost directly to wrong tokens. This stage measures the real-checkpoint decision fidelity on a denoise trajectory and drives the engineering-vs-product decision. Do NOT gate on AIME24 teacher-forcing top-k. Work only under models/experimental/diffusion_gemma/ and never edit models/demos/gemma4/ (a MoE-precision engineering fix, if pursued, is a shared-backbone change that must be scoped and owned separately, not made in-place here).

Goal completion requirements:
- A real-checkpoint HF-vs-TT replay measures per-position committed-argmax agreement, Gumbel-max sampled/argmax agreement, entropy PCC + max abs error, and entropy-budget accept IoU, on NON-DEGENERATE prompts (avoid all-EOS constant-vs-constant trajectories that trivially score 100%). Use injected reference noise (make_replay_canvas_init_fn / make_replay_noise_fn; demo/replay_hf_tt.py) so the comparison is token-exact.
- Any committed-argmax miss is localized to a source: backbone logits drift, prompt/cache state, adapter numerics, or accept/renoise/entropy ordering — with a per-layer / per-position top-logits capture at the failing positions. Distinguish clean-argmax-level misses from accept-mask drift.
- The precision sensitivity is characterized as diffusion-decision-aware: show where bfp8 small-probability drift flips accept/renoise even when argmax is unchanged. The router / logits→probability / entropy path is treated as high-sensitivity (keep BF16/FP32) and its dtype effect on decisions is measured, not assumed.
- A written decision: the achievable decision-fidelity floor on current kernels, whether it clears a usable bar, and the recommendation (MoE-precision engineering on the shared backbone as a separate owned effort, or product-accept a degraded floor). Note fp32 blockers (ttnn.topk TT_FATAL on FLOAT32; fp32 experts exceed QB2 DRAM).
- doc/decision_fidelity/README.md and work_log.md record the metrics, artifacts (replay .pt files), the localized miss, the decision, and remaining risk.
- `stage-review` returns clean-pass; findings fixed/rereviewed. Locally commit under models/experimental/diffusion_gemma/ (no Co-Authored-By); never push; log SHAs.

Unmet requirements, review findings, failed gates: work. Stop only after `autofix` fails.
