---
description: DiffusionGemma stage 08 — optimize the denoise-step / per-block performance.
---

Load `diffusion-gemma` first, then read the `DiffusionGemma denoise-step optimization playbook` in the `optimize` skill (just before `Evidence To Leave`) — it grounds where the real headroom is: the denoise step is OP-COUNT bound (~4176 ms/step, 98.8% per-layer backbone, 85–170× the bandwidth roofline), the ranked DG-local levers are `DG-OPT-D01..D06` (de-chunk the 256-canvas RMSNorms, fuse RoPE, close the SDPA L1 static-CB clash), and the shared-gemma4 MoE / #48291 ceiling is what you must NOT re-grind. Evidence lives under `doc/optimize_perf/` (`work_log.md`, `perf_summary.json`, `prof_denoise_step.py`, `bench_sampling_step.py`); the build is `ENABLE_TRACY=OFF`, so use traced Metal capture/replay + synchronized per-op device-time tables (see `Profiling without Tracy`), not `tt-perf-report` op-CSV.

This is the PERF stage (#47465). The optimization unit is the DENOISE STEP over the 256-token canvas (≤48 steps/block) + the commit, NOT per-token autoregressive decode. Preserve the selected precision config and the diffusion decisions. Do NOT begin vLLM work. Do NOT edit models/demos/gemma4/ — optimize DiffusionGemma-local code and drive the backbone through its existing knobs. Work only under models/experimental/diffusion_gemma/.

Goal completion requirements:
- Performance is reported per-denoise-step, per-block, and full-generation (prefill TTFT + Σ steps + commit), with a traced measured path — eager/untraced denoise-step numbers are not acceptable for ranking. Report tokens-per-block / blocks-per-second, never 1000/mean_tpot_ms.
- An operation-topology audit of the measured denoise step: the sort → cumsum → scatter/inverse-permutation entropy-accept chain, the per-step canvas recompute (256-token mini-prefill against the frozen prefix), the [P+C] KV concat, the soft-embedding vocab matmul, self-conditioning, and the logits path. Program configs / sharding / DRAM-vs-L1 placement for sort/cumsum/scatter over the 256 axis are tuned with before/after evidence (these ops have no generic autoregressive guidance — add a candidate table).
- The roofline is computed for the diffusion path: per step it re-reads weights and recomputes over the full 256 canvas against the frozen prefix; there is no incremental single-token KV read. Reconcile measured device time against this per-step-weight-traffic × steps model.
- The optimized loop keeps the fixed ≤48-step trace-safe shape (on-device cutoff mask, tensor-valued scatter indices, warmed program cache) from the denoise-loop stage; early-halt does not shorten a static trace — record the chosen static/fixed-count scheme.
- tt-perf-report tables + CSV/provenance for prefill, one denoise step, and commit. Actionable advice tried; rejections have before/after evidence. Runtime fallback audit clean; watcher-clean.
- doc/optimize_perf/README.md and work_log.md record before/after per-step/per-block perf, topology audit, roofline reconciliation, chosen configs, rejected options, limitations, exact artifacts. Applicable `optimize` checklist items have evidence.
- `stage-review` returns clean-pass; findings fixed/rereviewed. Locally commit under models/experimental/diffusion_gemma/ (no Co-Authored-By); then push; never edit models/demos/gemma4/; log SHAs.

Unmet requirements, review findings, failed gates: work. Stop only after `autofix` fails.
