# Phase review, round 1 (optimization-loop, iterations 1-3), 2026-10-29

Verdict: more-work-needed

The numbers re-derive cleanly (projections 1658.6 / 1592.5 / 1573.3 / 1398.7 / 1339.0 ms, residuals ≤ 0.6 %; MFU figures 38.53 / 38.06 / 29.15 / 28.74 / 28.25 / 27.10 % reproduce; `compare_loss.py` PASS at 0.0262; diffs show the targeted ops gone and the new ops as documented) and the exit is earned on the plan's target under every clock convention.

## Required Work
- P2: the TP=4 fused/transposed paths (the full-model report path: `fused_lora.py` tail_scale / dh_scale / reduce axes, `lora.py` TP branches; under TP w2 is transposed *and* fused, a combination never run on one chip) have no gradient-level verification; the harness is single-chip, and the TP loss trajectory cannot discriminate an adapter-gradient error (the composite TP baseline had a known diverging `lora_A` and the same trajectory). Required: one TP=4 gradient comparison at 1-2 layers (non-zero B), gathered `dA`/`dB` vs the single-chip path at the harness bars, cited from "Loop exit"; or relabel the TP numbers as perf-only.
- P2: the exit quotes "1559-1600 ms wall at 1130-1160 MHz" while iteration 3's own runs delivered 1090 MHz (MFU run) and 1041 MHz (profiled), i.e. 1661-1739 ms wall; the clock fell monotonically through the loop (profiled 1130 → 1041 MHz, MFU 1160 → 1090 MHz, 150-163 W under load), consistent with a power limit: cycle-time gains convert to wall at ~0.7x. Required: restate the exit at the measured clock and classify the trend in the anomaly ledger.
- P2 (documentation): tail writes are 37.1 ms (the `other` class), not 31.6; the ~3 ms dispatch is outside the 1339 ms kernel sum; math inefficiency against the math-only roofline (744) is 491 ms; SDPA forward is 7.6x its roofline; the fused forward error claim (1.7e-3 vs 2.4e-3) is the isolation test, the harness block shows 4.18e-3 vs 4.58e-3; no stated overhead threshold and the remaining glue / layout items carry no rung or reason; no commit SHAs per iteration in the work log.

## Other Concerns
- Iteration 1's single-chip MFU rerun (438.7 vs 434.0 ms) did not beat the best path in wall-clock (1101 vs 1160 MHz, no cool rerun); the cycle-time and TP gains carry the claim.
- Iteration 3's single-chip wall gain (-1.2 %) is inside noise; cycle-time (-3.8 %) and the two TP runs carry it.
- After the fusion the small reduce-scatter shows wait skew across chips (13-199 µs); collectives are 4.3-8.6 ms per chip per 4 layers and not attributed beyond "attributed separately".
- The plan's rule for moving the depths to 2/4/8/16 once 16 layers fit is now triggered (279 MiB retained per layer).
- `analyze_tracy.py` has no wasted class; the wasted = 0 claim rests on attribution.
- `iter1_loraapply_mfu_tp4_hot.json` was renamed after the fact (its internal loss_log path is the old name).

## Hard-Check Gaps
- The fused tail-scale bug is documented in prose only (the isolation script and its failing output are not in evidence).
- No iteration-3 op-sequence attribution file.
- The `*_harness.log` files have a post-loop mtime (force-add commit should say whether they were rewritten).

## Anomaly Ledger (reviewer)
- Iteration-1 hot TP run (chip 3 at 1.63-1.69x): controlled by the cool rerun and the per-chip tool.
- Falling delivered clock across iterations: more-work-needed (classification; P2).
- First iteration-2 profile (dX routes without core_grid): fixed by `iter2b`.
- `iter2b` harness stale assertion: fixed in `iter2c`.
- Fused tail written without the LoRA scale: fixed before profiling; proof in `iter3_fused_quick.log` and the harness.

## Scope Inspected
Skills: stage-review, optimization-loop. Artefacts: plan.md, perf_analysis.md, optimization README and work log, every per-iteration evidence file (harness, compact and full Tracy CSVs, analyses, clock analyses, MFU logs/JSON with tt-smi, TP profiles, hot artefacts, microbenchmarks, final validation, chain scripts and logs), composite references. Code: modules/{fused_lora,frozen_linear,lora,linear}.py, frozen_embedding.py, examples/mistral_train/*.py, tests/python/test_mistral.py; the six loop commits. Commands: analyze_tracy.py on every iteration, diff_tracy.py on consecutive pairs, clock_analysis.py on the full CSVs, tp_analysis.py on six TP profiles plus a per-chip collective breakdown, compare_loss.py, MFU and gap arithmetic. No devices opened.

## Residual Risk
- The delivered clock, not kernel work, now dominates the wall-clock gap; if the board is power-limited, per-cycle gains keep converting at ~0.7x.
- ~100 ms of removable overhead remains (glue 36, layout 31, tail writes 37); the stretch target is 6 % away.
- The 1/2/4/8 fit extrapolates 5x; a 16-layer point is now affordable.
