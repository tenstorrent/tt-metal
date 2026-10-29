# Optimisation loop: Mistral-Small-24B-Base-2501 LoRA

Phase 3 of the tt-train bring-up. Contract: `../plan.md` (target 1620 ms cycle-time per 40-layer step at 1350 MHz = 1562 ms of kernels + 58 ms planned dispatch, ±10 %; at the delivered 1.13-1.19 GHz that is 1885 ms wall; full-model MFU and loss runs with TP=4 per the user's scope change). State of the bring-up: `../perf_analysis.md` (MFU history, ranked cost table). Commands and artefacts per iteration: `work_log.md`; raw evidence under `evidence/`.

## Measurement conventions used in every iteration

- Single-chip Tracy at 1/2/4/8 layers (`analyze_tracy.py`, four-point fit, projection to 40) is the attribution and the cycle-time A/B (`diff_tracy.py`: targeted ops gone, nothing new). Tracy ms are cycles / 1.35 GHz; the delivered clock of each run is taken from the cycle counters (`clock_analysis.py`) so wall-clock comparisons are made at a stated clock.
- Single-chip MFU at 8 of 40 layers (no profiler) with `tt-smi` sampled alongside; TP=4 full-model MFU (40 layers, 594 TFLOP/s denominator) with all four chips sampled, after a cool-down, and a 4-layer TP profile checked per chip (`tp_analysis.py`) for a stalled chip before any TP number is believed.
- PCC harness (`tests/python/test_mistral.py`, 11 tests, every module and layer kind, forward and every gradient) must pass before a change lands.

## Iterations

(see below; one section per iteration, newest last)

### Iteration 1: 3-op LoRA apply (glue), landed

Port of `LoraApply` / `_LoraUpdateAdd` from `umales/qwen38` into `ttml/modules/lora.py`. Harness 11 passed. Cycle-time projection 1658.6 -> 1592.5 ms (-4.0 %), glue 134.6 -> 72.9 ms; TP=4 full model 598.7 -> 574.3 ms (27.10 -> 28.25 % MFU) on cool chips. Details and the hot-chip anomaly in `../perf_analysis.md`, "Iteration 1". Measurement lesson folded into the conventions above: after a long TP run one chip stayed throttled (every matmul 1.68x the cycles) and the mesh waited in every collective; TP numbers are only taken after a cool-down with per-chip sampling and the per-chip matmul ratio check.

### Iteration 2: base-linear matmuls on a transposed frozen weight (math), landed

`ttml/modules/frozen_linear.py`: frozen linears whose shape has a measured win (qkv, out, gate_up, LM head; w2 excluded by measurement) are stored `[in, out]` after loading, run a plain matmul forward on a tuned core grid and take the input gradient as `(W_t @ g^T)^T` on the full grid. Harness 13 passed (two new transposed-path cases). Cycle-time projection 1592.5 -> 1398.7 ms (-12.2 %; gate_up forward 44 -> 63 % of the delivered peak, its dX 68 -> 85 %); single-chip MFU at 8 of 40 layers 380.0 ms / 38.06 %; TP=4 full model 574.3 -> 564.5 ms (28.74 %). Lesson recorded: a `ttnn.matmul` call without an explicit `core_grid` falls into a 1.5-1.8x slower default config on these shapes (first profile of the iteration), and microbenchmarks must mirror the exact call. Details in `../perf_analysis.md`, "Iteration 2".

### Iteration 3: LoRA update fused into the base matmul (glue + LoRA math), landed

`ttml/modules/fused_lora.py`: `[x | x A^T] @ [[W_t], [s B^T]]` for the transposed adapters (qkv, out, gate_up), tail rewritten per step, one `W_cat_t @ g^T` in the backward. Harness 14 passed (a tail-scale bug on the first run was caught by the fused block case). Cycle-time projection 1398.7 -> 1339.0 ms (-4.3 %; glue 72 -> 36 ms); single-chip MFU at 8 of 40 layers 375.3 ms / 38.53 %; TP=4 full model 564.5 -> 556.5 ms (29.15 %). The remaining cost of the fusion is the per-step tail write (`ttnn.fill_cache`, 0.79 ms per gate_up layer, 31.6 ms per step), the next layout item. With 1339 ms of cycle-time the single-chip step is 17 % under the plan's 1620 ms target; details in `../perf_analysis.md`, "Iteration 3".
