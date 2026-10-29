# Optimisation loop: Mistral-Small-24B-Base-2501 LoRA

Phase 3 of the tt-train bring-up. Contract: `../plan.md` (target 1620 ms cycle-time per 40-layer step at 1350 MHz = 1562 ms of kernels + 58 ms planned dispatch, ±10 %; at the delivered 1.13-1.19 GHz that is 1885 ms wall; full-model MFU and loss runs with TP=4 per the user's scope change). State of the bring-up: `../perf_analysis.md` (MFU history, ranked cost table). Commands and artefacts per iteration: `work_log.md`; raw evidence under `evidence/`.

## Measurement conventions used in every iteration

- Single-chip Tracy at 1/2/4/8 layers (`analyze_tracy.py`, four-point fit, projection to 40) is the attribution and the cycle-time A/B (`diff_tracy.py`: targeted ops gone, nothing new). Tracy ms are cycles / 1.35 GHz; the delivered clock of each run is taken from the cycle counters (`clock_analysis.py`) so wall-clock comparisons are made at a stated clock.
- Single-chip MFU at 8 of 40 layers (no profiler) with `tt-smi` sampled alongside; TP=4 full-model MFU (40 layers, 594 TFLOP/s denominator) with all four chips sampled, after a cool-down, and a 4-layer TP profile checked per chip (`tp_analysis.py`) for a stalled chip before any TP number is believed.
- PCC harness (`tests/python/test_mistral.py`, 11 tests at the composite baseline, 14 at the final path with the transposed and fused cases; every module and layer kind, forward and every gradient) must pass before a change lands.
- From iteration 4 on (user priority, 2026-10-29): the full-model TP=4 MFU run after a cool-down is the result; the single-chip Tracy depths only attribute and rank; no single-chip MFU runs.

## Iterations

(see below; one section per iteration, newest last)

### Iteration 1: 3-op LoRA apply (glue), landed

Port of `LoraApply` / `_LoraUpdateAdd` from `umales/qwen38` into `ttml/modules/lora.py`. Harness 11 passed. Cycle-time projection 1658.6 -> 1592.5 ms (-4.0 %), glue 134.6 -> 72.9 ms; TP=4 full model 598.7 -> 574.3 ms (27.10 -> 28.25 % MFU) on cool chips. Details and the hot-chip anomaly in `../perf_analysis.md`, "Iteration 1". Measurement lesson folded into the conventions above: after a long TP run one chip stayed throttled (every matmul 1.68x the cycles) and the mesh waited in every collective; TP numbers are only taken after a cool-down with per-chip sampling and the per-chip matmul ratio check.

### Iteration 2: base-linear matmuls on a transposed frozen weight (math), landed

`ttml/modules/frozen_linear.py`: frozen linears whose shape has a measured win (qkv, out, gate_up, LM head; w2 excluded by measurement) are stored `[in, out]` after loading, run a plain matmul forward on a tuned core grid and take the input gradient as `(W_t @ g^T)^T` on the full grid. Harness 13 passed (two new transposed-path cases). Cycle-time projection 1592.5 -> 1398.7 ms (-12.2 %; gate_up forward 44 -> 63 % of the delivered peak, its dX 68 -> 85 %); single-chip MFU at 8 of 40 layers 380.0 ms / 38.06 %; TP=4 full model 574.3 -> 564.5 ms (28.74 %). Lesson recorded: a `ttnn.matmul` call without an explicit `core_grid` falls into a 1.5-1.8x slower default config on these shapes (first profile of the iteration), and microbenchmarks must mirror the exact call. Details in `../perf_analysis.md`, "Iteration 2".

### Iteration 3: LoRA update fused into the base matmul (glue + LoRA math), landed

`ttml/modules/fused_lora.py`: `[x | x A^T] @ [[W_t], [s B^T]]` for the transposed adapters (qkv, out, gate_up), tail rewritten per step, one `W_cat_t @ g^T` in the backward. Harness 14 passed (a tail-scale bug on the first run was caught by the fused block case). Cycle-time projection 1398.7 -> 1339.0 ms (-4.3 %; glue 72 -> 36 ms); single-chip MFU at 8 of 40 layers 375.3 ms / 38.53 %; TP=4 full model 564.5 -> 556.5 ms (29.15 %). The remaining cost of the fusion is the per-step tail write (`ttnn.fill_cache`, 0.79 ms per gate_up layer, 31.6 ms per step), the next layout item. With 1339 ms of cycle-time the single-chip step is 17 % under the plan's 1620 ms target; details in `../perf_analysis.md`, "Iteration 3".

### Iteration 4: multi-core tail write + swept gate_up program config (tail writes + math), landed; loop resumed

User direction on resuming: continue past the met target; HiFi4 with fp32 accumulation is mandatory; the full-model TP=4 MFU run is the result of an iteration, single-chip depth runs only rank what to fix next (no single-chip MFU runs, no 16-layer profile from here on). New ttml metal op `ttml.ops.metal.write_tile_block` (`metal/ops/write_tile_block/`, multi-core tiled block copy in place) replaces the one-core `fill_cache` tail write (gate_up tail 0.85 -> 0.06 ms); `FORWARD_PROGRAM_CONFIG` in `frozen_linear.py` applies the swept 1D config for the gate_up forward (7.92 -> 7.35 ms, bit-exact). Harness 14 passed. Cycle-time projection 1339.0 -> **1256.3 ms** (-6.2 %; gate_up fwd 296 -> 254, tail writes 37 -> 1). **TP=4 full model 558.5 ms / 29.05 %, unchanged**: the config is keyed on the single-chip shape and the TP-local `[5184, 16384]` keeps the default; the tail-write gain at TP (~7 ms) is under the thermal noise of that run (chip 2 at 1.10 GHz effective). Details in `../perf_analysis.md`, "Iteration 4".

### Iteration 5: batched TP all-reduce of the partial LoRA gradients (collectives), landed

The fused TP wrappers keep the LoRA `h` / `dh` partial per rank (the reductions are linear) and `reduce_partial_lora_grads` sums the stacked partial `dA` / `dB` once per step instead of two 64 KB latency-bound collectives per layer. Harness 14 passed; TP=4 gradient check PASS. TP=4 full model **556.9 ms / 29.13 %** vs 558.5: the removed collectives were waiting time on the fast chips, the step follows the slowest chip's compute (chip 2 at 1.12 GHz effective in this run). Details in `../perf_analysis.md`, "Iteration 5".

### Iteration 5b: batched TP gradient mode for the composite adapters (collectives), landed

qkv and out run on the composite LoRA path at their TP-local shapes; they now keep their `h` / `dA` partial too (the row-parallel update joins the partial base before the output all-reduce), so no LoRA path issues a per-layer collective any more (4 partial gradients per layer stacked into two collectives per step). Harness 15 passed, TP=4 gradient check PASS. TP=4 **563.2 ms / 28.81 %** at 1.11-1.15 GHz effective on all chips vs 556.9 at 1.12-1.26 GHz: identical compute rows in cycle-time, the difference is the clock; neutral on this thermally bound box. Details in `../perf_analysis.md`, "Iteration 5b".

### Iteration 6: RMSNorm backward on all cores (math), landed

`metal/ops/rmsnorm_bw_allcore/` (the two-phase all-core backward of `umales/qwen38` as its own op), used through `models/mistral/rmsnorm_allcore.py` for every frozen-gamma `RMSNormLayer` (no dgamma pass). New harness case (dx PCC 0.999997), harness 15 passed, TP=4 gradient check PASS. RMSNorm backward 0.37 -> 0.21 ms per call on one chip and on every TP chip. **TP=4 full model 547.5 ms / 29.63 %** (best 545.0), the best full-model number so far (556.9 before; composite baseline 598.7). Details in `../perf_analysis.md`, "Iteration 6".

### Iteration 7 trial: qkv / out on the fused path at TP (math + glue), rejected

The TP-local qkv / out shards were put on the transposed + fused LoRA path. Gradient check PASS, TP=4 551.3 ms vs 547.5: the adapter ops saved are paid back by the transposed route's layout ops and a slower out forward. Reverted; evidence kept. Details in `../perf_analysis.md`, "Rejected or deferred candidates".

### Iteration 8: frozen vocab-parallel embedding table untilized once (wasted), landed

`FrozenVocabParallelEmbedding` in `models/mistral/frozen_embedding.py`: the TP embedding's `[32768, 5120]` shard was untilized by the ttml op on every step (1.5 ms per step per chip). Gradient check PASS, loss identical. **TP=4 full model 545.9 ms / 29.72 %** (best 542.9). Details in `../perf_analysis.md`, "Iteration 8".

## State after the resumed loop (2026-10-29)

TP=4 full model 598.7 (composite baseline) -> 556.5 (loop exit, iteration 3) -> **545.7 ms / 29.73 % MFU** (iteration 8 + the review-3 follow-up). Loss match on this path: 8-layer host-loss max |diff| 0.027 vs torch, TP=4 real-weight trajectory as the baseline (`../perf_analysis.md`, "Review round 3 follow-up"). Stage review round 3: `review_round3.md` (more-work-needed, all findings answered in the follow-up). The remaining ranking at TP=4 and what each item needs is in `../perf_analysis.md`, "Loop resumed": the base matmuls at ttnn's optimum for the TP shapes, inherent TP collectives, and three kernel projects (SDPA, attention on the fused heads layout, residual-fused RMSNorm forward / all-core forward).

## Exit

After three iterations the single-chip step projects to **1339 ms of cycle-time per 40-layer step** (composite baseline 1659), 17 % under the plan's 1620 ms target and 14 % under its kernel budget; at the clocks iteration 3's own runs delivered (1090 MHz MFU run, 1041 MHz profiled run) that is 1661-1739 ms wall against the target's 2006-2101 ms at those clocks (the clock fell through the loop as the kernels got denser; see the anomaly ledger in `../perf_analysis.md`). Full model, TP=4 over the box: **556.5 ms/step, 29.15 % MFU** of 594 TFLOP/s (baseline 598.7 / 27.10 %), real weights 574.8 ms on warm chips with the baseline's loss trajectory. The final path passes the PCC harness (14 tests), the loss comparison (max |diff| 0.026 against torch fp64 with bf16-rounded parameters, bar 0.10) and a TP=4 gradient check (every adapter gradient of the fused/transposed TP path within PCC ≥ 0.9999 / 1.4e-2 of the single-chip path; `tp_gradient_check.py`). With 16 layers now fitting, the 2/4/8/16 fit projects 1346 ms (1/2/4/8: 1339). Gap decomposition, memory peak (13.8 GiB after the forward at 8 layers, 279 MiB retained per layer), the final ranked table and every rejected candidate with its measurement: `../perf_analysis.md`, "Loop exit". Highest remaining items: gate_up forward at 64 % of the delivered peak (program config), SDPA backward at ~10x its roofline, the `fill_cache` tail writes (37 ms, one-core kernel), the residual/LoRA `dx` accumulation adds (33 ms), RMSNorm backward on 32 cores, the w2 LoRA skinny matmuls; overhead items above the 1 % (13 ms) threshold are listed in the gap decomposition with their rung and "deferred: target met".
