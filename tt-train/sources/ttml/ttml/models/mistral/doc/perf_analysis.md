# Performance analysis

Setup: `mistralai/Mistral-Small-24B-Base-2501`, LoRA r16/a32 on all four block linears, seq 1024, batch 1, AdamW, bf16, HiFi4 + fp32 accumulate, no recompute; metric: TP=4 full-model (40 layers) MFU against 594 TFLOP/s (4 x 148.5; numerator 96.37 TFLOP/step, `plan.md` §3); box: QB2, 4 Blackhole chips (110 cores, 31.83 GiB each), 1x4 ring.

Conventions (details: `D` = `optimization/evidence/perf_analysis_details.md`, "Setup and clock"):

- Every Tracy "ms" is cycles / 1.35 GHz (cycle-time); wall = cycle-time x 1350 / f_delivered. Delivered single-chip clock 1130-1160 MHz at the baseline, 1007-1104 MHz later; TP chips 1.10-1.26 GHz effective, run to run.
- Full-model TP=4 runs are the results (user priority, 2026-10-29); single-chip 8-layer runs and 1/2/4/8-layer depth fits only rank the next item. Single-chip MFU denominator 148.5 TFLOP/s.
- Plan target: 1620 ms cycle-time per 40-layer step at 1350 MHz (1562 kernels + 58 dispatch), 40.0 % MFU ±10 %; stretch 1259 ms / 51.6 %. Met at iteration 3 (1339 ms), loop resumed on the user's "why stop? continue".
- Evidence paths are relative to `doc/`; `O/` = `optimization/evidence/`.

## MFU history

Full model, TP=4, 40 layers, random init after a cool-down unless marked "warm" (real weights, no cool-down).

| iteration | change | ms/step (best) | MFU | evidence |
|---|---|---|---|---|
| 0 | composite baseline | 598.7 (594.7) | 27.10 % | `O/full_tp4_baseline_random.log` |
| 0 | baseline, real weights, warm | 641.9 (639.6) | 25.27 % | `O/full_tp4_baseline_real.log` |
| 1 | 3-op `LoraApply` | 574.3 (571.3) | 28.25 % | `O/iter1_loraapply_mfu_tp4.log` |
| 2 | transposed frozen weights | 564.5 (561.5) | 28.74 % | `O/iter2c_transposed_tp_mfu_tp4.log` |
| 3 | LoRA fused into the base matmul | 556.5 (549.9) | 29.15 % | `O/iter3_fused_mfu_tp4.log` |
| 3 | repeat after 5 min | 557.9 | - | `O/iter3_fused_tp_mfu_tp4.log` |
| 3 | exit, real weights, warm | 574.8 | - | `O/final_full_tp4_real.log` |
| 4 | `write_tile_block` + gate_up config | 558.5 (554.9) | 29.05 % | `O/iter4_tp_mfu_tp4.log` |
| 5 | batched LoRA-gradient all-reduce (fused wrappers) | 556.9 (551.1) | 29.13 % | `O/iter5_tp_mfu_tp4.log` |
| 5b | batched mode for the composite adapters | 563.2 (559.5) | 28.81 % | `O/iter5b_tp_mfu_tp4.log` |
| 6 | RMSNorm backward on all cores | 547.5 (545.0) | 29.63 % | `O/iter6_tp_mfu_tp4.log` |
| 7 (rejected) | qkv / out on the fused path at TP | 551.3 (544.3) | 29.43 % | `O/iter7_tp_mfu_tp4.log` |
| 8 | frozen vocab-parallel embedding | 545.9 (542.9) | 29.72 % | `O/iter8_tp_mfu_tp4.log` |
| review 3 | graph-free frozen paths | 545.7 (542.5) | 29.73 % | `O/review3_tp_mfu_tp4.log` |
| review 3 | real weights, warm | 560.4 | - | `O/review3_full_tp4_real.log` |
| 9 | residual add fused into all-core RMSNorm | 537.4 (533.9) | 30.19 % | `O/iter9_tp_mfu_tp4.log` |
| 10 | K-split `skinny_matmul` for the LoRA products | 536.1 (531.2) | 30.27 % | `O/iter10_tp_mfu_tp4.log` |
| 10 | real weights, warm | 543.0 | - | `O/iter10_full_tp4_real.log` |
| 11 | SDPA work lists (LPT rows; head-split KV backward) | 524.7 (520.2) | 30.92 % | `O/iter11_tp_mfu_tp4.log` |
| 12 | `lowrank_update` op: LoRA add and `dx += dh @ A` in one pass | **518.5 (511.8)** | **31.29 %** | `O/iter12_tp_mfu_tp4.log` |

Per-chip effective clocks and temperatures of each run: `D`, "MFU history". tt-smi per chip alongside every run: `O/<run>_tt-smi.txt`.

### Ranking runs (not results)

Single chip, chip 0; MFU at 8 of 40 layers against 148.5 TFLOP/s; projections from the 1/2/4/8-layer Tracy fit.

| iteration | run | ms/step (best) | MFU | 40-layer projection, cycle-time | evidence |
|---|---|---|---|---|---|
| 0 | 8 layers, wall | 434.0 (432.4); warm 453.4 | 33.32 % | 1658.6 (wall fit 1947) | `composite/evidence/train_8layers.log` |
| 1 | 8 layers, 1104 MHz (MFU run; the profiled run 1101) | 438.7 (435.8) | 32.96 % | 1592.5 | `O/iter1_loraapply_mfu_8layers.log` |
| 2 | 8 layers, ~1.1 GHz | 380.0 (376.8) | 38.06 % | 1398.7 | `O/iter2b_transposed_mfu_8layers.log` |
| 3 | 8 layers, 1090 MHz | 375.3 (372.6) | 38.53 % | 1339.0 (2/4/8/16 fit 1345.8) | `O/iter3_fused_mfu_8layers.log` |
| 4 | 8-layer profile only, 1007 MHz | 285.3 cycle-time | - | 1256.3 | `O/iter4_tracy_analysis.txt` |
| 6 | 8-layer profile | 282.9 cycle-time | - | ~1242 | `O/iter6_tracy_8layers_ranked_ops.txt` |
| review 3 | 8-layer profile | 283.44 cycle-time | - | - | `O/review3_tracy_8layers_ranked_ops.txt` |
| 9 | 8-layer profile | 281.7 cycle-time | - | - | `O/iter9_tracy_8layers_ranked_ops.txt` |

Baseline depth fit `device ms = 48.83 + 40.245 L` (residuals ≤ 0.6 %), wall fit `56.1 + 47.27 L`; per-iteration fits and class projections: `D`, "Iteration N".

## Ranked cost table

- Source: TP=4 after iteration 6, chip 2 of `O/iter6_tp_ranked_ops.txt`; cycle-time, per-layer rows scaled 4 -> 40 layers, fixed rows as measured (57.0 ms of kernels per 4-layer step per chip).
- State per row and the next rung: `D`, "Ranked cost table".
- Order after iterations 9-12: matmuls ~208, collectives ~56, layout ~48, norms ~32 (adds folded in), SDPA ~30 (was ~39), SwiGLU ~15, adapters ~9 (was ~19), fixed ~11, adds ~3 (was ~13).

| ms/step/chip | class | item | sub-items (ms) |
|---|---|---|---|
| ~208 | math | base matmuls at TP-local shapes | gate_up fwd 61.5 (76 %) / dX 55.2 (85 %), w2 fwd 36.9 (63 %) / dX 28.0 (83 %), qkv 7.6 + 7.8, out 5.9 + 4.8 |
| ~56 | collectives | 18 reduce-scatter `[1024, 5120]` + 18 all-gather per 4 layers | 35 + 21; 17 + 17 after review 3 |
| ~48 | layout | heads create / concat, fused-route concat / slice / transposes, LoRA-grad stack / unstack | ~28 + ~14 + ~6 |
| ~39 | math | SDPA, 8 heads per chip | bwd KV 18.3, bwd Q 10.5, fwd 10.4; after iteration 11: bwd KV 11.5 + reduce 1.1, bwd Q 8.6, fwd 9.2 = ~30, all three near the DRAM-bandwidth bound of single-row K/V and Q/dO re-reads |
| ~31 | math | RMSNorm | fwd 13.8 (32 cores) + bwd 17.5 (96 cores); fwd fused with the adds in iteration 9 |
| ~25 | math + glue | composite LoRA adapters of qkv / out at TP | skinny matmuls, scale, add; ~19 after iteration 10; ~9 after iteration 12 (`lowrank_update` folds the add) |
| ~18 | glue | residual and gradient adds `[1024, 5120]` | 31 per 4 layers; ~13 after iteration 9; ~3 after iteration 12 (7 adds per 4 layers left) |
| ~15 | math | SwiGLU | bwd 10.5 + fwd 4.4 |
| ~11 | fixed | LM head, cross-entropy, LoRA-grad collectives | fwd 3.0 + dX 2.8, CE chain 3.8, collectives 0.1 |

Earlier tables, kept in full: composite baseline at 40 layers (31 rows, single chip) and the single-chip table at loop exit with rooflines (19 rows): `D`, "Ranked cost table".

## Iteration 0: composite baseline (2026-10-29)

- **Change**: composite bring-up output (frozen-weight dW skipped, frozen table untilized once, 5-op `LoraLinear`); TP=4 path added (`train.py --tp`, vocab-parallel embedding and LM head).
- **Result**: TP=4 598.7 ms / 27.10 % (real weights 641.9 / 25.27 %, warm); single chip 8 layers 434.0 ms / 33.32 %, projection 1658.6 ms cycle-time = 1933-1985 ms wall at 1160-1130 MHz, 33.6-32.7 %.
- **Attribution**: classes at 40 layers: math 1512.5, glue 134.6, layout 11.5, wasted 0, host/dispatch ~3 (device busy 99.8 %; the plan's 58 ms does not exist). Top rows: gate_up fwd 424.9 at 44 % of peak vs its dX 273.5 at 68 %; LoRA glue chain 70.6 + 33.5 + 29.3.
- **Verification**: harness 11 passed; loss match max |diff| 0.075 (composite phase).
- **Evidence**: `composite/evidence/tracy_analysis.txt`, `clock_analysis.txt`, `op_sequence_1layer.txt`; `O/full_tp4_baseline_{random,real}.log`; plan comparison and next-item ranking in `D`, "Ranked cost table".

## Iteration 1 (2026-10-29): LoRA scale+add chain -> 3-op `LoraApply`

- **Change**: glue, rung reuse: `LoraApply` / `_LoraUpdateAdd` from `umales/qwen38` into `ttml/modules/lora.py` (A matmul, B matmul, `addalpha`; scale folded; `dh` all-reduced before `dA` under TP, which also fixes a silent `lora_A` divergence across chips).
- **Result**: TP=4 574.3 ms / 28.25 % vs 598.7 (-4.1 %); single chip cycle-time 370.40 -> 357.12 ms at 8 layers (-3.6 %), projection 1658.6 -> 1592.5 (-66.1); wall 438.7 vs 434.0 at 1104 (MFU run; 1101 in the profiled run) vs 1160 MHz.
- **Attribution**: scalar `multiply` 160/step (33.5 ms) and scalar grad scale 160/step (29.3) gone; two-input `BinaryNg` 480 -> 160 ops (70.6 -> 38.6); +320 tiny `[*,32]` scales (31.2 stale-shape row); glue 134.6 -> 72.9; TP profile 73.4 -> 70.9 ms per 4 layers; 16 x 54 µs new small reduce-scatters per 4 layers.
- **Verification**: harness 11 passed; Tracy residuals ≤ 0.6 %; ops/step unchanged 94 / 175 / 337 / 661.
- **Evidence**: `O/iter1_loraapply_{harness,mfu_tp4,mfu_8layers,clock_analysis,tracy_analysis}.*`, `O/iter1_loraapply_tracy_tp4_4layers`; hot-chip anomaly in Ledger; remaining glue and the iteration-order rationale in `D`, "Iteration 1".

## Iteration 2 (2026-10-29): base-linear matmuls on a transposed frozen weight

- **Change**: math, rung reuse: `ttml/modules/frozen_linear.py` stores frozen qkv / out / gate_up / LM head as `[in, out]` (allow-listed shapes), plain matmul forward on a tuned grid (8x11 for gate_up), input gradient as `(W_t @ g^T)^T`; w2 left untransposed (measured).
- **Result**: TP=4 564.5 ms / 28.74 % vs 574.3 (-1.7 %); single chip 8 layers 380.0 ms / 38.06 %, cycle-time 357.12 -> 313.66 (-12.2 %), projection 1592.5 -> 1398.7 (-193.8).
- **Attribution**: gate_up fwd 424.3 -> 293.1 (44 -> 63 % of delivered peak), gate_up dX 273.5 -> 218.9 (85 %), qkv dX 26.9 -> 22.1, out dX 18.6 -> 14.3, LM head fwd 19.1 -> 18.1 / dX 13.7 -> 10.9; layout +3.9 (dX-route transposes); TP kernels per chip 63.9 -> 58.8 ms.
- **Verification**: harness 13 passed (two new transposed-path cases); Tracy residuals ≤ 0.6 %; ops/step 661 -> 686.
- **Evidence**: `O/iter2_matmul_bench{,2}_n*.log`, `O/iter2_matmul_bench{3,4}.log`, `O/iter2b_transposed_*`, `O/iter2c_harness.log`, `O/iter2c_transposed_tp_*`; microbenchmark findings (a)-(d), the `core_grid` lesson and the within-iteration rejections in `D`, "Iteration 2".

## Iteration 3 (2026-10-29): LoRA update fused into the base matmul (K-concatenation)

- **Change**: glue + LoRA math, rung delete: `ttml/modules/fused_lora.py`, `y = [x \| x A^T] @ [[W_t], [s B^T]]`, two-tile tail rewritten per step with `ttnn.fill_cache`, one `W_cat_t @ g^T` backward; qkv / out / gate_up (w2 stays on `LoraApply`); TP tails carry the local B shard / `s / tp`.
- **Result**: TP=4 556.5 ms / 29.15 % vs 564.5 (-1.4 %), repeat 557.9; single chip 8 layers 375.3 ms / 38.53 % at 1090 MHz, cycle-time 313.66 -> 301.72, projection 1398.7 -> 1339.0 (-59.7, -4.3 %): 17 % under the 1620 ms target, 1661-1739 ms wall vs 2006-2101 at the run's clocks.
- **Attribution**: gate_up B fwd 33.0, `dh` 38.5, 160 `addalpha`s 37.9, qkv/out B/`dh` 3.4-3.6 gone; new tail writes 31.6 (`fill_cache` 0.79 ms, one core), concats 3.8, slices 3.7; glue 71.8 -> 36.4, math 1310.3 -> 1234.9, layout 16.6 -> 30.5; ops/step 686 -> 830.
- **Verification**: harness 14 passed (fused block rel err 4.18e-3 vs 4.58e-3 unfused); isolation test caught a tail-scale bug (0.5 rel err) before the profile; loss match max |diff| 0.026 (`O/final_loss_comparison.txt`); TP gradient check PASS (PCC ≥ 0.9999, rel 1.25-1.42e-2).
- **Evidence**: `O/iter3_fused_*`, `O/iter3_fused_isolate.py`, `O/tp_grad_check_{single,tp4_plain,tp4_fused}.log`, `O/final_*`; exit condition, gap decomposition and memory peak under "Loop exit"; full text in `D`, "Iteration 3" and "Loop exit".

## Iteration 4 (2026-10-29): multi-core tail write + swept gate_up program config

- **Change**: tail writes + math, rung extend: new metal op `ttml.ops.metal.write_tile_block` (`metal/ops/write_tile_block/`, multi-core in-place tiled block copy) replaces the one-core `fill_cache`; `FORWARD_PROGRAM_CONFIG` (10x10 grid, per_core_N 21, in0_block_w 2, out_subblock 4x1, out_block 16x7, bit-exact) for the single-chip gate_up forward from a 1D-mcast sweep.
- **Result**: TP=4 558.5 ms / 29.05 %, unchanged vs 556.5 / 557.9; single chip cycle-time 301.72 -> 285.30 at 8 layers, projection 1339.0 -> 1256.3 (-82.7, -6.2 %) at a delivered 1007 MHz.
- **Attribution**: gate_up tail 0.848 -> 0.057 ms (qkv 0.095 -> 0.048, out 0.082 -> 0.030); `UpdateKVCache` 31.6 + 3.0 + 2.5 -> `WriteTileBlock` 0.8; gate_up fwd 7.92 -> 7.35 ms per call (296.3 -> 254.4 per step, 64 -> 74 % of nominal peak); TP unchanged: config keyed on the single-chip shape, tail gain ~7 ms under the clock swing (chip 2 at 1.097 GHz vs 1.157).
- **Verification**: harness 14 passed; TP smoke 2 layers 42.3 ms; Tracy residuals ≤ 0.1 %; sweep 1624 distinct configs (1564 timed, the rest rejected by the L1 filter), fastest overall 7.09 ms deferred (max |diff| 3.1e-2).
- **Evidence**: `O/iter4_write_tile_block_bench.{py,log}`, `O/iter4_matmul_sweep1d_gateup_fwd.log`, `O/iter4_tracy_analysis.txt`, `O/iter4_diff_vs_iter3.txt`, `O/iter4_clock_analysis.txt`, `O/iter4_tp_{mfu_tp4,analysis}.*`; `D`, "Iteration 4".

## Iteration 5 (2026-10-29): batched TP all-reduce of the partial LoRA gradients

- **Change**: collectives, rung delete: the fused TP wrappers keep `h` / `dh` partial per rank (reductions are linear) and `reduce_partial_lora_grads` all-reduces the stacked `dA` / `dB` once per step (`fuse_lora_into_frozen_weights(batched_tp_grads=True)`, `tp_partial_grads`).
- **Result**: TP=4 556.9 ms / 29.13 % vs 558.5 (-0.3 %); the step follows the slowest chip's compute (chip 2 at 1.119 GHz vs 1.164-1.261), the removed collectives were waiting time on the other three.
- **Attribution**: `[1024, 32]` reduce-scatters 16 -> 8 per 4 layers (3.42 -> 1.06 ms on chip 0; 130-214 us each, 3.4 ms per 4 layers before); two stacked collectives 28 + 26 us per step; chip 0 collective time 9.1 ms unchanged (big reduce-scatters 268 -> 308 us); ~24 ms per step expected on an even box.
- **Verification**: harness 14 passed; TP gradient check PASS (PCC ≥ 0.9999, rel 1.25-1.39e-2, loss 12.8125 identical, 2 partials per layer).
- **Evidence**: `O/iter4_tp_ranked_chip0.txt`, `O/iter5_harness.log`, `O/iter5_tp_grad_check_tp4_batched.log`, `O/iter5_tp_{mfu_tp4,analysis}.*`; `D`, "Iteration 5".

## Iteration 5b (2026-10-29): batched TP gradient mode for the composite adapters

- **Change**: collectives: the composite qkv (column-parallel, `reduce_axis=None`) and out (row-parallel, partial update added before the output all-reduce) adapters at TP-local shapes also keep partial `dA` / `dB`; no LoRA path issues a per-layer collective; 4 partials per layer.
- **Result**: TP=4 563.2 ms / 28.81 % vs 556.9, at 1.106-1.154 GHz effective vs 1.119-1.259 (chips 52-76 C in both runs; iteration 5's own text rounds its range to 54-76); neutral in wall on this box, ~12 ms per step net on an even box.
- **Attribution**: compute rows identical within 0.1 % in cycle-time; `[1024, 32]` reduce-scatters 8 -> 0 per 4 layers; stacked collectives 29 + 27 + 27 us; kernel-time difference (57.3 vs 55.9 ms, chip 2) is waiting in the big reduce-scatters (3.0 vs 1.9 ms); stack / unstack glue 0.63 ms per step per chip (~5-6 ms per 40 layers) vs ~18 ms of collectives removed.
- **Verification**: harness 15 passed (with iteration 6 on); TP gradient check PASS, loss identical; tile-aligned stacking benchmark: no gain (0.30 vs 0.23, 0.27 vs 0.26 ms).
- **Evidence**: `O/iter5b_tp_{mfu_tp4,analysis,ranked_ops}.*`, `O/iter6_harness.log`, `O/iter6_tp_grad_check_tp4.log`, `O/review3_stack_bench.{py,log}`; `D`, "Iteration 5b".

## Iteration 6 (2026-10-29): RMSNorm backward on all cores

- **Change**: math, rung reuse: `metal/ops/rmsnorm_bw_allcore/` (two-phase all-core backward from `umales/qwen38` 67ad5aeb349, 96 cores, no dgamma pass for frozen gamma), `models/mistral/rmsnorm_allcore.py`, `RMSNormLayer.rmsnorm_op` hook; `train.py --no-allcore-rmsnorm` keeps the old path.
- **Result**: TP=4 547.5 ms / 29.63 % vs 556.9 / 563.2 (-1.7 % / -2.8 %) at 1.135-1.193 GHz; single chip 8-layer cycle-time 285.3 -> 282.9, projection 1256 -> ~1242.
- **Attribution**: `RMSNormBackward` 0.37 ms per call on 32 cores (29.7 ms per 40 layers, 6x roofline) -> 6.23 -> 1.83 + 1.56 = 3.39 ms per 8-layer step (-46 %); per TP chip 2.93 -> 0.96 + 0.79 = 1.75 ms per 4 layers (-1.18, ~-12 ms per step); other rows unchanged except one extra norm backward + qkv dX + all-reduce for the frozen embedding output (removed in review 3).
- **Verification**: new harness case (out PCC 0.999998, dx 0.999997 / rel 3.2e-3); harness 15 passed; TP gradient check PASS; first test run failed on the Function's gradient arity (fixed).
- **Evidence**: `O/iter6_build_and_test.log`, `O/iter6_rmsnorm_test.log`, `O/iter6_harness.log`, `O/iter6_tp_grad_check_tp4.log`, `O/iter6_tracy_8layers_ranked_ops.txt`, `O/iter6_tp_{mfu_tp4,analysis,ranked_ops}.*`; `D`, "Iteration 6".

## Iteration 7 (2026-10-29, rejected): qkv / out TP-local shards on the fused path

- **Change**: trial: `(5120, 1536)` and `(1024, 5120)` added to `TRANSPOSED_SHAPES` so qkv / out take the transposed + fused LoRA path at TP=4.
- **Result**: TP=4 551.3 ms (best 544.3) / 29.43 % vs 547.5 (best 545.0): neutral; reverted.
- **Attribution**: kernels per chip 57.16 vs 57.01 ms per 4 layers: adapter matmuls and adds -1.0 ms, transposed route +0.39 (`[5184, 1024]` slice) + 0.4 (concats / transposes), out forward `[1024, 1088] x [1088, 5120]` 0.65 vs 0.59.
- **Verification**: TP gradient check PASS.
- **Evidence**: `O/iter7_tp_{mfu_tp4,analysis,ranked_ops,grad_check_tp4}.*`; `D`, "Iteration 7".

## Iteration 8 (2026-10-29): frozen vocab-parallel embedding table untilized once

- **Change**: wasted, rung delete: `FrozenVocabParallelEmbedding` (`frozen_embedding.py`): the TP embedding's `[32768, 5120]` shard was untilized by `ttml.ops.embedding.embedding` on every step; same masking, lookup, zeroing and all-reduce without it, no gradient to ids or table.
- **Result**: TP=4 545.9 ms / 29.72 % vs 547.5 (-0.3 %) at 1.22-1.24 GHz.
- **Attribution**: `UntilizeCodegen` 1.52 ms per step per chip gone from the profile.
- **Verification**: TP gradient check PASS, loss identical to single chip (the single-chip harness has no vocab-parallel path).
- **Evidence**: `O/iter8_tp_{mfu_tp4,analysis,ranked_ops,grad_check_tp4}.*`; `D`, "Iteration 8".

## Review round 3 follow-up (2026-10-29): graph-free frozen paths

- **Change**: wasted: `frozen_embedding.py` returns the lookup outside the autograd graph and `rmsnorm_allcore.py` sends no-grad inputs through the C++ norm op (finding 1: layer-0 norm backward, qkv input gradient and its all-reduce were computed for the frozen embedding output and discarded).
- **Result**: TP=4 545.7 ms / 29.73 % vs 545.9 (under run-to-run noise) at 1.18-1.25 GHz; real weights 560.4 ms warm.
- **Attribution**: per 4-layer step, chip 2: reduce-scatter 18 -> 17, all-gather 18 -> 17, `RMSNormBackward` 9 -> 8, qkv dX 4 -> 3, residual adds 32 -> 30, kernels 56.3 -> 55.6 ms; single chip 8 layers 283.44 ms (282.87 in iteration 6, noise).
- **Verification**: harness 15 passed; TP gradient check PASS, unchanged values; loss match on the final path max |diff| 0.027 (tol 0.10), mean +0.006, PASS; TP=4 real-weight trajectory 0.4453 -> 0.1406 -> 0.4512 as the baseline's within bf16 noise.
- **Evidence**: `optimization/review_round3.md`, `O/review3_followup.log`, `O/review3_{harness,tp_grad_check_tp4,tp_mfu_tp4,tp_analysis,tp_ranked_ops,tracy_8layers_ranked_ops,loss_comparison,full_tp4_real}.*`; `D`, "Review round 3 follow-up".

## Iteration 9 (2026-10-30): residual add fused into an all-core RMSNorm

- **Change**: math + glue, rung extend: new metal op `add_rmsnorm_fw_allcore` (`[sum, out, rms]`, two phases on 96 cores, fp32 `1/C`), `rmsnorm_bw_allcore(..., residual_grad)`, `models/mistral/fused_residual_norm.py` re-pairs the layer boundaries; `train.py --no-fused-residual-norm` keeps the block path. Also fixes a latent CB wrap hazard in the ported backward (Ledger).
- **Result**: TP=4 537.4 ms / 30.19 % vs 545.7 (-1.5 %) at 1.220-1.228 GHz on all chips; single chip 8-layer cycle-time 283.4 -> 281.7 (~-8.5 ms per 40 layers).
- **Attribution**: chip 0, per 4 layers, collectives excluded: norm fwd 1.38 -> 0.15 + 0.62 + 0.61 = 1.39 (unchanged: the fused forward moves 3x the bytes at ~420 GB/s), same-shape adds 2.27 -> 1.29, norm bwd 1.64 -> 1.85; kernels 55.73 -> 54.87 (steady steps 49.84 -> 48.92), -0.86 ms per 4 layers, ~-8 ms per 40; single chip: 16 fused norms 1.12 + 1.10 ms replace 2.45 ms of norms and 31 adds (6.6 -> 4.8 ms), backward +0.48.
- **Verification**: kernel tests at 32 and 64 tile-rows (sum PCC 0.999999, out 0.999997, grads 0.999997); 2-layer model equals the block path (loss 11.9375 both, 16 LoRA grads PCC 0.99994-0.99998); harness 20 passed; TP gradient check PASS; loss match rerun after iteration 10.
- **Evidence**: `O/iter9_{kernel_tests,model_test,harness,ninja}.log`, `O/iter9_tp_grad_check_tp4.log`, `O/iter9_tracy_8layers_ranked_ops.txt`, `O/iter9_tp_{mfu_tp4,analysis,ranked_ops}.*`; `D`, "Iteration 9".

## Iteration 10 (2026-10-30): K-split matmul for the rank-wide LoRA products

- **Change**: math, rung extend: new metal op `skinny_matmul` (`metal/ops/skinny_matmul/`, N <= 64, K split into 13 slices x 4-tile-row groups = 104 items, fp32 partials summed in phase B, HiFi4) routed from `modules/lora.py::skinny_matmul` for `x @ A^T` and `g @ B` at K >= 2048 (`SKINNY_MIN_K`).
- **Result**: TP=4 536.1 ms / 30.27 % vs 537.4 (-0.2 % wall) at 1.148-1.210 GHz (lower than iteration 9's 1.220-1.228: the wall understates the cycle-time gain).
- **Attribution**: chip 0, per 4 layers: `x @ A^T` K=5120 0.614 -> 0.365 (77 -> 46 us), K=8192 0.487 -> 0.260 (122 -> 65 us), `g @ B` 0.304 -> 0.180 (76 -> 45 us); routed rows 1.405 -> 0.805 (-0.60, ~-6 ms per 40 layers); math 40.78 -> 40.11; kernels 54.87 -> 54.19 (steady 48.92 -> 48.12); ops 508 -> 524 per step.
- **Verification**: standalone vs `ttnn.matmul` PCC 0.999999, rel 1.4-1.5e-3 at five shapes (78 -> 48 us wall at K=5120, 123 -> 68 at K=8192); harness 20 passed (32 with review 6's 12 `test_skinny_matmul_matches_torch` cases); TP gradient check PASS; loss match max |diff| 0.0273 (tol 0.10, mean +0.0056) PASS, TP=4 real-weight trajectory 0.4414 -> 0.1406 -> 0.4473 as the baseline's.
- **Evidence**: `O/iter10_skinny_check.{py,log}`, `O/iter10_harness.log`, `O/iter10_tp_grad_check_tp4.log`, `O/iter10_tp_{mfu_tp4,analysis,ranked_ops}.*`, `O/review6_skinny_tests.log`, `O/iter10_loss.log`, `O/iter10_loss_comparison.txt`, `O/iter10_full_tp4_real.log`; `D`, "Iteration 10" and "Loss match after iterations 9 and 10".

## Iteration 11 (2026-10-30): SDPA work distribution

- **Change**: math, rung extend, kernels we own: `WORK_LIST` mode in `sdpa_fw` and `sdpa_bw_q` (host LPT schedule of the causal rows, `lpt_schedule` in `metal/common/program_utils.hpp`, per-core row lists as runtime args; the pair scheme had 18 of 110 cores doing two pairs) and `HEAD_SPLIT` in `sdpa_bw_kv` (one item per (K/V row, query head), 256 items on 110 cores instead of 64 rows on 64 cores with a 32:1 work spread; fp32 per-head partials summed by the new `ttml_sdpa_kv_bw_reduce` op).
- **Result**: TP=4 524.7 ms / 30.92 % vs 536.1 (-2.1 %) at 1.149-1.273 GHz effective (iteration 10's run 1.148-1.210), chips 50-75 C.
- **Attribution**: chip 0, per 4 layers, cycle-time: SDPA forward 1.072 -> 0.915 (268 -> 229 us per call), backward Q 1.084 -> 0.862 (271 -> 216), backward KV 1.839 -> 1.154 + reduce 0.113 (460 -> 289 + 28); SDPA 3.995 -> 3.044 (-0.95 ms, ~-9.5 ms per 40 layers per chip); other rows unchanged within 1 %. Smaller than the core-balance arithmetic (1.7x / 3.3x) because all three kernels re-read K/V (forward, Q backward: 67 MB per call) or Q/dO/lse/u (KV backward: ~100 MB) per single Q tile-row and now run at 280-350 GB/s: bandwidth-bound; the next rung is Q-chunking (several Q rows per item sharing the K/V reads), `D`, "Iteration 11".
- **Verification**: standalone vs the previous kernels at six shapes (`O/iter11_sdpa_check.log` vs `O/iter11_baseline.log`): forward and dQ bit-identical, dK / dV PCC 0.999999, max |diff| one bf16 ulp (fp32 partials summed in a different order); harness 35 passed (new `test_sdpa_causal_gqa_forward_and_gradients` at three shapes vs torch fp64); TP gradient check PASS. Bug found on the way: the KV kernel's 2-arg `pack_reconfig_data_format(cb_grad_key, cb_attention_weights)` skips the reprogram when both CBs are fp32 and left the packer in its start-up bf16 state; unconditional reconfig under `HEAD_SPLIT`.
- **Evidence**: `O/iter11_baseline.{sh,log}`, `O/iter11_build_check.{sh,log}`, `O/iter11_sdpa_check.{py,log}`, `O/iter11.{sh,log}`, `O/iter11_harness.log`, `O/iter11_tp_grad_check_tp4.log`, `O/iter11_tp_{mfu_tp4,analysis,ranked_ops}.*`, `O/iter11_tp_mfu_tp4_tt-smi.txt`.

## Iteration 12 (2026-10-30): low-rank update in one pass

- **Change**: glue + math, rung write new: `metal/ops/lowrank_update/` (`out = acc + s * a @ b`, one K tile: the product per output tile in fp32 DEST, the acc tile added there, one bf16 rounding; items = output tiles column-major, the a column resident in L1, 110 cores) routed from `modules/lora.py::lowrank_update` for `base + s * h @ B^T` (composite qkv / out adapters, forward) and `dx += dh @ A` (fused gate_up / w2 backward); replaces a 33-88-core matmul writing a bf16 intermediate plus an add per call.
- **Result**: TP=4 518.5 ms / 31.29 % vs 524.7 (-1.2 %) at 1.111-1.197 GHz effective (iteration 11's run 1.149-1.273), chips 49-73 C.
- **Attribution**: chip 0, per 4 layers, cycle-time: the two LoRA-product matmul rows 0.567 + 0.534 and 16 adds (0.641 of the `[1024, 8192]`-labelled row, 0.357 of the `[1024, 1536]` row) -> `LowrankUpdate` 0.520 + 0.378 + 0.133; -1.07 ms per 4 layers (~-11 ms per 40 layers per chip); 7 adds per 4 layers remain (0.254); math 39.21 -> 37.89, glue 4.64 -> 3.59; kernels excluding collectives ~47.5 -> ~46.3.
- **Verification**: standalone vs `ttnn.matmul` + `addalpha` at five shapes PCC 0.999998-0.999999, rel 1.5-2.0e-3 (one rounding instead of two), 124 -> 72 us (N=5120), 188 -> 105 (8192), 52 -> 36 (1536); harness 38 passed (new `test_lowrank_update_matches_torch` at three shapes incl. two images); TP gradient check PASS. Bug found on the way: the compute kernel skipped the pop of the column's w tile when the image changed (two-image case wrong, PCC 0.99985); fixed, covered by the two-image test.
- **Evidence**: `O/iter12_lowrank_check.{py,log}`, `O/iter12_build_check.{sh,log}`, `O/iter12_lora_tests.log`, `O/iter12.{sh,log}`, `O/iter12_harness.log`, `O/iter12_tp_grad_check_tp4.log`, `O/iter12_tp_{mfu_tp4,analysis,ranked_ops}.*`.

## Loop exit (2026-10-29, after iteration 3) and resumed state

- Exit condition met: 1339 ms cycle-time (2/4/8/16 fit 1345.8; 16 layers 563.6 ms device / 694.4 wall at 1100 MHz, 24.8 GiB) vs the 1620 ms target, 17 % under; wall 1661-1739 vs 2006-2101 ms at the run's clocks (ratio 0.83); stretch 1259 ms 6 % away. Overhead threshold 1 % = 13 ms.
- Full-model report at exit: 556.5 ms / 29.15 % random init; 574.8 ms real weights warm, loss 0.4453 -> 0.1406 -> 0.4531 vs the baseline's 0.4453 -> 0.1436 -> 0.4551.
- Final verification at exit: harness 14; TP gradient check PASS on the composite and the fused paths (first attempt un-interleaved, Ledger); loss match max |diff| 0.026 (composite 0.075); device bf16 loss quantised to 0.0625.
- Memory peak: 13.79 GiB after the forward at 8 layers (weights + LoRA state 11.61), 279 MiB retained per layer (composite 778); TP=4 40 layers ~11 GiB weights + ~2.3 GiB activations per chip (scaled, not probed).
- Resumed loop (iterations 4-10): TP=4 556.5 -> 547.5 (6) -> 545.9 (8) -> 545.7 (review 3) -> 537.4 (9) -> 536.1 ms / 30.27 % (10); single-chip projection 1339 -> ~1242 (attribution only). Remaining: matmuls at ttnn's optimum for the TP shapes, inherent collectives, rank-16 layout glue, SDPA and attention-on-fused-heads kernel projects.

Gap decomposition at exit (cycle-time, 40 layers; notes per term in `D`, "Loop exit"):

| term | ms |
|---|---|
| roofline of the useful math | 744 |
| math inefficiency (big matmuls 771 vs 555, SDPA 124 vs 14, SwiGLU 57 vs 42, RMSNorm 42 vs 8, LoRA skinny ~110 vs 41, LM head 29 vs 18.5, CE 11 vs 1.6) | 491 |
| overhead: glue (adds and `dx` accumulations 32.6, loss-grad chain 1.4) | 36 |
| overhead: layout (heads 11.5, dX transposes 3.8, fusion concats / slices 7.5, tail transposes) | 31 |
| overhead: tail writes (`fill_cache` 31.6 + 3.0 + 2.5; 0.8 after iteration 4) | 37 |
| wasted | 0 |
| **step (device kernel sum)** | **1339** |
| host / dispatch (outside the sum; busy 99.8 %, 0.9-2 ms between steps) | ~3 |
| delivered clock x 1.24-1.30 (1090 / 1041 MHz) -> wall | 1661-1739 |

## Rejected

| candidate | cost (ms/step, cycle-time) | attempt | reason |
|---|---|---|---|
| TP-local program configs (gate_up / w2 / qkv / out fwd, gate_up / w2 dX) | ~208 (TP matmul row) | 1D-mcast sweeps `O/iter5_matmul_sweep1d_gateup_fwd_tp.log`, `O/iter7_matmul_sweep1d_*.log` | ttnn's choice at the optimum: 1.67 vs 1.66, 1.01 vs 0.99, 0.22 vs 0.21 ms; qkv 0.25 vs 0.46 (worse); dX shapes 0 configs tried |
| Iteration 7: qkv / out TP shards on the fused path | ~25 (adapter row) | measured, `O/iter7_tp_*` | 551.3 vs 547.5 ms; -1.0 ms adapters paid back by +0.79 layout and a slower out fwd (0.65 vs 0.59) |
| HiFi2 math fidelity for the base matmuls | 15-25 % of ~770 | microbenchmark `O/iter2_matmul_bench_n*.log` | rel err 4.7e-3 vs HiFi4; HiFi4 + fp32 accumulate is a user requirement |
| Fastest-overall gate_up 1D config (7.09 vs 7.35 ms) | ~10 (single chip only) | sweep `O/iter4_matmul_sweep1d_gateup_fwd.log` | max \|diff\| 3.1e-2 (one bf16 ulp); not taken without a loss rerun; deferred |
| dX^T route for w2 (`32768 -> 5120`) | 121.6 (w2 dX) | `O/iter2_matmul_bench3.log` | 3.72 -> 5.29 ms; the `[32768,1024]` transpose costs more than it saves |
| 8x11 grid for the LM head forward | 20.2 | `O/iter2_matmul_bench3.log` | 58.97 vs 21.48 ms, L1 overflow into a slow path |
| One-tile weight tail (K = 161 tiles) | 254-296 (gate_up fwd) | `iter3` tail measurement | +5 % (7.92 vs 7.51 / 7.59 ms for K = 5152 vs 5120 / 5184); two tiles used |
| `slice_write` / `paged_fill_cache` for the tail write | 37 (tail writes) | tried in iteration 3 | row-major only / page-table layout rejected the view; `fill_cache` then `write_tile_block` |
| Tile-aligned stacking of the LoRA partials | ~6 (stack / unstack glue) | `O/review3_stack_bench.{py,log}` | 0.30 vs 0.23 and 0.27 vs 0.26 ms, no gain; the 0.18 ms variant needs a 32-row-padded `dA` in the optimizer, deferred |
| LoRA B stored transposed | 22.8 -> ~11 (gate_up `dB`) | microbenchmark, iteration 2 (0.59 -> 0.39 ms) | queued behind the skinny-matmul item, not taken |
| Host/dispatch | ~3 | cycle analysis | the plan's 58 ms does not exist (device busy 99.8 %) |
| 320 tiny per-adapter scale ops | 31 (stale-shape row) | iteration 3 fusion | absorbed for qkv / out / gate_up; w2's remain (~8) |
| Gamma re-read per work item in the fused norm forward | ~0.2 per 4 layers per chip | not attempted | multicast or resident copy; listed |
| `UnpackToDestFp32` read-back of fp32 partials (three all-core ops) | 0 | review round 6 note | relative 2^-11, below bf16 rounding; PCCs 0.999997-0.999999; left as is |
| TP collective parameters (links, workers per link, chunks per sync, buffers per channel) through env knobs in `ttnn_fixed/distributed/ttnn_ops.cpp` | 6.8 ms per 4 layers per chip on chip 0 (17 reduce-scatters at 281 us + 17 all-gathers at 120 us) | standalone sweep on the 1x4 mesh at the step's shapes (`O/iter12_ccl_sweep.{py,sh,log}`): reduce-scatter 122 us, all-gather 114 us with the defaults (2 links, 8 workers per link); 1 link 217 / 209, 1-2 workers 176 / 148-278, chunks per sync 1-4 slower for the all-gather, buffers per channel no effect | the library defaults are the fastest; the in-step 281 / 120 us vs 122 / 114 standalone is waiting for the slower chips (1.149-1.273 GHz effective), i.e. the collectives carry the clock skew, not a transfer cost; nothing kernel-level to gain |

Full text of each: `D`, "Rejected".

## Ledger

- Iteration 1 hot chip (resolved): first TP run 816.9 ms, chip 3 at 1.68x cycles on every matmul after a 13-min TP run without cool-down (31 % idle on the others); cool rerun chips within 3 %. Protocol: cool-down, tt-smi per chip, per-chip matmul ratio check. `O/iter1_loraapply_mfu_tp4_hot.log`.
- Single-chip delivered clock fell through the loop: profiled 1130 / 1101 / 1074 / 1041 / 1007 MHz (baseline, 1-4), MFU runs 1160 / 1104 / 1119 / 1090; chip 0 at 150-163 W; cycle-time -18.5 % vs wall -13.5 % at 8 layers. Likely a board power limit; thermal / run-order confound not excluded (same code 1096 vs 1041 MHz an hour apart). Controlled: every wall number carries its run's clock.
- TP effective clock swings ±5 % run to run at equal temperature (1.10-1.26 GHz, 48-76 C, 63-85 W, AICLK reads 1350): chip 2 slow in iterations 4-5 but never the hottest (draws 10-15 W more); cause not identified. A/B decisions on per-chip cycle-time tables, not wall ms.
- Host/dispatch gap misread in the first draft: wall - Tracy = 14.0-63.6 ms is the clock conversion, not dispatch (device busy 99.8 %).
- Stale `BinaryNg` shape labels in Tracy (first call of a program hash): glue rows keyed by sequence position; times correct, labels not.
- Iteration 2 first profile without `core_grid`: 1.5-1.8x slower default config (gate_up dX 274 -> 358 ms); microbenchmarks must mirror the exact call.
- Iteration 3 tail written without the LoRA scale on the first run (every adapter gradient 50 % off): caught by the harness; harness runs before every profile.
- TP gradient check, first attempt: no rank interleave of the fused rows, uncorrelated gradients on every path; descent check (12.8125 -> 12.6250, prediction 0.23) showed two different models; fixed, invalid artefacts removed.
- Frozen-embedding backward computed and discarded (review 3, finding 1): ~0.9 ms per step per chip from iterations 6 / 8's Functions; removed in the follow-up.
- Latent CB wrap hazard in the ported all-core backward (second work item per core, invisible at 96 items on 110 cores): found by `skinny_matmul`'s first run (PCC 0.01 at K=5120); all three all-core kernel sets now move full blocks, two-items-per-core tests added.
- Iteration 6 first test run: three gradients returned for two inputs; fixed.
- fp32 partial read-back through the default unpack path (~19-bit mantissa) in the three all-core ops: relative 2^-11, invisible in the PCCs; left as is (review 6).
- Baseline real-weight TP run 43 ms slower than the random-init run minutes earlier: warm chips.
- Numbers that appear twice with different sources: iteration 1 clock 1104 (MFU run) / 1101 MHz (profiled run); iteration 4 sweep 1624 distinct / 1564 timed configs; baseline gate_up fwd 424.9 (depth fit) / 424.3 (per-call sum at 8 layers), LM head 20.2 / 13.6 (fit) vs 19.1 / 13.7 (per call); gate_up fwd after iteration 2 at 63 % (7.33 ms per call = 94 TFLOP/s of 148.5); iteration 5 temperatures 52-76 C per tt-smi.

Full ledger entries: `D`, "Ledger".
