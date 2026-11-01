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
| 12 | `lowrank_update` op: LoRA add and `dx += dh @ A` in one pass | 518.5 (511.8) | 31.29 % | `O/iter12_tp_mfu_tp4.log` |
| 13 | `transpose_split` op: dX and s dh from the fused backward product in one pass | 516.4 (512.0) | 31.42 % | `O/iter13_tp_mfu_tp4.log` |
| 14 | fused vocab-parallel cross-entropy backward (`vp_cross_entropy_bw`) | 512.5 (503.3) | 31.66 % | `O/iter14_tp_mfu_tp4.log` |
| 14 | real weights, warm | 526.1 (521.0) | - | `O/iter14_full_tp4_real.log` |
| 15 | SDPA forward chain mode: K/V read once per chain of cores, forwarded core to core | **506.0 (502.4)** | **32.06 %** | `O/iter15_tp_mfu_tp4.log` |
| 16 | SDPA backward chain mode (Q and KV kernels) | 508.4 (502.2) | 31.91 % | `O/iter16_tp_mfu_tp4.log` |
| 17 | single-phase RMSNorm backward (`rmsnorm_bw_rows`) | 505.6 (499.4) | 32.09 % | `O/iter17_tp_mfu_tp4.log` |
| 18 A | `skinny_contract` op for the wide LoRA dB products | 498.0 (484.6) | 32.58 % | `O/iter18_tp_mfu_tp4.log` |
| 18 B | LoRA dB / dA on ttnn's `[r, N]` orientation (+ transpose), op for the narrow dB | 503.9 (495.1) | 32.19 % | `O/iter18b_tp_mfu_tp4.log` |
| 19 | single-phase fused add + RMSNorm forward (`add_rmsnorm_fw_rows`) + 2D w2-forward matmul config | **495.4 (487.9)** | **32.75 %** | `O/iter19_tp_mfu_tp4.log` |
| 19 | real weights, warm | 509.1 (503.0) | - | `O/iter19_full_tp4_real.log` |
| 22 | SDPA backward: Q kernel in K/V blocks, one read barrier per stream chunk in the three chain injectors | **494.5 (486.9)** | **32.81 %** | `O/iter22_tp_mfu_tp4.log` |
| 22 | real weights, warm | 504.9 (498.5) | - | `O/iter22_full_tp4_real.log` |
| 23 | SDPA chain forwarding on the writer RISC / second NoC (forward, Q, KV), chains reversed | **480.3 (473.1)** | **33.78 %** | `O/iter23_tp_mfu_tp4.log` |
| 23 | real weights, warm | 502.2 (495.1) | - | `O/iter23_full_tp4_real.log` |
| 24 | w2 K-concat removed: SwiGLU writes the wide LoRA input, `skinny_matmul` writes h into its tail | 481.2 (470.6) | 33.72 % | `O/iter24b_tp_mfu_tp4.log` |
| 24 | real weights, warm | 497.9 (490.7) | - | `O/iter24b_full_tp4_real.log` |
| 25 | low-rank update fused into `transpose_split` (one DEST pass per dx tile) | **476.7 (468.2)** | **34.03 %** | `O/iter25_tp_mfu_tp4.log` |
| 25 | real weights, warm | 492.2 (489.6) | - | `O/iter25_full_tp4_real.log` |
| 26 | SDPA chain credits (three chunks in flight), 8-row K/V chunks in the forward, forward chains of 8 cores | **472.5 (465.9)** | **34.33 %** | `O/iter26_tp_mfu_tp4.log` |
| 26 | real weights, warm | 493.2 (481.4) | - | `O/iter26_full_tp4_real.log` |
| 27 | per-round chain stream bounds (forward / Q: up to the round's heaviest row; KV: from its lightest item), backward chains of 7 | **468.1 (465.4)** | **34.66 %** | `O/iter27_tp_mfu_tp4.log` |
| 27 | real weights, warm | 487.5 (483.3) | - | `O/iter27_full_tp4_real.log` |
| 28 | KV backward in query-row blocks (dP never leaves DEST, dV / dK contracted over the block) | **468.4 (463.5)** | **34.63 %** | `O/iter28b_tp_mfu_tp4.log` (first build `O/iter28_tp_mfu_tp4.log`: 464.7 (460.6) / 34.92 %) |
| 28 | real weights, warm | 486.9 (480.1) | - | `O/iter28b_full_tp4_real.log` |

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

- Source: TP=4 after iteration 27, chip 0 of `O/iter27_tp_ranked_ops.txt` (2 steady steps, cycle-time); per-layer rows x10 (4 -> 40 layers), fixed rows as measured; 49.4 ms of kernels per 4-layer step on chip 0, of which 13.5 collectives in this profile (iteration 26's profile: 44.0 / 4.5; chips 2-3 of 27: 46.4 / 7.6 and 42.8 / 3.9: the collective rows carry the clock skew of the run).
- Collectives: the standalone transfer is 17 x (122 + 114 us) = 4.0 ms per 4 layers (iteration 12 sweep); overlap with compute not tried.
- Older tables (composite baseline, loop exit with rooflines, TP after iterations 6, 12 and 18): `D`, "Ranked cost table".

| ms/step/chip | class | item | rows (ms per 4 layers, chip 0, iteration 27) | remaining rung |
|---|---|---|---|---|
| ~201 | math | base matmuls at TP-local shapes | gate_up fwd 6.15 (76 % of HiFi4 peak) / dX 5.52 (85 %), w2 fwd 3.25 (72 %, 2D config) / dX 2.80 (83 %), qkv 0.76 + 0.59, out 0.59 + 0.48 | ttnn's best configs (1D sweeps iterations 4 / 7, 2D sweep 19); further gain = ttnn matmul kernel work |
| 40-90 | collectives | 17 reduce-scatter `[1024, 5120]` + 17 all-gather `[1024, 1280]` per 4 layers | chip 0 6.92 + 2.12 in this profile (2.33 + 1.86 in iteration 26's), waiting for the slowest chip included | transfer at the library optimum (iteration 12); overlap with the LoRA gradient products not tried |
| ~25 | math + glue | LoRA adapters (rank 16) | products 0.47 + 0.37 + 0.20 + 0.19 + 0.17, `skinny_matmul` 0.29 + 0.22 + 0.15, `lowrank_update` (composite qkv / out) 0.26 + 0.13 | products at 85-88 % of bandwidth, narrow ones launch-bound (iteration 18); fuse dA / dB launches (~1 ms/step) |
| ~24 | math | RMSNorm | bwd rows 1.33, fused fwd 1.03 (one phase), first-layer plain fwd 0.15 | 126 / 165 us per call vs 100 / 125 us bandwidth rooflines; pass 2 of the backward on the FPU (-15 %, 19-bit operands: iteration 20 variant A, admissible under the 10-31 allowance) |
| ~17 | math | SDPA, 8 heads per chip | bwd KV 0.72 + reduce 0.11, bwd Q 0.44, fwd 0.40 (110 / 345 us per call vs 14.5 / 36 us FLOP rooflines) | stages per unit of work (iteration 26 attribution: math 35-45 % of each kernel): block the KV backward (iteration 28), then the forward's per-chunk stages |
| ~13 | math | SwiGLU | bwd 0.81 + fwd 0.45 | bwd at ~300 GB/s (mechanism of the iteration-24 drop open, Ledger) |
| ~12 | layout | fused-route transposes / splits / concats, heads, RoPE | `TransposeSplit` 0.41 + 0.25 (with the fused low-rank update), gate_up `Concat` 0.21, RoPE / heads / slices ~0.4 | gate_up concat: a 128-row tail or a pre-collective slice estimated at its cost (Rejected) |
| ~8.5 | fixed | LM head, cross-entropy chain, AdamW, embedding | LM head 3.04 + 2.76; CE stats 0.69 + 0.63, backward 0.34; broadcast 0.49; AdamW ~0.4; embedding 0.16 | forward stats in one pass (~1.3 ms) |
| ~2.5 | glue | remaining adds | 7 adds of `[1024, 1024]`-class shapes 0.25 | - |

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
- **Attribution**: chip 0, per 4 layers, cycle-time: SDPA forward 1.072 -> 0.915, backward Q 1.084 -> 0.862, backward KV 1.839 -> 1.154 + reduce 0.113; SDPA 3.995 -> 3.044 (-0.95, ~-9.5 ms per 40 layers per chip); other rows unchanged within 1 %. Below the core-balance arithmetic because the kernels re-read K/V or Q/dO per Q row (`D`, "Iteration 11").
- **Verification**: standalone vs the previous kernels at six shapes (`O/iter11_sdpa_check.log` vs `O/iter11_baseline.log`): forward and dQ bit-identical, dK / dV PCC 0.999999, max |diff| one bf16 ulp (fp32 partials summed in a different order); harness 35 passed (new `test_sdpa_causal_gqa_forward_and_gradients` at three shapes vs torch fp64); TP gradient check PASS. Bug found on the way: the KV kernel's 2-arg `pack_reconfig_data_format(cb_grad_key, cb_attention_weights)` skips the reprogram when both CBs are fp32 and left the packer in its start-up bf16 state; unconditional reconfig under `HEAD_SPLIT`.
- **Evidence**: `O/iter11_baseline.{sh,log}`, `O/iter11_build_check.{sh,log}`, `O/iter11_sdpa_check.{py,log}`, `O/iter11.{sh,log}`, `O/iter11_harness.log`, `O/iter11_tp_grad_check_tp4.log`, `O/iter11_tp_{mfu_tp4,analysis,ranked_ops}.*`, `O/iter11_tp_mfu_tp4_tt-smi.txt`.

## Iteration 12 (2026-10-30): low-rank update in one pass

- **Change**: glue + math, rung write new: `metal/ops/lowrank_update/` (`out = acc + s * a @ b`, one K tile: the product per output tile in fp32 DEST, the acc tile added there, one bf16 rounding; items = output tiles column-major, the a column resident in L1, 110 cores) routed from `modules/lora.py::lowrank_update` for `base + s * h @ B^T` (composite qkv / out adapters, forward) and `dx += dh @ A` (fused gate_up / w2 backward); replaces a 33-88-core matmul writing a bf16 intermediate plus an add per call.
- **Result**: TP=4 518.5 ms / 31.29 % vs 524.7 (-1.2 %) at 1.111-1.197 GHz effective (iteration 11's run 1.149-1.273), chips 49-73 C.
- **Attribution**: chip 0, per 4 layers, cycle-time (CSV diff, review 7): gone are the adapter-forward matmuls `[1024,32[16]] x [5120,32[16]]` 0.196 and `[1536,32[16]]` 0.075, half of the `[32[16]] x [32[16],5120]` row (0.567 -> 0.189; the 4 qkv composite dx matmuls stay) and the `[32[16],8192]` row 0.534, plus 16 adds (0.641 + 0.357); new `LowrankUpdate` 0.520 + 0.378 + 0.133; net -1.15 (chip 2: -1.18); ops 528 -> 512 per step; kernels excluding collectives 47.5 -> 46.1.
- **Verification**: standalone vs `ttnn.matmul` + `addalpha` at five shapes PCC 0.999998-0.999999, rel 1.5-2.0e-3 (one rounding instead of two), 124 -> 72 us (N=5120), 188 -> 105 (8192), 52 -> 36 (1536); harness 38 passed (new `test_lowrank_update_matches_torch` at three shapes incl. two images); TP gradient check PASS. Bug found on the way: the compute kernel skipped the pop of the column's w tile when the image changed (two-image case wrong, PCC 0.99985); fixed, covered by the two-image test.
- **Evidence**: `O/iter12_lowrank_check.{py,log}`, `O/iter12_build_check.{sh,log}`, `O/iter12_lora_tests.log`, `O/iter12.{sh,log}`, `O/iter12_harness.log`, `O/iter12_tp_grad_check_tp4.log`, `O/iter12_tp_{mfu_tp4,analysis,ranked_ops}.*`.

## Iteration 13 (2026-10-30): transpose and split in one pass

- **Change**: layout, rung write new: `metal/ops/transpose_split/` (`{input[:rows]^T, s * input[rows:rows+tail]^T}` for the fused LoRA backward's `[in + TAIL, T]` product: one tile transpose per output tile, the tail scaled in DEST, 110 cores) in place of two `ttnn.slice`, two `ttnn.transpose` and a `multiply` per fused layer (`modules/fused_lora.py`).
- **Result**: TP=4 516.4 ms / 31.42 % vs 518.5 (-0.4 %) at 1.093-1.210 GHz effective (iteration 12's run 1.111-1.197), chips 50-74 C.
- **Attribution**: chip 0, per 4 layers, cycle-time: `Slice [8256,1024]` 0.329 + `Transpose [8192,1024]` 0.331 + `Slice [5184,1024]` 0.206 + 4 of the 5 `Transpose [5120,1024]` calls 0.208 -> `TransposeSplit` 0.309 + 0.193; -0.57 ms per 4 layers (~-5.7 ms per 40 layers per chip); ops 512 -> 488 per step; kernels excluding collectives 46.1 -> 45.3.
- **Verification**: standalone vs slice + transpose (+ multiply) at four shapes bit-exact (`max |diff| 0`), 117 -> 54 us (gate_up route), 184 -> 85 (w2), 120 -> 54 with the TP scale; harness 40 passed (new `test_transpose_split_matches_numpy` at two shapes); TP gradient check PASS.
- **Evidence**: `O/iter13_transpose_split_check.{py,log}`, `O/iter13_build_check.{sh,log}`, `O/iter13_lora_tests.log`, `O/iter13.{sh,log}`, `O/iter13_harness.log`, `O/iter13_tp_grad_check_tp4.log`, `O/iter13_tp_{mfu_tp4,analysis,ranked_ops}.*`.

## Iteration 14 (2026-10-30): fused vocab-parallel cross-entropy backward

- **Change**: fixed, rung write new: `metal/ops/vp_cross_entropy_bw/` (per logits tile `exp(x - logZ) * scale * g` in fp32 DEST from the bf16 shard and the fp32 log-normalizer, the one-hot term patched in L1 by the writer at the shard's target column, mesh-aware per-device shard window as `subtract_at_target`; 256 items on 110 cores) called from the `vocab_parallel_cross_entropy_loss` backward (`ops/distributed/losses.cpp`) in place of subtract+exp (fp32 intermediate, 128 MB), multiply, `subtract_at_target` (32 cores) and the multiply by g.
- **Result**: TP=4 512.5 ms / 31.66 % vs 516.4 (-0.8 %) at 1.064-1.158 GHz effective (iteration 13's run 1.093-1.210), chips 49-74 C.
- **Attribution**: chip 0, per step (fixed rows), cycle-time: `BinaryNg [1024,32768]` 3 calls 1.624 + `SubtractAtTarget` 1.121 + `BinaryNg` 0.341 -> `VpCrossEntropyBw` 0.332 + the forward's remaining `BinaryNg` 0.623; -2.13 ms per step per chip; ops 488 -> 487; kernels excluding collectives 45.3 -> 43.0 per 4-layer step.
- **Verification**: standalone vs numpy fp64 at three shapes PCC 1.000000, rel 3.7-3.9e-3 (one bf16 rounding), 398 us for the `[1024, 32768]` shard; harness 42 passed (new `test_vp_cross_entropy_bw_matches_numpy` at two shapes); TP gradient check PASS; loss match rerun on this path: 8-layer host loss max |diff| 0.0294 vs torch fp64 (tol 0.10, mean +0.0059) PASS, TP=4 real weights 10 steps 0.4414 -> 0.1406 -> 0.4512 (the baseline's trajectory), 526.1 ms warm. Bug found on the way: the first version multiplied by the padded `[1, 1, 1, 1]` gradient tile (zero outside element (0, 0)); the loss backward now repeats g into a full tile.
- **Evidence**: `O/iter14_vpce_check.{py,log}`, `O/iter14_build_check.{sh,log}`, `O/iter14_loss_tests.log`, `O/iter14.{sh,log}`, `O/iter14_harness.log`, `O/iter14_tp_grad_check_tp4.log`, `O/iter14_tp_{mfu_tp4,analysis,ranked_ops}.*`, `O/iter14_train_8layers_hostloss.log`, `O/iter14_loss_comparison.txt`, `O/iter14_full_tp4_real.log`.

## Iteration 15 (2026-10-30): SDPA forward data flow (kernel boost)

- **Change**: math, kernel rewrite of the forward's data flow (`CHAIN` mode in `metal/ops/sdpa_fw/`): the cores are split per (batch, K/V group) stream (2 x 55 at the TP shard), the stream's 128 rows LPT-scheduled over them, and the K/V chunks are read from DRAM once per chain of 14 cores by its first core and forwarded core to core (variable_matmul's in0 protocol: ready counter, data write to the same CB address on the next core, valid flag), every core consuming the whole stream per round and dropping the chunks past its row's diagonal. DRAM traffic per call 67 MB -> ~12 MB. Checklist: `optimization/work_log.md`, "Iteration 15".
- **Result**: TP=4 506.0 ms / 32.06 % vs 512.5 (-1.3 %) at 1.106-1.200 GHz effective (iteration 14's run 1.064-1.158), chips 49-74 C.
- **Attribution**: chip 0, per 4 layers, cycle-time: `SDPAForward` 0.911 -> 0.559 (228 -> 140 us per call, -0.35 ms, ~-3.5 ms per 40 layers per chip); the other SDPA rows unchanged (bwd Q 0.862, bwd KV 1.151); standalone 238 -> 150 us wall at the TP shard, 882 -> 422 at the single-chip 32/8-head shape, 827 -> 445 at S=2048. Expected ~120 us (compute-bound estimate); landed at 140: the chain fill latency (14 hops) and the per-chunk handshakes are the rest.
- **Verification**: standalone vs the pre-iteration-11 kernel outputs at six shapes: forward bit-identical (`max |diff| 0`), dQ bit-identical, dK / dV one bf16 ulp (as iteration 11); harness 42 passed; TP gradient check PASS.
- **Evidence**: `O/iter15_build_check.{sh,log}`, `O/iter15_sdpa_check.log`, `O/iter15_attention_tests.log`, `O/iter15.{sh,log}`, `O/iter15_harness.log`, `O/iter15_tp_grad_check_tp4.log`, `O/iter15_tp_{mfu_tp4,analysis,ranked_ops}.*`.

## Iteration 16 (2026-10-30): SDPA backward data flow

- **Change**: math, the forward's chain data flow (iteration 15) in the Q backward (K/V stream per (batch, K/V group), chunks of 4 rows) and the head-split KV backward (Q / dO / lse / u stream per (batch, query head), chunks of 4 query rows); `metal/ops/sdpa_bw/`.
- **Result**: TP=4 508.4 ms / 31.91 % vs 506.0 (+0.5 % wall, inside the run-to-run noise: best 502.2 vs 502.4) at 1.102-1.236 GHz effective; chips 50-74 C.
- **Attribution**: chip 0, per 4 layers, cycle-time: `SDPABackwardQ` 0.862 -> 0.783, `SDPABackwardKV` 1.151 -> 1.097; -0.13 ms per 4 layers (~-1.3 ms per 40 layers per chip), far below the forward's gain: the backward kernels are compute-bound at ~5 us per (row, tile) unit (12 / 16 matmul tiles, the exp, the dS chain and two DEST transposes), the removed re-reads were mostly hidden (`optimization/work_log.md`, "Iteration 16").
- **Verification**: standalone vs the pre-iteration-11 outputs at six shapes unchanged (dQ bit-identical, dK / dV one bf16 ulp); harness 42 passed; TP gradient check PASS.
- **Evidence**: `O/iter16_build_check.{sh,log}`, `O/iter16_sdpa_check.log`, `O/iter16_attention_tests.log`, `O/iter16.{sh,log}`, `O/iter16_harness.log`, `O/iter16_tp_grad_check_tp4.log`, `O/iter16_tp_{mfu_tp4,analysis,ranked_ops}.*`.

## Iteration 17 (2026-10-30): single-phase RMSNorm backward (kernel boost)

- **Change**: math, kernel rewrite of the data flow (`metal/ops/rmsnorm_bw_rows/`, dispatched from `rmsnorm_bw_allcore` when rows x slices fit the grid and gamma is frozen): one (tile-row, 54-tile slice) item per core with the input, gamma and dL_dout slices resident in L1, the row's three per-slice dot products exchanged over the NoC (each core writes its fp32 partial into the others' CB slot and bumps a semaphore), then the gradient pass from the resident tiles with the residual gradient streamed. Every operand read once (50 MB per call) instead of 60 MB in two launches with a DRAM round trip of the partials. Checklist: `optimization/work_log.md`, "Iteration 17".
- **Result**: TP=4 505.6 ms / 32.09 % vs 508.4 / 506.0 (iterations 16 / 15) at 1.121-1.178 GHz effective; chips 50-74 C.
- **Attribution**: chip 0, per 4 layers, cycle-time: `RMSNormBackward` 1.093 + `RMSNormBackwardPartial` 0.740 -> `RMSNormBackwardRows` 1.320 (229 -> 165 us per call; -0.51 ms, ~-5 ms per 40 layers per chip); ops 487 -> 479 per step; other norm rows unchanged. Landed at 165 us, above the expected 130-150 us band: the exchange and the resident-slice load do not overlap with DRAM reads as the two-phase streaming did.
- **Verification**: standalone PCC 0.999999 / rel 1.7e-3 vs torch fp64 at three shapes (the two-phase op's accuracy; `O/iter17_rmsnorm_check.log`); norm harness tests 8 passed (the 2-batch case exercises the two-phase fallback); harness 42 passed; TP gradient check PASS.
- **Evidence**: `O/iter17_build_check.{sh,log}`, `O/iter17_rmsnorm_check.{py,log}`, `O/iter17_rmsnorm_tests.log`, `O/iter17.{sh,log}`, `O/iter17_harness.log`, `O/iter17_tp_grad_check_tp4.log`, `O/iter17_tp_{mfu_tp4,analysis,ranked_ops}.*`.

## Iteration 18 (2026-10-30): LoRA gradient products (kernel boost, partial)

- **Change**: math, the six rank-16 products per fused layer (`dB = s g^T h` -> `[N, r]`, `dA = dh^T x` -> `[r, N]`). New op `metal/ops/skinny_contract/` (small operand resident, column groups over all cores, scale folded in); it beats ttnn's `[N, r]` orientation but not ttnn's `[r, N]` orientation plus a transpose of the small output, so the router (`modules/lora.py::skinny_contract`) sends dA and the wide dB (gate_up / out / w2) there and only the narrow qkv dB to the op. Checklist: `optimization/work_log.md`, "Iteration 18".
- **Result**: run A (op for the wide dB) 498.0 ms / 32.58 %; run B (final router) 503.9 ms / 32.19 % (best 495.1); vs 505.6 / 32.09 % (iteration 17). The 5.9 ms between A and B is inside run-to-run noise (per-step spread 484-511 ms inside one run; the MFU runs carry no per-chip clock); the router follows the cycle-time attribution (review round 8).
- **Attribution**: chip 0, per 4 layers, cycle-time. gate_up dB 0.575 (17) -> 0.472 (A, op) -> 0.369 (B, ttnn `[r, N]` + transpose; 92 us per call); N = 5120 dB 0.389 -> 0.387 -> 0.234; the `[16, N]` transposes + pads 0.09 (B). All six product rows plus the transposes: 1.63 (17) -> 1.47 (A) -> 1.32 (B) (`tp_rank_ops.py --top 70` on the three CSVs). Chip 0 kernels without collectives 42.2 (17) / 41.6 (A) / 41.5 (B) ms per profiled step (`O/iter17_tp_analysis.txt`, `O/iter18_tp_analysis.txt`, `O/iter18b_tp_analysis.txt`: kernels minus collectives 6.0 / 4.4 / 4.4; the earlier "45.9 vs 46.0 vs 48.2" counted collective waiting). The dA N = 5120 row fell 0.289 -> 0.234 in run A on an unchanged ttnn call because the router dropped its `core_grid=` argument (ttnn's default grid is faster for that shape). Standalone per call: gate_up dB 156 (ttnn `[N, r]`) / 123 (op) / 110 us (ttnn `[r, N]` + transpose); N = 5120 dB 55 / 53 / 41; N = 1536 dB 28 / 25 / 39; dA N = 5120 32 (ttnn) / 53 (op). The op landed below half its expected gain (190-265 GB/s vs ttnn's 312 in the fast orientation: strided 8 KB reads per K step, 32 `matmul_tiles` per output tile for 16 useful rows); the fast orientation is the better fix for these shapes.
- **Verification**: standalone PCC 0.999999 / 1.000000 at seven shapes vs ttnn (`O/iter18_build_check.log`), orientation variants rel 1.2e-3 (bf16 output rounding; `O/iter18_orientation_check.log`); harness 45 passed (3 new `skinny_contract` cases); TP gradient check PASS (runs A and B).
- **Evidence**: `O/iter18_build_check.{sh,log}`, `O/iter18b_build_check.log`, `O/iter18_contract_check.py`, `O/iter18_orientation_check.{py,log}`, `O/iter18.sh` (`TAG=iter18b` for run B), `O/iter18_chain.log`, `O/iter18b_chain.log`, `O/iter18{,b}_harness.log`, `O/iter18{,b}_tp_grad_check_tp4.log`, `O/iter18{,b}_tp_{mfu_tp4,analysis,ranked_ops}.*`.

## Iteration 19 (2026-10-30): single-phase fused norm forward (kernel boost) + w2 forward matmul config

- **Change**: (a) math, kernel rewrite: `metal/ops/add_rmsnorm_fw_rows/`, one (tile-row, 54-tile slice) item per core with residual, delta, gamma and the sum resident in L1, the row's sums of squares exchanged over the NoC, dispatched from `add_rmsnorm_fw_allcore` when rows x slices fit the grid with >= 2 slices. (b) math config: a swept 2D block-multicast program config for the w2 forward matmul `[1024, 8256] x [8256, 5120]` (`frozen_linear.py::PROGRAM_CONFIG_2D`, grid 11x8, `in0_block_w` 6, subblock 4x1); the other big shapes keep ttnn's choice (sweeps found nothing better). Checklist: `optimization/work_log.md`, "Iteration 19".
- **Result**: TP=4 495.4 ms / 32.75 % (best 487.9) vs 503.9 / 32.19 % (iteration 18 B); real weights warm 509.1 ms (515.5 before the config applied, 526.1 after iteration 14).
- **Attribution**: chip 0, per 4 layers, cycle-time: w2 forward matmul 3.693 -> 3.249 ms (-12 %, 0.92 -> 0.81 ms per call, 71 % of peak); norm forward 0.565 + 0.558 -> 1.010 (`AddRMSNormForwardRows`, 140 -> 126 us per call); chip 0 kernels without collectives 41.5 -> 41.1 ms per profiled step (`O/iter19_tp_analysis.txt`, collectives 4.7). Standalone: norm forward 166 -> 138 us per call (bit-exact; the streaming version, `O/iter19_norm_check.log`; a first resident-slice version measured 158 and was slower in the step); w2 forward 1174 -> 1058 us with the committed 11x8 grid (interleaved A/B medians, -10 %; the sweep's 11x10 at 1043 was within noise of it and 11x8 was taken, no stronger reason recorded).
- **Verification**: norm op bit-exact vs the two-phase op at the model shape (`O/iter19_norm_check.log`); matmul config within one bf16 ulp of ttnn's default (accumulation order), so the loss match was rerun: PASS (8-layer host loss vs torch max |diff| 0.0298, tol 0.1, `O/iter19_loss_comparison.txt`; review round 9: that single-chip run has K = 32832 and never hits the TP-local config key `(1024, 8256, 5120)`, so the config is covered only by the TP=4 real-weight trajectory below, which has no torch reference: a hard-check gap; TP=4 real-weight trajectory 0.4434 -> 0.3281 -> ... -> 0.4473 vs 0.4414 -> ... -> 0.4492 before the config, `O/iter19_full_tp4_real.log`); harness 47 passed (2 new `skinny_contract` cases); TP gradient check PASS.
- **Evidence**: `O/iter19_build_check{,2,3,4}.log`, `O/iter19_norm_check.{py,log}`, `O/iter19_norm_tests.log`, `O/iter19_w2_ktail_bench.{py,log}`, `O/iter19_w2_sweep2d.{py,log}`, `O/iter19_matmul_sweep2d.py`, `O/iter19_sweep2d_*.log`, `O/iter19_baseline_check.{py,log}`, `O/iter19_ab_check.{py,log}`, `O/iter19.sh`, `O/iter19_chain.log`, `O/iter19_harness.log`, `O/iter19_tp_grad_check_tp4.log`, `O/iter19_tp_{mfu_tp4,analysis,ranked_ops}.*`, `O/iter19_loss_comparison.txt`, `O/iter19_full_tp4_real.log`.

## Iteration 21 (2026-10-31, rejected): SDPA forward from the `agobeljicTT/sdpa_scheduler` stack

- **Change tried**: math, kernel: the forward of PRs #61884 / #62646 (multi-row Q chunks with `matmul_block` score rectangles, K/V steps of several tiles, GQA head stacking, longest-first placement, heavy causal items cut into K/V step ranges and merged in-kernel) checked out over the chain-mode forward (same base commit, drop-in API) and measured standalone at the shard shape before any full-model run. Checklist: `optimization/work_log.md`, "Iteration 21".
- **Result**: shard shape (1, 8, 2, 1024, 128): chain-mode 149-157 us (python wall); the port's default pick (four-row items: 64 items on 110 cores) 186, best of 21 configs 166 (two-row items with cuts `q64 k128 h2 p4`, or `q128 k128 h1 p0`), main's chunking inside the port 240. At (1, 8, 2, 2048, 128) the port wins (401 vs 444). Outputs agree with the chain kernel to one bf16 ulp; attention harness tests 4 passed.
- **Why**: at S=1024 with 8 heads the item count, not the per-row overhead, sets the makespan: 64 four-row items leave 46 cores idle and the cuts bring the time back only to 166; the chain kernel's 256 one-row items LPT-balance to within a row and read K/V once per chain. The two designs share the placement (iteration 11's LPT lists are the stack's longest-first) and differ in what they amortise: the stack amortises the softmax bookkeeping over rows and heads, the chain kernel the K/V reads over cores. Per score tile both sit at 2.7-3.9 us against a ~1.5 us SFPU + FPU estimate (exp, max, sum, 8 matmul tiles, 3 pack / unpack passes).
- **Reverted**: forward restored from HEAD and rebuilt; bit-exact against the iteration-20 outputs (`O/iter21_sdpa_reverted.log`).
- **Evidence**: `O/iter21_build_check.{sh,log}`, `O/iter21_sdpa_baseline.log` (chain kernels, same session), `O/iter21_sdpa_check.log`, `O/iter21_sdpa_benchmark{,2}.log` (the stack's `sdpa_fw_benchmark` with a `mistral24b_tp4_h8kv2d128_s1k` scenario, not kept in the tree), `O/iter21_attention_tests.log`, `O/iter21_sdpa_reverted.log`.
- **Carry-over**: a two-row chain kernel (score rectangles over the 4-row K/V stream chunks) is the forward's remaining lever (~10 %, ~0.5 ms/step). Constraint change (user, 2026-10-31): bf16 intermediates are allowed wherever the PCC holds, and the loop runs on toward 40 % with a kernel boost per iteration; the backward kernels (18.7 ms/step at ~5 us per (row, K tile) unit against a ~2 us estimate) are next.

## Iteration 22 (2026-11-01): SDPA backward data path (kernel boost)

- **Change**: math, kernels we own, two parts. (1) `sdpa_bw_q`'s compute kernel works on K/V blocks of four rows (the chain chunk): P, dP and dS two columns per fp32 DEST pass (two score tiles plus the mask / lse tile; two dP plus two P tiles), dQ accumulated over the block in DEST and added to the L1 accumulator once per block instead of once per column (`sdpa_bw_q_compute_kernel.cpp`, `apply_mask_on_reg_at`, `apply_softmax_statistics_on_dst_n`; the factory sizes the P / dP / dS CBs to a block). (2) The chain injectors of all three SDPA kernels issue a chunk's DRAM tile reads together behind one `noc_async_read_barrier` instead of one barrier per row (Q and KV readers, forward reader). Checklist: `optimization/work_log.md`, "Iteration 22".
- **Attribution** (chip 0, per 4 layers, cycle-time): `SDPABackwardKV` 1.093 -> 0.896, `SDPABackwardQ` 0.784 -> 0.574, `SDPAForward` 0.558 -> 0.545: -0.42 ms per 4 layers, ~-4.2 ms per step per chip. Standalone at the shard shape: backward 532 -> 427 us (blocked Q kernel alone 517, the injector barriers the rest), forward 149 -> 145.
- **Result**: TP=4 495.4 -> 494.5 ms (best 487.9 -> 486.9), 32.81 % MFU. The -0.9 ms wall is inside the run-to-run spread (steps 480-506 vs 485-506 ms; tt-smi read 1350 MHz on all chips in both MFU runs at equal temperatures), so the landed gain is the attribution's -4.2 ms/step cycle-time, not the wall difference; the 1.03-1.09 vs 1.08-1.16 GHz effective clocks are the 4-layer profile runs' (`O/iter22_tp_analysis.txt`, `O/iter19_tp_analysis.txt`), where the collectives row also moved (6.1 vs 4.7 ms per 4 layers on chip 0).
- **What the time is**: the attribution experiments (`O/iter22_attrib{,2,3,4}.log`, `O/iter22_tracy_modes.log`) show the chain-mode SDPA kernels are bound by the K/V stream, not the math: a skeleton Q compute kernel with every matmul, exp and pack removed costs only ~10 us less than the real one (517 -> 506 us of the backward), each single stage 6-11 us; the stream mode changes it (chain 517; the plain per-row list mode read 548 but ran the blocked kernel on a CB layout it straddles, review round 9 P1; with one-row blocks outside the chain it is 483 us and bit-exact with the pre-iteration-22 kernel, `O/review9_list_mode_check.log`) and the injector's per-row read barriers alone were 45 us (472 without the reads). The chain hop (ready handshake, 64 KB unicast, flush, relay; ~5 us per chunk, 24 chunks per call) now sets the Q and forward kernels' time; the next lever is overlapping a core's receive and forward (forward from the writer RISC on the second NoC).
- **Verification**: harness 47 passed; TP gradient check PASS; standalone vs the iteration-20 kernels: forward, dK, dV bit-exact, dQ within one bf16 ulp (the block accumulation order; 139 of 1 M elements at the shard shape); 8-layer host-loss match PASS (max |diff| 0.0292, tol 0.1, `O/iter22_loss_comparison.txt`); TP=4 real weights warm 504.9 ms (509.1 after iteration 19).
- **Rejected on the way**: a multicast K/V stream (one or two injectors per stream, NoC rectangles, ready / valid semaphores, `O/iter22c_*`, `O/iter22e_*`): correct where it ran but no faster than the chain (526 / 514 vs 517 us), and it misbehaved on some shapes (two injectors: 1.5-4.7x slower once the injector reads were fast; one injector: hung on the single-stream (1, 4, 1, 1024, 128) case); the chip needed `tt-smi -r` twice. Removed from the tree; the `TTML_SDPA_BW_Q_STREAM=list` knob for attribution stays.
- **Evidence**: `O/iter22_build_check.{sh,log}` (first build: dQ garbage from the matmul operands' source-register formats, fixed in `O/iter22b_build_check.log`), `O/iter22{c,d,e,g,h,i}_*` (multicast, hang, single-barrier injectors Q / KV, final tree), `O/iter22d_chain_sanity.log`, `O/iter22_attrib*.log`, `O/iter22_tracy_modes.log`, `O/iter22.{sh,log}`, `O/iter22_harness.log`, `O/iter22_tp_grad_check_tp4.log`, `O/iter22_tp_{mfu_tp4,analysis,ranked_ops}.*`, `O/iter22_tp_mfu_tp4_tt-smi.txt`.

## Iteration 23 (2026-11-01): SDPA chain forwarding on the writer RISC (kernel boost)

- **Change**: math (owned kernels): the chain forwarding of a chunk moved from the reader RISC to the writer RISC on NOC_1 (received / forwarded counters between the two RISCs); chains reversed to NOC_1's routing direction (injector = highest core). Checklist: `optimization/work_log.md`, "Iteration 23".
- **Result**: TP=4 494.5 -> 480.3 ms (best 473.1), 33.78 % MFU; the wall drop beyond the attribution is the clock mix (chips 0-1 at 1.16 GHz effective in the profile, 1.03-1.09 before).
- **Attribution** (chip 0, per 4 layers, cycle-time): `SDPABackwardKV` 0.896 -> 0.824, `SDPABackwardQ` 0.574 -> 0.523, `SDPAForward` 0.545 -> 0.512 (-0.16 per 4 layers, ~-1.6 ms/step); standalone forward 145 -> 137 us, backward 427 -> 393 (the upward-forwarding first build 592).
- **Verification**: bit-exact against iteration 22 on six standalone cases; harness 47; TP gradient check PASS; 8-layer loss match PASS (0.0292, unchanged numerics); real weights warm 502.2 ms.
- **Evidence**: `O/iter23{,b,c,d}_build_check.log`, `O/iter23.{sh,log}`, `O/iter23_harness.log`, `O/iter23_tp_grad_check_tp4.log`, `O/iter23_tp_{mfu_tp4,analysis,ranked_ops}.*`, `O/iter23_loss_comparison.txt`, `O/iter23_full_tp4_real.log`; full text `D`, "Iteration 23".

## Iteration 24 (2026-11-01): the w2 K-concat removed, producers write the wide LoRA input (layout)

- **Change**: layout (owned kernels): the w2 `concat(x, pad(h))` is gone: the SwiGLU and `skinny_matmul` write the wide `[T, in + 64]` input (h and the zero tile in the tail), the backward ops take the wide gradient. gate_up keeps its concat (Rejected table). Checklist: `optimization/work_log.md`, "Iteration 24".
- **Result**: TP=4 480.3 -> 481.2 ms (best 470.6), 33.72 % MFU (chips 2-3 at 1.03 GHz effective in the profile, 1.05-1.07 before: within the run spread).
- **Attribution** (chip 0, per 4 layers, cycle-time): w2 `Concat` 0.321 and its `Pad` gone (483 -> 475 ops), `SwigluPackedBw` 1.096 -> 0.745 (mechanism unexplained, Ledger), kernels without collectives 40.7 -> 40.0 ms per profiled step (~-7 ms/step); standalone every wide piece bit-exact with the concat path, pad + concat 101 us per call removed.
- **Verification**: harness 48 (new `test_mlp_fused_lora_wide_w2` at the shard width, PCC 0.99999, rel 4-5e-3: the wide path's torch reference); TP gradient check PASS (the first chain's failed in the wrapper's backward, fixed); 8-layer loss match PASS (0.0292; the single-chip run never fuses w2, Ledger); real weights warm 497.9 ms.
- **Evidence**: `O/iter24_wide_check.{py,log}`, `O/iter24_build_check.{sh,log}`, `O/iter24.log` + `O/iter24_*` (failed first chain), `O/iter24b.log`, `O/iter24b_harness.log`, `O/iter24b_tp_grad_check_tp4.log`, `O/iter24b_tp_{mfu_tp4,analysis,ranked_ops}.*`, `O/iter24b_loss_comparison.txt`, `O/iter24b_full_tp4_real.log`; full text `D`, "Iteration 24".

## Iteration 25 (2026-11-01): the low-rank update fused into the transpose (kernel boost)

- **Change**: math (owned kernels): `transpose_split(w = A)` computes `dx = dcat^T[:in] + s_dh @ A` in one DEST pass per dx tile (the s_dh tile once per row block into a resident CB); the second pass over dx (`lowrank_update`) is gone from the fused LoRA backward. Checklist: `optimization/work_log.md`, "Iteration 25".
- **Result**: TP=4 481.2 -> 476.7 ms (best 468.2), 34.03 % MFU, at a lower clock mix than iteration 24 (1.03-1.13 vs 1.03-1.20 GHz effective).
- **Attribution** (chip 0, per 4 layers, cycle-time): the fused layers' `LowrankUpdate` rows (0.257 + 0.370) gone, `TransposeSplit` 0.301 -> 0.399 and 0.185 -> 0.253 (-0.46 per 4 layers, ~-4.6 ms/step); kernels without collectives 40.0 -> 39.5; standalone bit-exact, gate_up 180 -> 74 us, w2 shard 277 -> 116.
- **Verification**: fused-LoRA harness subset 8; harness 48; TP gradient check PASS; 8-layer loss match PASS (0.0292, unchanged numerics); real weights warm 492.2 ms.
- **Evidence**: `O/iter25{,b,c}_build_check.log` (the first two measured the plain transpose: patch not applied), `O/iter25_fused_check.py`, `O/iter25c_fused_check.log`, `O/iter25.{sh,log}`, `O/iter25_harness.log`, `O/iter25_tp_grad_check_tp4.log`, `O/iter25_tp_{mfu_tp4,analysis,ranked_ops}.*`, `O/iter25_loss_comparison.txt`, `O/iter25_full_tp4_real.log`; full text `D`, "Iteration 25".

## Iteration 26 (2026-11-01): SDPA chain credits, 8-row forward chunks, forward chain length (kernel boost)

- **Change**: math (owned kernels), three parts in one commit, each measured on its own: credit-based chain hops (three chunks in flight, counting semaphores); 8-row K/V chunks in the forward (QK^T / exp in DEST-sized passes, two-chunk score CB); forward chains of 8 cores (sweep 4-14). Attribution first: math 50 / 53 / 91 us of 136 / 132 / 198 (fw / Q / KV). Checklist: `optimization/work_log.md`, "Iteration 26".
- **Result**: TP=4 476.7 -> 472.5 ms (best 465.9), 34.33 % MFU (profile at 1.16-1.18 GHz effective on all chips, 1.03-1.13 in iteration 25); kernels without collectives 39.5 ms per profiled step, flat.
- **Attribution** (chip 0, per 4 layers, cycle-time): `SDPAForward` 0.511 -> 0.422, `SDPABackwardQ` 0.523 -> 0.508, `SDPABackwardKV` 0.824 -> 0.807 (-0.12 per 4 layers, ~-1.2 ms/step, at the 2-step profile's noise floor); standalone forward 136 -> 132 (credits, bit-exact) -> 131 (8-row chunks, one bf16 ulp) -> 115 (chain 8), backward 393 -> 387.
- **Verification**: harness 48; TP gradient check PASS; 8-layer loss match PASS (0.0298 vs 0.0292: the forward's rounding order changed); real weights warm 493.2 ms. Rejected on the way (Rejected table): bf16 DEST on the unchanged algorithm, position-aware LPT loads.
- **Evidence**: `O/iter26*` (attribution: `iter26_attrib*`, per-core `iter26_percore_spans.*`, `iter26_lpt_fit.py`, `iter26_tracy_ops_perf_results.csv`; patches `iter26{,b,c}_patch.py`; sweeps `iter26c_sweep*`; checks `iter26{,b,c}_build_check.log`; chain `iter26.log`, `iter26_tp_*`, `iter26_loss_comparison.txt`, `iter26_full_tp4_real.log`); full text `D`, "Iteration 26".

## Iteration 27 (2026-11-01): per-round chain stream bounds (kernel boost)

- **Change**: math (owned kernels): per chain and round the stream carries only the chunks the round's rows need (forward / Q: up to the heaviest row's diagonal; KV: from the lightest item's K row), passed after the row list; readers, writers and compute loop over it. Backward chains re-swept to 7. Checklist: `optimization/work_log.md`, "Iteration 27".
- **Result**: TP=4 472.5 -> 468.1 ms (best 465.4), 34.66 % MFU; the profile's clock mix 0.98-1.21 GHz (chip 3 at 0.98, Ledger), kernels without collectives 39.5 ms per profiled step, flat.
- **Attribution** (chip 0, per 4 layers, cycle-time): `SDPABackwardKV` 0.807 -> 0.722, `SDPABackwardQ` 0.508 -> 0.435, `SDPAForward` 0.422 -> 0.397 (-0.18 per 4 layers, ~-1.8 ms/step); standalone bit-exact, shard backward 387 -> 345 us, forward 114 -> 110 (the forward's stream was already hidden behind its compute after iteration 26: `nofwd,noread` 121 vs 132; the backward kernels were not).
- **Verification**: harness 48; TP gradient check PASS; 8-layer loss match PASS (0.0298, identical to 26); real weights warm 487.5 ms.
- **Evidence**: `O/iter27_patch.py`, `O/iter27{,b}_build_check.log`, `O/iter27_sweep.{sh,log}` + `O/iter27_sweep_*.log`, `O/iter27.{sh,log}`, `O/iter27_harness.log`, `O/iter27_tp_grad_check_tp4.log`, `O/iter27_tp_{mfu_tp4,analysis,ranked_ops}.*`, `O/iter27_loss_comparison.txt`, `O/iter27_full_tp4_real.log`; full text `D`, "Iteration 27".

## Iteration 28 (2026-11-01): KV backward in query-row blocks (kernel boost)

- **Change**: math (owned kernels): the KV backward processes the chain chunk's query rows as a block: scores two rows per DEST pass (a masked row alone), dP and dS in one pass per row (dP never leaves DEST: the u subtraction and the P multiply on the SFPU), dV and dK contracted over the block in DEST with one L1 accumulate each: 8 DEST passes and 20 packs per 4 rows instead of 20 and 48. Checklist: `optimization/work_log.md`, "Iteration 28".
- **Result**: TP=4 468.1 -> 468.4 ms (best 465.4 -> 463.5), 34.63 % MFU; the first build measured 464.7 / 34.92 % at the same clock mix (1.17-1.24 GHz), so the wall is within the run spread and the landed gain is the attribution; kernels without collectives 39.5 -> 39.2 ms per profiled step.
- **Attribution** (chip 0, per 4 layers, cycle-time): `SDPABackwardKV` 0.722 -> 0.630 (-0.09 per 4 layers, ~-0.9 ms/step); standalone shard backward 345 -> 323 us, 32 heads 1210 -> 1118, S=2048 1046 -> 962; dP no longer rounds through a source register, so dK moves by one bf16 ulp (21 % of elements), dV by a few (PCC >= 0.999999).
- **Verification**: harness 48; TP gradient check PASS; 8-layer loss match PASS (0.0293); real weights warm 486.9 (480.1) ms. The first build hid a mask-register bug under a DEST priming cycle (Ledger); the final build passes the causal six shapes, the explicit-mask (non-chain) cases and the MHA cases, chain rerun `O/iter28b*`.
- **Evidence**: `O/iter28_patch.py`, `O/iter28{,b,c}_build_check.log`, `O/iter28_diag.py` (scratch), `O/iter28_noncausal_check.{py,log}`, `O/iter28c_noncausal_check.log`, `O/iter28_swiglu_bw_ab.{py,log}`, `O/iter28_bf16dest_patch.py`, `O/iter28_bf16dest_check.log`, `O/iter28{,b}.{sh,log}`, `O/iter28b_harness.log`, `O/iter28b_tp_grad_check_tp4.log`, `O/iter28b_tp_{mfu_tp4,analysis,ranked_ops}.*`, `O/iter28b_loss_comparison.txt`, `O/iter28b_full_tp4_real.log`, `O/iter28{,c}_sdpa_baseline.log`.

## State after iteration 20 (2026-10-30 night): where the remaining time is

> Written after iteration 20 (2026-10-30 night), kept as the record of that night's conclusion. Review round 9: the "owned kernels at their exact-fp32 ceilings" statement held for the compute side only; iteration 22 (2026-11-01) found the SDPA kernels' data path (the K/V stream) worth another ~4 ms/step with no numerics change, and iteration 23 continues there. The ttnn matmul and chip-skew statements stand.

Chip 0, run B / iteration 19 profiles, cycle-time per 40-layer step (~410 ms of kernels; wall 495 ms at ~1.1 GHz):

| share | item | ceiling under the hard constraints | lever |
|---|---|---|---|
| 63 % | base matmuls (ttnn) at 63-85 % of HiFi4 peak | ttnn's program configs; 1D and 2D sweeps exhausted except the w2 forward (taken, iteration 19) | ttnn matmul kernel work (outside ttml), or fewer FLOPs (K-concat tail: +1.2 % FLOPs) |
| 10 % | collectives = waiting for the slowest chip (clock skew 1.03-1.16 GHz across chips) | transfer itself 4.0 ms per 4 layers at the library optimum | overlap with independent backward work (two command queues / async CCL), or a cooler / evenly clocked box |
| 6 % | SDPA fwd + bwd at ~10 % of FLOP peak | iteration 22 found the chain-mode kernels bound by the K/V stream (chain hop ~5 us per 64 KB chunk, 24 chunks per call), the math hidden under it; iteration 23 overlapped receive and forward (writer RISC / second NoC) | bf16 scores / P in DEST (the compute, ~2 ms/step); the `sdpa_scheduler` forward measured slower at this shape (iteration 21) |
| 6 % | RMSNorm fwd + bwd, exact fp32 SFPU per tile | 126 / 165 us per call vs 100 / 125 us bandwidth rooflines; FPU routes round fp32 (iteration 20) | none exact; -1 ms/step at most |
| 8 % | LoRA adapters | wide products at 85-88 % of bandwidth, narrow ones launch-bound; the dx update pass fused into the transpose (iteration 25) | fewer launches (fuse dA / dB pairs): ~1 ms/step |
| 4 % | layout (K-concats 0.56 ms per 4 layers at 433 GB/s, transposes) | the concats are at bandwidth; the w2 one removed in iteration 24 (the producers write the wide buffer), gate_up's stays (TP all-reduce tile split) | ~2 ms/step left in gate_up's concat, not worth a 128-row tail |
| 3 % | SwiGLU | bwd at 343 GB/s | - |

The kernel-boost loop (iterations 15-20) took the owned kernels' compute to their exact-fp32 ceilings (SDPA forward -39 %, RMSNorm backward -28 %, norm forward -10 %, LoRA products -19 %; TP=4 536.1 -> 495.4 ms). Corrected after iteration 27: the owned kernels' data paths were not at their ceilings: iterations 22-27 removed ~20 ms/step cycle-time from them (SDPA stream and scheduling 8.8, K-concat 7, fused transpose 4.6) at unchanged numerics but one rounding order; after 27 the owned kernels are ~95 ms/step (SDPA 17, norms 24, adapters 25, SwiGLU 13, layout 12) and the remaining rungs in the ranked table total ~10 ms. Reaching 40 % (405 ms from 468) still needs the ttnn matmul efficiency (~201 ms at 72-85 % of peak) and the collectives (40-90 ms incl. skew).

## Loop exit (2026-10-29, after iteration 3) and resumed state

- Exit condition met: 1339 ms cycle-time (2/4/8/16 fit 1345.8; 16 layers 563.6 ms device / 694.4 wall at 1100 MHz, 24.8 GiB) vs the 1620 ms target, 17 % under; wall 1661-1739 vs 2006-2101 ms at the run's clocks (ratio 0.83); stretch 1259 ms 6 % away. Overhead threshold 1 % = 13 ms.
- Full-model report at exit: 556.5 ms / 29.15 % random init; 574.8 ms real weights warm, loss 0.4453 -> 0.1406 -> 0.4531 vs the baseline's 0.4453 -> 0.1436 -> 0.4551.
- Final verification at exit: harness 14; TP gradient check PASS on the composite and the fused paths (first attempt un-interleaved, Ledger); loss match max |diff| 0.026 (composite 0.075); device bf16 loss quantised to 0.0625.
- Memory peak: 13.79 GiB after the forward at 8 layers (weights + LoRA state 11.61), 279 MiB retained per layer (composite 778); TP=4 40 layers ~11 GiB weights + ~2.3 GiB activations per chip (scaled, not probed).
- Resumed loop (iterations 4-18): TP=4 556.5 -> 547.5 (6) -> 545.9 (8) -> 545.7 (review 3) -> 537.4 (9) -> 536.1 (10) -> 524.7 (11) -> 518.5 (12) -> 516.4 (13) -> 512.5 (14) -> 506.0 (15) -> 508.4 (16) -> 505.6 (17) -> 498.0 (18 A) -> 503.9 (18 B) -> 495.4 ms / 32.75 % (19); single-chip projection 1339 -> ~1242 (attribution only, up to iteration 9). Remaining: see the ranked table (matmuls at ttnn's best config, collectives at the library optimum with overlap untried, adapters, norms, SDPA, layout).

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
| Fastest-overall gate_up 1D config (7.09 vs 7.35 ms) | ~10 (single chip only) | sweep `O/iter4_matmul_sweep1d_gateup_fwd.log` | max \|diff\| 3.1e-2 (one bf16 ulp); not taken without a loss rerun; deferred |
| dX^T route for w2 (`32768 -> 5120`) | 121.6 (w2 dX) | `O/iter2_matmul_bench3.log` | 3.72 -> 5.29 ms; the `[32768,1024]` transpose costs more than it saves |
| 8x11 grid for the LM head forward | 20.2 | `O/iter2_matmul_bench3.log` | 58.97 vs 21.48 ms, L1 overflow into a slow path |
| One-tile weight tail (K = 161 tiles) | 254-296 (gate_up fwd) | `iter3` tail measurement | +5 % (7.92 vs 7.51 / 7.59 ms for K = 5152 vs 5120 / 5184); two tiles used |
| `slice_write` / `paged_fill_cache` for the tail write | 37 (tail writes) | tried in iteration 3 | row-major only / page-table layout rejected the view; `fill_cache` then `write_tile_block` |
| Tile-aligned stacking of the LoRA partials | ~6 (stack / unstack glue) | `O/review3_stack_bench.{py,log}` | 0.30 vs 0.23 and 0.27 vs 0.26 ms, no gain; the 0.18 ms variant needs a 32-row-padded `dA` in the optimizer, deferred |
| LoRA B stored transposed | 22.8 -> ~11 (gate_up `dB`) | microbenchmark, iteration 2 (0.59 -> 0.39 ms) | queued behind the skinny-matmul item, not taken |
| Iteration 22: multicast K/V stream for the Q backward (one / two injectors per stream, NoC rectangles) | 7.8 (SDPA Q backward) | standalone, `O/iter22c_*`, `O/iter22e_*`, `O/iter22_attrib3.log` | 526 / 514 us vs 517 chain: no gain; two injectors 1.5-4.7x slower once their reads were fast, one injector hung on the single-stream shape (two chip resets). Removed; the chain injectors' single read barrier kept |
| Iteration 26: bf16 DEST in the SDPA forward, algorithm unchanged (4-tile passes) | 5 (SDPA forward) | one run, `O/iter26_attrib_fw_bf16dest.log` | 145 vs 132 us, PCC 0.9999: a straw-man variant (review round 10), superseded by the row below |
| Review round 10: bf16 DEST in the SDPA forward as intended (`fp32_dest_acc_en = false`, 8-tile passes: one QK^T and one exp pass per 8-row chunk) | 5 (SDPA forward) | six shapes vs the iteration-28 outputs, `O/iter28_bf16dest_patch.py`, `O/iter28_bf16dest_check.log` | slower at every shape (shard 119 vs 109 us, 32 heads 372 vs 336, S=2048 322 vs 288) at PCC 0.9999 (max \|diff\| 0.03-0.16, ~1 % relative): the bf16 DEST halves the stages but each pass costs more (fp32 CBs through SrcA) and the accumulation loses precision; under the 10-31 allowance (PCC holds) it is admissible and loses on time |
| Iteration 26: position-aware LPT for the chains (initial load per hop from the injector) | ~5 (SDPA, slowest core over the median) | sweep, `O/iter26c_sweep.log`, fit `O/iter26_lpt_fit.py` | forward 131 -> 137-142, backward 385 -> 397-420 (three hop loads, two chain lengths): the sink's excess is the stream fill, every core consumes the round's stream; the chain length is the lever |
| Iteration 24: gate_up K-concat removal | ~2 (gate_up `Concat` 0.21 per 4 layers) | estimated, not measured | the wide dx `[1024, 5184]` (162 tiles) is not divisible by the four TP ranks; a 128-row tail (+2.5 % K on the two largest matmuls) or a pre-collective slice (10 MB) estimated at the concat's cost |
| Iteration 21: SDPA forward from the `sdpa_scheduler` stack (multi-row items, head stacking, cut heavy items) | 5.6 (SDPA forward) | standalone at the shard shape, `O/iter21_*` | 186 us default / 166 best config vs 149-157 chain-mode: 64 four-row items on 110 cores; wins only at S=2048 (401 vs 444). Reverted |
| Iteration 20: norm kernels' per-tile math on the FPU (3-4 tiles per DEST acquire) | ~23 (norm fwd + bwd) | standalone, `O/iter20_*` | forward -3 %; backward -16 % only with `*_reuse_dest` ops that round the fp32 DEST tile through 19-bit source registers (rel error vs fp64 1.66e-3 -> 4.09e-3); the precision-preserving hybrid gains nothing (173 vs 172 us). Reverted |
| Host/dispatch | ~3 | cycle analysis | the plan's 58 ms does not exist (device busy 99.8 %) |
| 320 tiny per-adapter scale ops | 31 (stale-shape row) | iteration 3 fusion | absorbed for qkv / out / gate_up; w2's remain (~8) |
| Gamma re-read per work item in the fused norm forward | ~0.2 per 4 layers per chip | not attempted | multicast or resident copy; listed |
| `UnpackToDestFp32` read-back of fp32 partials (three all-core ops) | 0 | review round 6 note | relative 2^-11, below bf16 rounding; PCCs 0.999997-0.999999; left as is |
| TP collective parameters (links, workers per link, chunks per sync, buffers per channel) through env knobs in `ttnn_fixed/distributed/ttnn_ops.cpp` | 6.8 ms per 4 layers per chip on chip 0 (17 reduce-scatters at 281 us + 17 all-gathers at 120 us) | standalone sweep on the 1x4 mesh at the step's shapes (`O/iter12_ccl_sweep.{py,sh,log}`): reduce-scatter 122 us, all-gather 114 us with the defaults (2 links, 8 workers per link); 1 link 217 / 209, 1-2 workers 176 / 148-278, chunks per sync 1-4 slower for the all-gather, buffers per channel no effect | parameter tuning rejected: defaults fastest; the in-step excess over 122 / 114 us is chip skew; overlap with compute untried |

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
- Iteration 22 multicast experiment (review round 9 correction): the two-injector run with single-barrier reads did not deadlock, it completed all six cases 1.5-4.7x slower (`O/iter22d_sdpa_check.log`: 534 / 572 / 551 us backward) and the process was killed before it exited; the one-injector run hung on the single-stream (1, 4, 1, 1024, 128) case (`O/iter22e_sdpa_check.log`, rc 124). `tt-smi -r` was run twice (no output kept; the second time the chain-mode sanity run `O/iter22d_chain_sanity.log`, timing-only and concurrent with the attention tests, showed the chip healthy). `O/iter22_attention_tests.log` and `O/iter22_sdpa_check.log` are the failed first build (dQ inf), labelled by `O/iter22_build_check.log`; the chain-mode sanity run after the reset separated a wedged chip from a kernel deadlock. The stream-mode knob's first attribution run (`O/iter22_attrib2.log`, "no-chain") ran against a stale binary (a relative build-log path after `cd` again, build rc=1 unnoticed in a background chain): its chain / no-chain rows are the same binary and void; `O/iter22_attrib3.log` is the valid comparison.
- Review round 9 P1 (blocked Q kernel, non-chain modes): `kv_block = 1` outside the chain (commit after iteration 23); the list mode then reproduces the old kernel's dQ bit for bit on all six shapes (`O/review9_list_mode_check.log`), the chain mode is unchanged (`O/review9_chain_mode_check.log`), attention tests pass. Only `TTML_SDPA_BW_Q_STREAM=list`, non-causal calls and > 55 streams reach those modes; the model never did.
- Iteration 22 wall vs attribution: -0.9 ms wall for -4.2 ms/step of SDPA cycle-time; the wall difference is within the per-step spread of the two MFU runs (both at 1350 MHz per tt-smi); the lower effective clocks (1.03-1.09 vs 1.08-1.16 GHz) belong to the 4-layer profile runs (review round 9 correction of the first wording).
- HiFi2 (iteration 2, before the hard-constraints block): a microbenchmark of the base matmuls at HiFi2 (`O/iter2_matmul_bench_n*.log`, rel err 4.7e-3 vs HiFi4) was taken once; HiFi4 + fp32 accumulate is a user requirement and no lower fidelity has been evaluated since.
- Iteration 18 run B profile: the chain's Tracy step died in the device profiler (`TT_FATAL: End marker found without a corresponding start marker`, `profiler.cpp:2209`; `O/iter18b_chain.log` rc=1, partial dump `O/iter18b_tp_tracy_tp4_4layers_failed1/`, untracked); the committed profile is a manual rerun of the identical command minutes later, its tt-smi sample file is from the failed attempt. Cause not identified; a plain rerun worked (review round 8).
- Iteration 13's documentation (perf_analysis section and MFU row, README row, work-log section) is in the review-round-7 commit `456f9cf7a8e`, 15 s before the code commit `a1fc69ebed8`: iteration 13 does not bisect with its docs (review round 8). `O/iter18_cmake.log` was committed with iteration 18 by accident and is removed with iteration 19.
- Iteration 20: a kernel run against a stale `_ttml` build (the factory's new CBs not yet compiled in) hung the chip; after the killed run every device open hung until `tt-smi -r` (bounded reset, 2026-10-30 21:09; all four chips back, checks reproduced). Cause: the build step's log path was relative to a changed directory, so the rebuild silently failed before the check.
- Iteration 26 attribution: two device hangs from skeleton variants with the `*_init` calls removed (`O/iter26_attrib3.log` rc=1 twice; not valid programs); `tt-smi -r` 16:30, the remaining variants reran. Removing inits is not a skeleton.
- Iteration 27 profile: clock mix 0.98-1.21 GHz effective (chip 3 at 0.98; tt-smi 1350 MHz, 66-70 C on all chips), chip 0's reduce-scatter row 10.08 ms per profiled step against 2.33 in iteration 26; matmul rows x1.00 on every chip; the MFU run itself tight (464-472 ms per step). Cycle-time rows used.
- Iteration 24: `SwigluPackedBw` 1.096 -> 0.745 ms per 4 layers with the same bytes, only the gradient operand's row stride changed (8192 -> 8256 tiles). Standalone A/B (review round 10, `O/iter28_swiglu_bw_ab.log`): narrow 250 / 240 us, wide +64 249, wide +128 255: no stride effect; the in-step drop is not the operand's stride (the profile's neighbourhood changed with the concat's removal; row noise ~0.07 between profiles), left as an unexplained profile change.
- Iteration 28, first build: the blocked KV kernel put the mask tile in DEST[2] next to a two-row pass; `mask_tile` uses the register after the data register regardless of the argument, so DEST[1] (the second row's scores, or stale DEST on the first pass) acted as the mask: the causal path survived (the -inf add masks on its own) but for one core's first pass (`O/iter28_diag*`: head 3, K row 0, query row 0, S = 0) and the explicit-mask (non-chain) path broke (dK / dV PCC 0.90-0.95, `O/iter28_noncausal_check.log`); a DEST priming cycle had hidden the first-pass symptom and was removed with the fix (masked rows run alone with the mask in DEST[1]; `O/iter28c_*`). The round-10 non-causal case found it.
- Iteration 24 loss match: the single-chip 8-layer run fuses qkv / gate_up / out and never w2 (not a transposed shape), so the match (identical to 23's) does not exercise the wide path; its torch reference is the harness case `test_mlp_fused_lora_wide_w2` at the shard width and the TP gradient check.
- Review round 10 process items: iteration 26 bundled three changes in one commit (each measured on its own; recorded in the commits table), `lpt_schedule`'s unused `initial_load` parameter dropped, the sections 23-27 shortened to the document format (full text in `D`).
- Rank-16 K lanes: `lowrank_update`, `skinny_contract`, the `[r, N]` + transpose path and the previous `ttnn.matmul` path all rely on the padded lanes 16..31 of `h` / `dh` and of `A` / `B` being zero (zero-padded parameters; `h` from `skinny_matmul` whose padded output columns are exact zeros); not validated on device.
