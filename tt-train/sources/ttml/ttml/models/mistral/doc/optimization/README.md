# Optimisation loop: Mistral-Small-24B-Base-2501 LoRA

Phase 3 of the tt-train bring-up. Contract `../plan.md`; state and numbers `../perf_analysis.md`; commands and artefacts `work_log.md`; raw evidence `evidence/` (`O/`); long prose `evidence/perf_analysis_details.md`.

- Target: 1620 ms cycle-time per 40-layer step at 1350 MHz (1562 kernels + 58 planned dispatch), ±10 %; 1885 ms wall at the delivered 1.13-1.19 GHz.
- Result metric (user scope change): full-model TP=4 MFU against 594 TFLOP/s, random init, cool chips.

## Conventions

- Single-chip Tracy at 1/2/4/8 layers (`analyze_tracy.py` four-point fit, projection to 40) attributes and ranks; `diff_tracy.py` is the cycle-time A/B (targeted ops gone, nothing new). Tracy ms = cycles / 1.35 GHz; each run's delivered clock from `clock_analysis.py`.
- TP=4 full-model MFU after a cool-down (600 s), tt-smi on all four chips, and a 4-layer TP profile checked per chip (`tp_analysis.py`, `tp_rank_ops.py`) for a stalled chip before a TP number is believed.
- PCC harness `tests/python/test_mistral.py` passes before a change lands: 11 tests at the baseline, 13 (iteration 2), 14 (3), 15 (6), 16 (review 4), 20 (9), 32 (review 6), 35 (11), 38 (12), 40 (13), 42 (14), 45 (18), 47 (19), 48 (24).
- From iteration 4 on (user priority 2026-10-29): the TP=4 run is the result; depth runs only rank; no single-chip MFU runs, no 16-layer profiles; HiFi4 + fp32 accumulate mandatory.

## Iterations

| iteration | class | change | harness | TP=4 ms/step | MFU | status |
|---|---|---|---|---|---|---|
| 0 | - | composite baseline | 11 | 598.7 | 27.10 % | baseline |
| 1 | glue | `LoraApply` 3-op port (`modules/lora.py`) | 11 | 574.3 | 28.25 % | landed |
| 2 | math | transposed frozen weights (`modules/frozen_linear.py`) | 13 | 564.5 | 28.74 % | landed |
| 3 | glue + math | LoRA fused into the base matmul (`modules/fused_lora.py`) | 14 | 556.5 | 29.15 % | landed, target met |
| 4 | tail writes + math | `write_tile_block` op, swept gate_up config | 14 | 558.5 | 29.05 % | landed, loop resumed |
| 5 | collectives | batched LoRA-gradient all-reduce, fused wrappers | 14 | 556.9 | 29.13 % | landed |
| 5b | collectives | batched mode for the composite adapters | 15 | 563.2 | 28.81 % | landed |
| 6 | math | `rmsnorm_bw_allcore` op | 15 | 547.5 | 29.63 % | landed |
| 7 | math + glue | qkv / out on the fused path at TP | - | 551.3 | 29.43 % | rejected, reverted |
| 8 | wasted | `FrozenVocabParallelEmbedding` | - | 545.9 | 29.72 % | landed |
| review 3 | wasted | graph-free frozen embedding and no-grad norm | 15 | 545.7 | 29.73 % | landed |
| 9 | math + glue | `add_rmsnorm_fw_allcore` op, residual grad in the backward | 20 | 537.4 | 30.19 % | landed |
| 10 | math | `skinny_matmul` op for the LoRA products | 20 | 536.1 | 30.27 % | landed |
| 11 | math | SDPA LPT work lists, head-split KV backward + reduce op | 35 | 524.7 | 30.92 % | landed |
| 12 | glue + math | `lowrank_update` op (LoRA add and `dx += dh @ A` in one pass) | 38 | 518.5 | 31.29 % | landed |
| 13 | layout | `transpose_split` op (fused backward's dX route in one pass) | 40 | 516.4 | 31.42 % | landed |
| 14 | fixed | `vp_cross_entropy_bw` op (loss backward in one pass) | 42 | 512.5 | 31.66 % | landed |
| 15 | math (kernel boost) | SDPA forward chain data flow (K/V once per chain of cores) | 42 | **506.0** | **32.06 %** | landed |
| 16 | math (kernel boost) | SDPA backward chain data flow (Q, KV kernels) | 42 | 508.4 | 31.91 % | landed, -0.13 ms per 4 layers per chip (compute-bound kernels) |
| 17 | math (kernel boost) | single-phase RMSNorm backward (`rmsnorm_bw_rows`) | 42 | 505.6 | 32.09 % | landed |
| 18 | math (kernel boost, partial) | LoRA gradient products: `skinny_contract` op (narrow dB) + ttnn `[r, N]` orientation (dA, wide dB) | 45 | 503.9 | 32.19 % | landed |
| 19 | math (kernel boost) + config | single-phase fused add + RMSNorm forward (`add_rmsnorm_fw_rows`), 2D w2-forward matmul config | 47 | **495.4** | **32.75 %** | landed |
| 20 | math (kernel boost) | norm kernels' per-tile math on the FPU | - | - | - | rejected: -16 % only at 2.5x the error (19-bit DEST feedback); exact variant no faster |
| 21 | math (kernel boost) | SDPA forward from the `sdpa_scheduler` stack (multi-row items, head stacking, cut heavy items) | 47 (attention subset 4) | - | - | rejected: 186 / 166 us vs 149-157 chain-mode at the shard shape; reverted |
| 22 | math (kernel boost) | SDPA backward data path: Q kernel in K/V blocks, one read barrier per stream chunk in the chain injectors | 47 | **494.5** | **32.81 %** | landed (SDPA rows -4.2 ms/step cycle-time; the -0.9 ms wall is within the run spread) |
| 23 | math (kernel boost) | SDPA chain forwarding on the writer RISC / second NoC, chains reversed | 47 | **480.3** | **33.78 %** | landed (SDPA rows -1.6 ms/step cycle-time; the wall drop beyond that is the run's clock mix) |
| 24 | layout | w2 K-concat removed: the producers write the wide LoRA input buffer (SwiGLU out_cols, skinny_matmul tail write), the wide gradient flows back | 48 | 481.2 | 33.72 % | landed (concat + pad rows gone, SwiGLU backward 1.10 -> 0.75 ms per 4 layers: ~-7 ms/step cycle-time; wall within spread, best step 470.6) |
| 25 | math (kernel boost) | low-rank update fused into transpose_split (one DEST pass per dx tile) | 48 | 476.7 | 34.03 % | landed (gate_up 180 -> 74, w2 277 -> 116 us per call standalone, bit-exact) |
| 26 | math (kernel boost) | SDPA chain credits (three chunks in flight), 8-row forward K/V chunks, forward chains of 8 cores | 48 | 472.5 | 34.33 % | landed (forward 136 -> 115 us per call standalone; bf16 DEST and position-aware LPT rejected with evidence) |
| 27 | math (kernel boost) | per-round chain stream bounds (the chain streams only what the round's rows need), backward chains of 7 | 48 | 468.1 | 34.66 % | landed (backward 387 -> 345, forward 114 -> 110 us per call standalone, bit-exact) |
| 28 | math (kernel boost) | KV backward in query-row blocks (dP in DEST, dV / dK contracted over the block) | 48 | 468.4 (463.5) | 34.63 % | landed (backward 345 -> 323 us per call standalone; a mask-register bug found by the round-10 explicit-mask case, fixed) |
| 29 | math (kernel boost) | RMSNorm backward pass 2 on the FPU (iteration 20's variant A, now admissible) | 48 | 472.2 (best 462.8) | 34.36 % | landed (172 -> 148 us per call, -2.1 ms/step cycle-time; the wall is the run's clock mix) |
| 30 | math (kernel boost) | SDPA forward: the diagonal chunk computes only the tiles up to the diagonal | 48 | **468.4** | **34.63 %** | landed (bit-exact; -0.1 ms/step at the shard shape, -6 % forward at 32 heads) |
| 31 | math (kernel boost) | `skinny_matmul` x reads in flight before one barrier | - | - | - | rejected (48 -> 47 us per call: not latency-bound; evidence `O/iter31_*`) |
| 32 | fixed (kernel boost) | cross-entropy forward statistics in one pass over the logits shard (`vp_ce_row_stats`) | 48 | **464.2 (459.5)** | **34.95 %** | landed (500 -> 215 us standalone at identical precision) |
| 33 | collectives | the backward's dx all-reduce on a second command queue under the LoRA gradient products; dispatch on the Ethernet cores | - | - | - | rejected (one sub-device runs one program at a time across queues; ETH dispatch unavailable here; evidence `O/iter33_*`) |
| 34 | math (ttnn matmul configs) | base matmuls with the full-sync DEST (8-tile output subblocks) | - | - | - | rejected (ttnn's mcast / reuse factories do not support full sync; 4-tile subblocks unchanged; evidence `O/iter34_*`) |
| 12 candidate | collectives | TP collective parameter sweep | - | - | - | rejected: library defaults fastest |

Single-chip cycle-time projection (40 layers): 1658.6 -> 1592.5 (1) -> 1398.7 (2) -> 1339.0 (3) -> 1256.3 (4) -> ~1242 (6); attribution only after iteration 4.

### Per-iteration notes

- 1: glue 134.6 -> 72.9 ms. Lesson: after a long TP run one chip stayed throttled (every matmul 1.68x cycles) and the mesh waited in every collective; hence the cool-down and per-chip checks above.
- 2: gate_up forward 44 -> 63 % of the delivered peak , its dX 68 -> 85 %; single chip 8 layers 380.0 ms / 38.06 %. Lesson: `ttnn.matmul` without an explicit `core_grid` is 1.5-1.8x slower on these shapes; microbenchmarks must mirror the exact call.
- 3: glue 72 -> 36 ms; single chip 375.3 ms / 38.53 %; the first run's tail-scale bug was caught by the fused block harness case. Remaining cost: the one-core `fill_cache` tail write (0.79 ms per gate_up layer, 31.6 ms per step).
- 4: gate_up tail 0.85 -> 0.06 ms, gate_up fwd 7.92 -> 7.35 ms bit-exact; TP unchanged (config keyed on the single-chip shape; ~7 ms tail gain under the clock swing, chip 2 at 1.10 GHz).
- 5 / 5b: no per-layer LoRA collective left (4 partials per layer, two collectives per step); the step follows the slowest chip, so the gain is invisible on this box (5b: identical compute rows, lower effective clock at equal temperature, cause not identified).
- 6: RMSNorm backward 0.37 -> 0.21 ms per call on every chip; new harness case dx PCC 0.999997.
- 7: adapter ops saved (-1.0 ms per 4 layers) paid back by the transposed route's layout ops and a slower out forward.
- 8: the `[32768, 5120]` shard untilize (1.5 ms per step per chip) gone; correctness evidence is the TP gradient check (loss identical), the single-chip harness has no vocab-parallel path.
- 9: -0.86 ms per 4 layers per chip, collectives excluded (the removed adds; the fused forward costs what the 32-core norm did, moving 3x the bytes at bandwidth); latent CB wrap hazard of the ported backward fixed, two-items-per-core cases added; 2-layer model equals the block path.
- 10: 76-122 -> 45-65 us per call, -0.60 ms per 4 layers per chip; standalone PCC 0.999999 vs `ttnn.matmul`; measured at lower effective clocks than iteration 9's run.
- 11: SDPA 3.995 -> 3.044 ms per 4 layers per chip (fwd 268 -> 229 us, bwd Q 271 -> 216, bwd KV 460 -> 289 + 28 reduce); the kernels are now bandwidth-bound on single-row K/V re-reads, next rung Q-chunking. Lesson: a 2-arg `pack_reconfig_data_format(prev, next)` encodes an assumption about the previous pack target; changing an output CB's format invalidates it.
- 12: 16 LoRA-product matmuls + 16 adds per 4 layers -> 16 `lowrank_update` calls, -1.07 ms per 4 layers per chip; standalone 124 -> 72 us at N=5120. Collective sweep: defaults already fastest; the in-step collective time is chip skew.
- 13: two slices + two transposes (+ multiply) per fused layer -> one `transpose_split`, bit-exact, 117 -> 54 us; -0.57 ms per 4 layers per chip.
- 14: the loss backward's four passes over the `[1024, 32768]` shard (3.09 ms, one on 32 cores) -> one 0.33 ms kernel; loss match rerun PASS. Lesson: a `[1, 1, 1, 1]` tensor is a zero-padded tile on device; an elementwise multiply by it is not a scalar broadcast.
- 15: first kernel-boost iteration (user direction 2026-10-30 evening): SDPA forward 228 -> 140 us per call, bit-identical; chains of 14 cores forward the K/V chunks, DRAM traffic 67 -> ~12 MB per call.
- 16: the same for the two backward kernels: -0.13 ms per 4 layers only; they are compute-bound (12-16 matmul tiles plus SFPU per unit), next fix is one merged backward pass with dQ partials.
- 17: RMSNorm backward 229 -> 165 us per call (one program, operands resident, partial dots exchanged over the NoC); -0.51 ms per 4 layers per chip.
- 18: new `skinny_contract` op beat ttnn's `[N, r]` orientation (157 -> 123 us gate_up dB) but not ttnn's `[r, N]` orientation + transpose (110 us); router sends dA and the wide dB there, the narrow qkv dB to the op (25 vs 28 us). Run A (op for the wide dB, not the committed router) 498.0 / 32.58 %, run B (committed) 503.9 ms / 32.19 %; product rows 1.63 -> 1.32 ms per 4 layers per chip.
- 19: fused norm forward in one phase 166 -> 138 us per call standalone, 140 -> 126 in the step (bit-exact; the first resident-slice version was slower in the step, the streaming one landed below the expected 100-125: compute-bound per tile); w2 forward matmul 2D config -11 % standalone, 3.69 -> 3.25 ms per 4 layers (ttnn's 1D choice beaten only for this shape). 495.4 ms / 32.75 %.
- 20 (rejected): FPU per-tile math in both norm kernels; the only fast variant feeds fp32 DEST tiles back through 19-bit source registers (rel error 1.66e-3 -> 4.09e-3 vs fp64); the exact variant is no faster. Kernels unchanged.

## State (2026-11-02)

- TP=4 full model: 598.7 (baseline) -> 556.5 (exit) -> 545.7 (review 3) -> 537.4 (9) -> 536.1 (10) -> 524.7 (11) -> 518.5 (12) -> 516.4 (13) -> 512.5 (14) -> 506.0 (15) -> 508.4 (16) -> 505.6 (17) -> 498.0 (18 A) -> 503.9 (18 B) -> 495.4 (19) -> 494.5 (22) -> 480.3 (23) -> 481.2 (24) -> 476.7 (25) -> 472.5 (26) -> 468.1 (27) -> 468.4 (28) -> 472.2 (29) -> 468.4 (30) -> 464.2 (32) -> **462.9 (459.1) ms / 35.05 %** (35, best step 459.1) -> 463.8 (36, cycle-time -3.4 ms/step, wall inside the clock spread).
- Hard constraints (user, 2026-10-30): HiFi4 + fp32 accumulate (no lower fidelity evaluated); result = full-model TP=4 MFU only; a kernel-level item every iteration; a stage-review pass is a checkpoint, not a stop; one change per commit.
- Loss match on this path (after 14): 8-layer host loss max |diff| 0.0294 vs torch fp64 (tol 0.10); TP=4 real-weight trajectory as the baseline's (526.1 ms warm).
- Reviews: round 3 more-work-needed, answered in the follow-up; round 4 five P3 doc / test items fixed; round 5 clean-pass; round 6 (iterations 9-10) items answered, clean-pass on re-check; round 7 (iterations 11-12, `review_round7.md`) two P2 doc / hygiene items answered; round 8 (iterations 13-18, `review_round8.md`) four P2 doc items answered in the iteration-19 commit; round 9 (iterations 19-22, `review_round9.md`) more-work-needed: one P1 (the blocked Q kernel's non-chain modes straddle the K/V CB; fixed with `kv_block = 1` outside the chain and a list-mode check in the commit after iteration 23) and P2 doc items (answered in the docs commit that follows iteration 22).
- 36: the fused w2 forward in the transposed orientation (`W_cat^T @ x_cat^T`, the weight as the row-multicast operand): 88 -> 110 cores, 0.883 -> 0.730 + 0.060 ms per call standalone, a second w2 weight copy (+3.4 GB per chip); cycle-time -0.34 ms per 4 layers (w2 fwd 3.25 -> 2.66, transpose +0.26); wall 468.2 / 463.8 ms (34.65 / 34.98 %) at clock mixes 1.02-1.15 GHz, inside the spread.
- 35: ttnn matmul kernel attribution: operand streaming is 50-77 % of each base matmul's time, the pack nothing, the FPU thread at 71 cycles per tile matmul with ttnn's 2-tile K blocks; the TP gate_up forward at 9-tile K blocks (-5 % standalone, 462.9 ms / 35.05 %, gate_up fwd 6.15 -> 5.85 ms per 4 layers); the writer-kernel and L1-sharded-activation experiments gave nothing (reverted).
- 34 (rejected): the matmuls' 72-85 % of peak is a ttnn kernel limit (half-sync DEST in the mcast factories), not a config one.
- 33 (rejected): the collectives cannot overlap compute on this device without a sub-device split that costs more matmul cores than it hides; dispatch on the Ethernet cores is unavailable. The collectives are a ~40 ms/step floor (plus the clock skew) at TP=4.
- 32: the cross-entropy forward's max and sum-of-exp in one pass each over the logits shard (new `vp_ce_row_stats`): 500 -> 215 us standalone at the old precision.
- 30: the forward's diagonal chunk computes only the tiles up to the diagonal (bit-exact): -0.1 ms/step at the shard shape, more at longer sequences.
- 29: the norm backward's pass 2 on the FPU (iteration 20's rejected variant, admissible under the relaxed rule): 172 -> 148 us per call, -2.1 ms/step.
- 28: the KV backward in query-row blocks (8 DEST passes per 4 rows instead of 20): backward 345 -> 323 us per call; `mask_tile` takes its mask from the register after the data register, so masked rows run alone (bug found by the review's explicit-mask case).
- 27: the chain streams per round only the chunks its rows need (half of before): backward 387 -> 345 us per call, bit-exact; -1.8 ms/step cycle-time.
- 26: SDPA attribution: the math is 35-45 % of each chain kernel, the rest per-stage pipeline latency and the per-core fixed cost; credits + 8-row chunks + chains of 8 took the forward 136 -> 115 us per call; bf16 DEST measured slower at PCC 0.9999 (rejected).
- Review round 10 (iterations 23-27, `review_round10.md`): more-work-needed, no P1; five P2 doc / process items answered in the round-10 commit; device items (bf16 DEST as intended, non-causal case, SwiGLU backward A/B) after iteration 28.
- 25: `lowrank_update` folded into `transpose_split` (the s_dh tile times A's tile column, added to the transposed tile in the same DEST pass): the second pass over dx is gone, 180 -> 74 and 277 -> 116 us per call, bit-exact.
- 24: the w2 K-concat and pad removed (the SwiGLU writes the [T, in + 64] buffer, skinny_matmul writes h into its tail, transpose_split / lowrank_update / the SwiGLU backward take the wide gradient): ~-7 ms/step cycle-time, 481.2 ms / 33.72 % (wall within spread). gate_up keeps its concat (its wide dx would break the TP all-reduce's tile split).
- 23: the chain hop's receive and forward overlapped (writer RISC / NOC_1, chains reversed to NOC_1's routing direction): forward 145 -> 137, backward 427 -> 393 us standalone, bit-exact; 480.3 ms / 33.78 %.
- 22: SDPA backward 532 -> 427 us standalone, in the step KV 1.093 -> 0.896, Q 0.784 -> 0.574, fwd 0.558 -> 0.545 ms per 4 layers; found by attribution that the chain kernels are stream-bound (a math-free skeleton costs the same); the blocked Q kernel gave -3 %, the injectors' single read barrier the rest; multicast stream tried, no gain, deadlocked, removed. 494.5 ms / 32.81 %.
- 21: the `sdpa_scheduler` forward measured at the shard shape: 186 / 166 us vs 149-157 chain-mode (64 four-row items on 110 cores); reverted. Constraint change 2026-10-31: bf16 intermediates allowed where the PCC holds; the loop continues toward 40 %.
- State after iteration 27 (`../perf_analysis.md`, "Ranked cost table" and "State after iteration 20", corrected): the owned kernels are ~95 ms/step (SDPA 17, norms 24, adapters 25, SwiGLU 13, layout 12) with ~10 ms of remaining rungs; iterations 22-27 took ~20 ms/step from their data paths after the "exact-fp32 ceilings" statement of iteration 20 (their compute was at its ceiling, their streams and schedules were not); the gap to 40 % MFU (405 ms from 464) is the ttnn matmuls (~201 ms at 72-85 % of peak: a kernel limit, iteration 34) and the collectives (40-90 ms incl. skew: not overlappable on one sub-device, iteration 33); under the constraints (HiFi4 + fp32 accumulate, TP=4, batch 1, seq 1024) the step's ceiling on this box is ~36-37 %, and 40 % needs a recipe or ttnn-kernel decision (see the loop's last recap).
- Remaining ranking (`../perf_analysis.md`, "Ranked cost table", after iteration 18, ms/step/chip): base matmuls ~206 (w2 forward at 63 % of peak, the rest 76-85 %), collectives 41-79 (transfer at the library optimum, chip skew on top, overlap untried), LoRA adapters ~34, norms ~26, SDPA ~25, layout ~17, SwiGLU ~14, fixed ~8, adds ~3.

## Exit (2026-10-29, after iteration 3)

- Single chip: 1339 ms cycle-time per 40-layer step (baseline 1659), 17 % under the 1620 ms target, 14 % under its kernel budget; 1661-1739 ms wall at the runs' 1090 / 1041 MHz vs the target's 2006-2101 ms at those clocks; 2/4/8/16 fit 1346 ms.
- Full model TP=4: 556.5 ms / 29.15 % (baseline 598.7 / 27.10 %); real weights 574.8 ms warm with the baseline's loss trajectory.
- Verification: harness 14; loss match max |diff| 0.026 (bar 0.10); TP=4 gradient check PCC ≥ 0.9999 / rel ≤ 1.4e-2 on the composite and fused paths (`tp_gradient_check.py`).
- Memory: 13.8 GiB after the forward at 8 layers, 279 MiB retained per layer.
- Highest remaining items at exit: gate_up forward at 63 % of the delivered peak, SDPA backward ~10x roofline, `fill_cache` tail writes (37 ms), residual / `dx` accumulation adds (33 ms), RMSNorm backward on 32 cores, w2 LoRA skinny matmuls; overhead items above 13 ms carry their rung and "deferred: target met" in the gap decomposition (`../perf_analysis.md`, "Loop exit").
