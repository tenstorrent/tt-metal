# Optimisation loop: Mistral-Small-24B-Base-2501 LoRA

Phase 3 of the tt-train bring-up. Contract `../plan.md`; state and numbers `../perf_analysis.md`; commands and artefacts `work_log.md`; raw evidence `evidence/` (`O/`); long prose `evidence/perf_analysis_details.md`.

- Target: 1620 ms cycle-time per 40-layer step at 1350 MHz (1562 kernels + 58 planned dispatch), ±10 %; 1885 ms wall at the delivered 1.13-1.19 GHz.
- Result metric (user scope change): full-model TP=4 MFU against 594 TFLOP/s, random init, cool chips.

## Conventions

- Single-chip Tracy at 1/2/4/8 layers (`analyze_tracy.py` four-point fit, projection to 40) attributes and ranks; `diff_tracy.py` is the cycle-time A/B (targeted ops gone, nothing new). Tracy ms = cycles / 1.35 GHz; each run's delivered clock from `clock_analysis.py`.
- TP=4 full-model MFU after a cool-down (600 s), tt-smi on all four chips, and a 4-layer TP profile checked per chip (`tp_analysis.py`, `tp_rank_ops.py`) for a stalled chip before a TP number is believed.
- PCC harness `tests/python/test_mistral.py` passes before a change lands: 11 tests at the baseline, 13 (iteration 2), 14 (3), 15 (6), 16 (review 4), 20 (9), 32 (review 6), 35 (11), 38 (12), 40 (13), 42 (14).
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
| 14 | fixed | `vp_cross_entropy_bw` op (loss backward in one pass) | 42 | **512.5** | **31.66 %** | landed |
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

## State (2026-10-30)

- TP=4 full model: 598.7 (baseline) -> 556.5 (exit) -> 545.7 (review 3) -> 537.4 (9) -> 536.1 (10) -> 524.7 (11) -> 518.5 (12) -> 516.4 (13) -> **512.5 ms / 31.66 % MFU** (14).
- Hard constraints (user, 2026-10-30): HiFi4 + fp32 accumulate (no lower fidelity evaluated); result = full-model TP=4 MFU only; a kernel-level item every iteration; a stage-review pass is a checkpoint, not a stop; one change per commit.
- Loss match on this path (after 14): 8-layer host loss max |diff| 0.0294 vs torch fp64 (tol 0.10); TP=4 real-weight trajectory as the baseline's (526.1 ms warm).
- Reviews: round 3 more-work-needed, answered in the follow-up; round 4 five P3 doc / test items fixed; round 5 clean-pass; round 6 (iterations 9-10) items answered, clean-pass on re-check; round 7 (iterations 11-12, `review_round7.md`) two P2 doc / hygiene items answered, re-check pending.
- Remaining ranking (`../perf_analysis.md`, "Ranked cost table", after iteration 12, ms/step/chip): base matmuls ~206 at ttnn's best config, collectives 40-85 (transfer at the library optimum, chip skew on top, overlap untried), LoRA adapters ~40, norms ~32, SDPA ~30, layout ~29, SwiGLU ~15, fixed ~10, adds ~5.

## Exit (2026-10-29, after iteration 3)

- Single chip: 1339 ms cycle-time per 40-layer step (baseline 1659), 17 % under the 1620 ms target, 14 % under its kernel budget; 1661-1739 ms wall at the runs' 1090 / 1041 MHz vs the target's 2006-2101 ms at those clocks; 2/4/8/16 fit 1346 ms.
- Full model TP=4: 556.5 ms / 29.15 % (baseline 598.7 / 27.10 %); real weights 574.8 ms warm with the baseline's loss trajectory.
- Verification: harness 14; loss match max |diff| 0.026 (bar 0.10); TP=4 gradient check PCC ≥ 0.9999 / rel ≤ 1.4e-2 on the composite and fused paths (`tp_gradient_check.py`).
- Memory: 13.8 GiB after the forward at 8 layers, 279 MiB retained per layer.
- Highest remaining items at exit: gate_up forward at 63 % of the delivered peak, SDPA backward ~10x roofline, `fill_cache` tail writes (37 ms), residual / `dx` accumulation adds (33 ms), RMSNorm backward on 32 cores, w2 LoRA skinny matmuls; overhead items above 13 ms carry their rung and "deferred: target met" in the gap decomposition (`../perf_analysis.md`, "Loop exit").
