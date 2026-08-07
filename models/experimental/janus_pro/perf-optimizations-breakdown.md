# Janus-Pro vision tower: optimization breakdown

One page. Full detail in [PERF.md](PERF.md) and [`perf_reports/`](perf_reports/).

## Result

| | before | after |
|---|---:|---:|
| kernel time, one forward pass | 29.501 ms | **9.841 ms** |
| device ops per pass | 393 | 295 |
| binding accuracy gate (`test_vision_transformer`, 0.99) | 0.998631 | **0.998811** |

**−66.6% of device compute time, and the strictest accuracy gate ended higher than it started.**
Wormhole N150. 27 stages, each re-measured through the same harness — same warm-up, same trace,
same ten replays — so the deltas sum.

## Where the 19.5 ms came from

Four op families carry 96% of it:

| what was slow | why it was slow | what fixed it | saved |
|---|---|---|---:|
| matmuls | bfloat16 weights at the most expensive math fidelity, no explicit program configs | narrower dtypes, lower fidelity, sharded outputs, per-shape configs | **7.4 ms** |
| 148 elementwise ops | each bias was its own op, ~29 us of kernel time regardless of its arithmetic | fold every bias into the matmul before it | **5.6 ms** |
| 24 standalone gelus | gelu ran as a separate op after the MLP's first matmul | fuse it into that matmul, then approximate it | **2.9 ms** |
| attention (SDPA) | chunk sizes did not divide 576 evenly; ran at the highest fidelity | chunk 192, one k-iteration, HiFi2 without fp32 accumulation | **2.8 ms** |

The five largest single changes are half the total on their own:

| Δ kernel | change |
|---:|---|
| −3.343 ms | HiFi2 across the tower, MLP gelu fused into its matmul |
| −2.321 ms | approximate gelu in `c_fc` |
| −1.729 ms | SDPA chunk 256 → 192 |
| −1.456 ms | `wo` and `c_proj` outputs written as L1 block shards |
| −1.373 ms | bfloat8_b projection weights |

## Every step, where it was, and what it gained

27 stages. Each was checked out and re-measured through the same harness, so the deltas sum to the
total. Every row links to its own report: the explanation of what was done, next to that stage's
per-op and per-matmul breakdown.

| # | layer | change | Δ kernel | kernel after |
|--:|---|---|---:|---:|
| 0 | — | **baseline, unoptimized** | — | **29.501** |
| 1 | whole tower, MLP | [HiFi2 across the tower, fused MLP gelu](perf_reports/01-hifi2-fused-gelu.md) | -3.327 ms | 26.174 |
| 2 | SDPA | [SDPA chunk 256 → 192](perf_reports/02-sdpa-chunk-192.md) | -1.729 ms | 24.445 |
| 3 | MLP `c_fc` | [c_fc as 1D reuse, output in L1](perf_reports/03-cfc-1d-reuse-l1.md) | -0.909 ms | 23.536 |
| 4 | MLP `c_fc` | [Approximate gelu in c_fc](perf_reports/04-approx-gelu.md) | -2.321 ms | 21.215 |
| 5 | attn `qkv` | [qkv bias fused into its matmul](perf_reports/05-qkv-bias-fused.md) | -1.178 ms | 20.037 |
| 6 | attn `wo`, MLP `c_proj` | [Post-reduce biases fused, single-device only](perf_reports/06-post-reduce-biases-fused.md) | -0.902 ms | 19.135 |
| 7 | aligner | [Aligner biases fused, aligner to HiFi2](perf_reports/07-aligner-biases-hifi2.md) | -0.191 ms | 18.944 |
| 8 | all body projections | [bfloat8_b projection weights](perf_reports/08-bfp8-projection-weights.md) | -1.373 ms | 17.571 |
| 9 | attn `qkv` | [bfloat8_b fused qkv output](perf_reports/09-bfp8-qkv-output.md) | -1.276 ms | 16.295 |
| 10 | MLP | [bfloat8_b c_fc intermediate](perf_reports/10-bfp8-cfc-intermediate.md) | -0.433 ms | 15.862 |
| 11 | attn `wo`, MLP `c_proj` | [bfloat8_b wo and c_proj outputs](perf_reports/11-bfp8-branch-outputs.md) | -0.574 ms | 15.288 |
| 12 | aligner | [bfloat8_b aligner weights](perf_reports/12-bfp8-aligner-weights.md) | -0.215 ms | 15.073 |
| 13 | SDPA | [Asymmetric SDPA chunks](perf_reports/13-asymmetric-sdpa-chunks.md) | -0.190 ms | 14.883 |
| 14 | both norms | [bfloat8_b layer-norm outputs](perf_reports/14-bfp8-norm-outputs.md) | -0.508 ms | 14.375 |
| 15 | both norms | [Block-sharded layer norm on 48 cores](perf_reports/15-block-sharded-layernorm.md) | -0.544 ms | 13.831 |
| 16 | `qkv`, `wo`, `c_proj` | [Explicit 2D configs, in0_block_w per shape](perf_reports/16-explicit-2d-configs.md) | -0.350 ms | 13.481 |
| 17 | attn `wo`, MLP `c_proj` | [wo and c_proj outputs L1 block-sharded](perf_reports/17-wo-cproj-sharded.md) | -1.456 ms | 12.025 |
| 18 | attn `qkv` | [qkv output L1 block-sharded](perf_reports/18-qkv-output-sharded.md) | -0.288 ms | 11.737 |
| 19 | attn | [qkv unshard into L1 rather than DRAM](perf_reports/19-qkv-unshard-to-l1.md) | -0.167 ms | 11.570 |
| 20 | SDPA | [SDPA HiFi4 → HiFi2](perf_reports/20-sdpa-hifi2.md) | -0.170 ms | 11.400 |
| 21 | SDPA | [fp32 dest accumulation off on SDPA](perf_reports/21-sdpa-no-fp32-acc.md) | -0.509 ms | 10.891 |
| 22 | `qkv`, `wo`, `c_fc`, `c_proj` | [LoFi on the body matmuls](perf_reports/22-lofi-body-matmuls.md) | -0.693 ms | 10.198 |
| 23 | `ln_1` + `qkv` | [ln_1's shard fed to qkv in place](perf_reports/23-ln1-shard-into-qkv.md) | -0.109 ms | 10.089 |
| 24 | `ln_2` + `c_fc` | [ln_2's shard fed to c_fc in place](perf_reports/24-ln2-shard-into-cfc.md) | -0.073 ms | 10.016 |
| 25 | aligner | [Aligner activation fused into its matmul](perf_reports/25-aligner-activation-fused.md) | -0.033 ms | 9.983 |
| 26 | MLP `c_fc` | [c_fc output block-sharded in L1](perf_reports/26-cfc-output-block-sharded.md) | -0.142 ms | **9.841** |

## Where to look next

| for | see |
|---|---|
| the 27-stage table with layer, type, delta, PCC and what selected each change | [PERF.md](PERF.md#change-log) |
| one stage's explanation next to its own per-op and per-matmul breakdown | [`perf_reports/NN-*.md`](perf_reports/) |
| what was tried and did not pay, and the largest win deliberately left out | [`perf_reports/DEAD_ENDS.md`](perf_reports/DEAD_ENDS.md) |
| how the profiler misleads on this tower | [`perf_reports/PROFILER_NOTES.md`](perf_reports/PROFILER_NOTES.md) |
| reproducing any of it | [PERF.md](PERF.md#how-to-reproduce) |
