# Qwen3-32B — optimized decoder analysis

**Verdict:** Shipped **DRAM-sharded all-BFP4/LoFi**; advisor 1D-mcast rejected at final precision (1.217 vs 1.737 ms) — Pattern B done right. **Pattern A present in the capture but deliberately avoided as a Copy** (residual moves straight to public DRAM output; 0 Copy ops), though ~118 µs/replay of boundary `ReshapeView` remains (~10%).

_Analyzed read-only from committed artifacts on `mvasiljevic/model/qwen-qwen3-32b`._

## Model & shape
| | |
| --- | --- |
| hidden | 5120 |
| heads / kv | 64 / 8 (GQA 8:1), head_dim 128 |
| intermediate | 25600 |
| layers | 64 (dense) · rep layer 32 |
| batch / seq | 32 / 17 (stressed 31, pos 31–34) |
| device | Blackhole P300 |
| KV cache | BFP8 |
| special | **QK-norm** (per-head RMSNorm on Q and K before RoPE) |

## Headline decode perf
| Path | Prefill (warmed) | Traced decode |
| --- | ---: | ---: |
| functional BF16 | 83.252 | 82.101 |
| **final optimized (all-BFP4/LoFi, DRAM-sharded)** | **5.477** | **1.217** (200 replay) |
| speedup | 15.2× | **67.4×** |

Also 10.9% faster than the strongest earlier correct trace (1.367 ms). PCC 0.99899. Harness: 25 warmup / 200 replay.

## Precision policy shipped
All projections **BFP4 / LoFi** (attention compute too); activations/residuals/norm-weights BF16; KV cache **BFP8**. HiFi2 rejected (raises decode to 1.971 MLP / 1.368 attn-only).

## Decode matmul strategy shipped
All five projections **DRAM-sharded** (`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`; CSV `INPUT_1_MEMORY = DEV_0_DRAM_WIDTH_SHARDED`). QKV 100 cores, O 80, gate 100, up 100, down 80 (32-core K-shard by 500-replay tie-break). Advisor 1D-mcast rejected.

## Shard-advise: applied vs rejected
- Run this pass (26 ops, 23 choices, 1 spill, 6.9% occupancy). Applied directionally: lower-precision grouping as a *seed* only (precision then swept independently on real data); the sharded residual/norm/linear skeleton → shipped 40-core L1 width-sharded chain.
- **Rejected (with evidence):** full legal advisor layouts — decode PCC **0.985 fails the 0.99 bar** (5.627/1.743 ms); advisor 1D matmuls + production heads — correct (0.9993) but slower (5.645/**1.737 ms**). Selected DRAM-sharded is correct (0.99899) and ~30% faster than the correct advisor-matmul trial (1.217 vs 1.737).
- **Unfixable op:** `ttnn.nlp_concat_heads_decode` (requires sharded input).

## Per-op decode matmul table
Tracy CSV `PERF_DECODE` region, per replay. All `LoFi BF16×BFP4`, DRAM-width-sharded weights.

| Role (in0 × in1) | Device µs | Cores | DRAM GB/s | % of 512 |
| --- | ---: | ---: | ---: | ---: |
| packed QKV 32×5120 · 5120×10240 | 100.7 | 100 | 260 | ~51% |
| O 32×8192 · 8192×5120 | 79.6 | 80 | 263 | ~51% |
| gate 32×5120 · 5120×25600 | 233.4 | 100 | 281 | ~55% |
| up 32×5120 · 5120×25600 | 233.3 | 100 | 282 | ~55% |
| down 32×25600 · 25600×5120 | 220.9 | 80 | 297 | ~58% |

Five matmuls = 867 µs/replay (~72% of 1.202 ms device). (DRAM % column blank under trace replay; GB/s from `tt-perf-report`.) SDPA-decode ~50 µs (8×8), NLPCreateQKVHeads ~29 µs. **No Copy/Tilize/Untilize/Typecast in the decode region.**

## Pattern A — segment-boundary reverts
**Present in capture, avoided as a Copy.** `final_ir.mlir` line 51: `reshape → return: l1/interleaved/10x11 → dram/interleaved (output revert)` + entry DRAM→L1 conversions. The shipped path recognized this — "the last residual moves directly to the public DRAM output before its view, avoiding an extra copy" — Tracy confirms **zero CopyDeviceOperation**. But the boundary still costs a ~50 µs input + ~68 µs output DRAM→DRAM `ReshapeView` ≈ **118 µs/replay (~10%)**; `perf_report.md` estimates only ~3 µs recoverable since the path is traced.

## Pattern B — strategy × precision
**Not stale.** An early MLP-core seed sweep used the shared-shard precision, but geometry/strategy was **re-swept coherently at final all-BFP4** (500-replay tie-break → down-32 at 1.217 ms), and DRAM-sharded-vs-1D was compared at the shipped precision (1.217 vs 1.737). No dominant matmul below 40% DRAM (QKV/O ~51%, gate/up ~55%, down ~58%) — remaining gap is projection kernel/dataflow efficiency, not a missed strategy.

## Notable
QK-norm shipped as two block-sharded RMSNorms (Q 40 cores, K 32 cores) before RoPE. Packed QKV (10240) and fused SwiGLU (SiLU activation arg, ~34 µs). Packed gate/up rejected (1.551 vs 1.219). Capacity: batch-32 prefill passes at 8192; 16384 hits a hard DRAM OOM. Roofline 0.540 ms; wall = 2.25× that.
