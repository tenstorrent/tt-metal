# Falcon3-7B-Base — optimized decoder analysis

**Verdict:** Shipped **DRAM-sharded all-BFP4/LoFi**; the advisor's 1D-mcast was implemented as a control and **rejected at the final precision** (0.773 vs 0.768 ms) — Pattern B done right. **Pattern A present:** a per-step boundary `Copy` (24 µs) + reshapes ≈ **78 µs/iter (~10%)** around an otherwise L1-resident residual.

_Analyzed read-only from committed artifacts on `mvasiljevic/model/tiiuae-falcon3-7b-base`._

## Model & shape
| | |
| --- | --- |
| hidden | 3072 |
| heads / kv | 12 / 4 (GQA 3:1), head_dim 256 |
| intermediate | 23040 |
| layers | 28 (dense) · rep layer 14 |
| batch / seq | 32 (+batch 1), cache 128 |
| device | Blackhole P300 |
| KV cache | BFP8 |

## Headline decode perf
| Path | Traced decode ms (batch 32) |
| --- | ---: |
| functional BF16 | 1.798 |
| **selected (DRAM-sharded, all-BFP4/LoFi)** | **0.768** |
| speedup | **2.34× (57% faster)** |

Batch 1: 1.402 (BF16 advisor) → **0.644**. Harness: 100 traced replays. Selected is ~0.6% faster than the all-BFP4 advisor 1D control.

## Precision policy shipped
All projections **BFP4 / LoFi**; KV cache **BFP8**. MLP BFP4 LoFi vs HiFi2 isolated (0.783 vs 0.786; LoFi wins).

## Decode matmul strategy shipped
All five projections **DRAM-sharded** (12 workers reported). Per-role geometry: QKV 32-way `in0_block_w=3`/`per_core_N=5`; O 32-way `in0=3`/`pcN=3`; gate/up 48-way `in0=2`/`pcN=15`; down 48-way `in0=15`/`pcN=2`. Advisor 1D-mcast (`@11x8/11x9/11x10`) rejected.

## Shard-advise: applied vs rejected
- **Applied:** coherent **32-core L1 width-sharded residual/norm chain** (final, faster than advisor's 96-core residual + 103/90-core MLP grid); RoPE full-table indexing; direct head/O boundary.
- **Rejected (with evidence):** advisor 1D-mcast matmul family lost the all-BFP4 cross (0.773 vs 0.768 batch-32; 0.653 vs 0.644 batch-1); kept as a regression control.
- **Unfixable op:** `ttnn.nlp_concat_heads_decode` (requires sharded input; advisor proposed DRAM interleaved).

## Per-op decode matmul table
From committed `tracy/dense_layer/decode_perf_report.txt`. All `LoFi BF16×BFP4→BF16`, DRAM-sharded, 12 workers.

| Role (M×K×N) | Device µs | Cores | DRAM GB/s | DRAM % |
| --- | ---: | ---: | ---: | ---: |
| QKV 32×3072×5120 | 38 | 12 | 207 | 40.4% |
| O 32×3072×3072 | 28 | 12 | 170 | 33.2% |
| gate 32×3072×23040 | 145 | 12 | 244 | 47.6% |
| up 32×3072×23040 | 146 | 12 | 243 | 47.4% |
| down 32×23040×3072 | 128 | 12 | 277 | 54.0% |

SDPA-decode 38 µs/64c (maskless causal). Per iter ≈ 42 ops, 0.758 ms kernels, 30.6% aggregate DRAM.

## Pattern A — segment-boundary reverts
**Present.** `final_ir.mlir` reshard #18: `reshape → return: l1/interleaved/10x11 → dram/interleaved (output revert)`. Per iteration this materializes as exit `ReshapeView` 34 µs + boundary **`Copy` 24 µs** + next-block entry `ReshapeView` 20 µs ≈ **78 µs/iter (~10% of 758 µs)** on 110/96 cores. Residual chain itself is L1-resident (add ops ~1 µs); the DRAM revert is by public single-layer contract and is the clearest fuse-away candidate in a stacked model.

## Pattern B — strategy × precision
**Not stale.** Precision (BFP4/LoFi) selected first, then the strategy cross (advisor-1D vs DRAM-48, split vs packed) run **at final all-BFP4, 100 replays**. Lowest-util dominant rows are **O (33.2%)** and **QKV (40.4%)** — both small (M=32) and cheap (28/38 µs); the report's only matmul advice is accuracy-oriented, no untried faster config identified for them; the cost center (gate/up/down, 47–54%) is already best-utilized.

## Notable
AutoFix: near-zero DRAM-BFP4 PCC was a decode-formatted weight leaking into the large-M prefill path → fixed with **separate width-sharded decode weight copies** (persistent storage cost, capacity preserved). Maskless causal decode SDPA (0.783 vs 1.004 masked). Split gate/up beats packed (0.773 vs 0.938).
