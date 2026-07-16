# Falcon3-10B-Base — optimized decoder analysis

**Verdict:** Shipped **DRAM-sharded all-BFP4/LoFi** with a **precision-aware core target** (BFP4→24 cores); advisor 1D-mcast rejected at final precision (0.793 vs 0.804 ms) — Pattern B done right. **Pattern A present:** exit `ReshapeView` 34 µs + boundary `Copy` 25 µs ≈ **59 µs/layer (~7.3%)**, characterized as the public DRAM layer boundary.

_Analyzed read-only from committed artifacts on `mvasiljevic/model/tiiuae-falcon3-10b-base`._

## Model & shape
| | |
| --- | --- |
| hidden | 3072 |
| heads / kv | 12 / 4 (GQA 3:1), head_dim 256 |
| intermediate | 23040 |
| layers | 40 (dense) · rep layer 20 |
| batch / seq | 32 (+batch 1), cache 128 (capacity 6528) |
| device | Blackhole P300 |
| KV cache | BFP8 (functional was BF16) |

## Headline decode perf
| Path | Traced decode ms (batch 32) |
| --- | ---: |
| functional BF16 | 4.201 |
| **selected (DRAM-sharded, all-BFP4/LoFi, 24-core)** | **0.793** |
| speedup | **5.29× (81% faster)** |

Batch 1: 1.433 → **0.668**. PCC 0.99999667. Harness: 100 traced replays.

## Precision policy shipped
All projections **BFP4 / LoFi**; KV cache **BFP8** (cache PCC 0.9966/0.9950). Rejected: attention-HiFi2 (no accuracy need), MLP-HiFi2 (40.7% slower decode).

## Decode matmul strategy shipped
All five projections **DRAM-sharded** (12 workers). **Precision-aware core target:** BFP4→24-core, BFP8/BF16→48-core; `in0_block_w=4` gate/up, `in0_block_w=30` down. Advisor 1D-mcast (`@11x8/11x9/11x10`) seeded, measured, rejected.

## Shard-advise: applied vs rejected
- Run once (24 ops, 21 choices, 1 spill). Every recommendation seeded then **rejected on whole-layer measurement**: report-sharded-inputs chain 0.819; report residual chain 0.817; exact advisor all-BFP4 0.804 — all lose to selected **0.793**. Exact advisor BFP4 kept as a regression test.
- Residual: advisor 96-core residual + 103/90-core MLP grid → 0.817 vs selected 24-core phase-specific → 0.793.
- **Unfixable op:** `ttnn.nlp_concat_heads_decode` (requires sharded input).

## Per-op decode matmul table
From committed `tracy/.../decode_perf_report.txt`. All `LoFi BF16×BFP4→BF16`, DRAM-sharded, 12 workers.

| Role (M×K×N) | Device µs | Cores | DRAM GB/s | DRAM % |
| --- | ---: | ---: | ---: | ---: |
| QKV 32×3072×5120 | 38 | 12 | 207 | 40.4% |
| O 32×3072×3072 | 28 | 12 | 168 | 32.8% |
| gate 32×3072×23040 | 133 | 12 | 267 | 52.2% |
| up 32×3072×23040 | 133 | 12 | 266 | 51.9% |
| down 32×23040×3072 | 123 | 12 | 289 | 56.4% |

Per iteration ≈ 42 ops, 0.782 ms kernels, 29.7% aggregate DRAM. SDPA-decode 38 µs/64c (maskless).

## Pattern A — segment-boundary reverts
**Present.** `final_ir.mlir`: exit `reshape → return: l1/interleaved/10x11 → dram/interleaved (output revert)` + two entry `dram→dram [to_layout]` conversions. Reproduced per layer as `ReshapeView` 34 µs + `Copy` 25 µs ≈ **59 µs/layer (~7.3%)** on 110 cores — the largest non-matmul rows, flagged "High Op-to-Op Gap". Residual L1-resident within the layer; only the layer output reverts to public DRAM (single-layer contract). In a 40-layer stack this revert-then-re-enter would be a fuse-away candidate.

## Pattern B — strategy × precision
**Not stale.** DRAM-sharded-vs-1D and 24-vs-48-core were re-measured on all-BFP4/LoFi with real layer-20 weights (0.793 vs 0.804; 200-replay 0.793@24 vs 0.799@48); default is precision-aware. **Low-DRAM% hint:** **O (32.8%)** and QKV (40.4%) are the least-efficient dominant rows. A **per-op mixed strategy (DRAM-sharded MLP + 1D-mcast QKV/O)** was *not* tried in isolation and is the plausible remaining lever for the ~33% O row — though upside is bounded (~28 µs O row; MLP is already near roofline).

## Notable
Maskless SDPA (mask removed) was the biggest graph-rewrite win (2.62 → 0.80 ms). Separate decode weight copies (prefill interleaved / decode sharded). Smaller-core geometries hit exact L1 blockers (12-core gate/up `in0=8` = 1.585 MB > 1.573 MB). Bytes floor 0.265 ms = 32% of wall.
