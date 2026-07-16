# Mistral-Small-24B-Instruct-2501 — optimized decoder analysis

**Verdict:** Shipped **DRAM-sharded BFP4/LoFi** for all five matmuls and **rejected the advisor's 1D-mcast at the final precision** (1.288 vs 1.788 ms) — i.e. it did Pattern B *correctly* (unlike Llama-3.1-8B). **Pattern A is present** (advisor output-revert), but the residual is kept L1-resident; the ~119 µs/step boundary reshapes are the public `[1,32,1,H]` interface flatten and would be largely elidable in a stacked model.

_Analyzed read-only from committed artifacts on `mvasiljevic/model/mistralai-mistral-small-24b-instruct-2501` (not re-run on device)._

## Model & shape
| | |
| --- | --- |
| hidden | 5120 |
| heads / kv | 32 / 8 (GQA 4:1), head_dim 128 |
| intermediate | 32768 |
| layers | 40 (dense, SwiGLU) · rep layer 20 |
| batch / seq | 32 / 18, cache 128 |
| device | Blackhole P300 (grid 11×10) |
| KV cache | BFP8 |

## Headline decode perf
| Path | Prefill ms | Traced decode ms |
| --- | ---: | ---: |
| functional BF16 | 94.153 | 93.315 |
| **selected (all-BFP4/LoFi, DRAM-sharded)** | **5.387** | **1.288** |
| speedup | 17.5× | **72.4×** |

Harness: warmed prefill + traced decode, 3 warmup / 100 replay. Real PCC 0.99980 prefill / 0.99983 decode.

## Precision policy shipped
All projections **BFP4 / LoFi**; KV cache **BFP8**; SDPA compute **HiFi4**; norms BF16 (11-core block-sharded). BFP4/LoFi won the sweep (1.288) over BFP8/HiFi2 (2.420), BFP8/LoFi (1.723), BFP4/HiFi2 (2.228).

## Decode matmul strategy shipped
All five dominant matmuls are **DRAM-sharded** (`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`), BFP4/LoFi. Separate gate/up (packed decode measured 1.857 ms, loses). No MoE / no LM-head in this stage.

## Shard-advise: applied vs rejected
- **Applied:** 11-core block-sharded norms; coherent **L1 width-sharded residual chain** (improved decode 1.529 → 1.377 ms); width-sharded QKV/O + direct QKV→heads / concat→O edges (retuned to DRAM-sharded BFP4 block-16); full-table indexed RoPE (removed two slices + both untilize/tilize triplets); separate gate/up.
- **Rejected (with evidence):** exact advisor **1D-mcast** seed → 1.788 ms vs 1.288 selected; BFP4/HiFi2 → 2.228; packed gate/up decode → 1.857.
- **Unfixable op:** `ttnn.nlp_concat_heads_decode` (needs sharded input; abstract input is DRAM) — runtime repairs SDPA→height shard→concat→one L1 reshard into O.

## Per-op decode matmul table
From committed `tracy/final/decode_perf_report.txt` (Blackhole, 110 cores). All `LoFi BF16×BFP4→BF16`, DRAM-sharded, 12 workers.

| Role (M×K×N) | Device µs | Cores | DRAM GB/s | DRAM % |
| --- | ---: | ---: | ---: | ---: |
| QKV 32×5120×6144 | 59 | 12 | 268 | 52.4% |
| O 32×4096×5120 | 41 | 12 | 258 | 50.3% |
| gate 32×5120×32768 | 292 | 12 | 287 | 56.0% |
| up 32×5120×32768 | 295 | 12 | 285 | 55.6% |
| down 32×32768×5120 | 286 | 12 | 293 | 57.2% |

Other: SDPA-decode 43 µs/64c; 2× PagedUpdateCache 12 µs/32c; SiLU-fused multiply 54 µs/110c. **Decode device total 1,281 µs, 31 ops, 42.4% aggregate DRAM.** MLP (gate+up+down) = 873 µs ≈ 67% of decode.

## Pattern A — segment-boundary reverts
**Present.** `final_ir.mlir`: `reshape → return: l1/interleaved/10x11 → dram/interleaved (output revert)`. The pure revert (ShardedToInterleaved) is only 2 µs, but the associated boundary reshapes are real full-tensor ops: **entry ReshapeView 51 µs + exit 68 µs = 119 µs/step (~9.3%)** on 110 cores. The residual itself stays L1 width-sharded within the block, so the *residual-copy* form of the artifact is avoided; the DRAM round-trip is retained by the public single-layer contract and would be a fuse-away candidate in a 40-layer stack.

## Pattern B — strategy × precision
**Not stale — done right.** The DRAM-sharded-vs-1D choice was measured directly at the shipped BFP4/LoFi precision (1.288 vs 1.788 ms), and BFP4 was crossed with geometry. No dominant matmul below 40% DRAM (all 50–57%); the two widest MLP geometries were rejected by a hard L1 blocker (3.98 MB CB > 1.57 MB), not left unexplored.

## Notable
Packed QKV (48 heads→6144); GQA 4:1; fused SiLU into gate×up multiply; `nlp_concat_heads_decode` is the one advisor-unfixable op; prefill keeps a per-user cache-fill loop deferred to a later generator stage.
