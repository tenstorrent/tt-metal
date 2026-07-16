# Llama-3.1-70B-Instruct — optimized decoder analysis

**Verdict:** The **second Pattern-B gap** (with Llama-3.1-8B). Ships **1D-mcast all-BFP4/LoFi**, but the DRAM-sharded-vs-1D decision was adjudicated at **BFP8-attention precision** and the all-BFP4 sweep ran **only on the 1D family** — the O matmul sits at 38.7% DRAM. (Mitigated: lower-core DRAM-sharded hits a hard L1 limit here.) Also notable: the source **TP=4 / 2×2-mesh** capture was **collapsed to dense single-device** — no collectives in the shipped decode.

_Analyzed read-only from committed artifacts on `mvasiljevic/model/meta-llama-llama-3.1-70b-instruct`._

## Model & shape
| | |
| --- | --- |
| hidden | 8192 |
| heads / kv | 64 / 8 (GQA 8:1), head_dim 128 |
| intermediate | 28672 |
| layers | 80 · rep layer 39 |
| batch / seq | 32 (+batch 1 primary) / 18, cache 128 |
| device | **1× Blackhole P300** — `MeshShape(1,1)` (source captured 2×2 mesh, TP=4, collapsed to dense) |
| KV cache | BF16 primary (BFP8 supported) |

## Headline decode perf
| Path (batch 1) | Prefill ms | Traced decode ms | tok/s/user |
| --- | ---: | ---: | ---: |
| functional BF16 | 5.059 | 4.917 | ~203 |
| **final optimized** | **3.179** | **1.846** | **~542** |
| speedup | 1.59× | **2.66×** | |

Batch 32: 142.9 → **2.114 ms** decode (67.6×). Harness: 5 warmed prefill + **500 traced replays** for signoff. PCC 0.99990 prefill / 0.99995 decode (BFP4 attention drops K/V-append PCC to ~0.993, above the 0.99 bar).

## Precision policy shipped
All projections **BFP4 / LoFi**; activations/norms BF16; **KV cache BF16** primary (BFP8 supported/caller-owned). No multi-chip → no CCL payload.

## Decode matmul strategy shipped
All five projections **1D-mcast** (`MatmulMultiCoreReuseMultiCast1DProgramConfig`, DRAM-sharded rejected). QKV 11×10/107c `in0=2`/`pcN=3`; O 11×8/86c `in0=2`/`pcN=3`; gate/up 11×10/100c each `pcN=9`; **down tuned off the advisor seed to 11×6/64c `in0=4`/`pcN=4`** (was 11×8/86c → 589 µs; tuned → 423 µs). No row/column-parallel decomposition, no collectives (TP collapsed to dense).

## Shard-advise: applied vs rejected
- Run this pass (28 ops, 25 choices, 1 spill), single-chip (1×1). Applied: QKV 107c, O 86c, gate/up 100c; down as a **seed then tuned** (11×8→11×6).
- **Rejected (with evidence):** block-sharded 11-core norm + exact 86-core residual chain — PCC-fine but **2.545 vs 2.454 ms decode (~3.7% slower)**; advisor L1 mask layouts / explicit SDPA (composite faster); full-phase packed gate/up (2.26 MB CB > 1.57 MB L1).
- **Unfixable op:** `ttnn.nlp_concat_heads_decode` ("Input tensor must be sharded") — runtime inserts a targeted InterleavedToSharded (~1.9 µs).
- **Provenance gap** (flagged by stage-review): the saved advisor IR is a **BFP8-attention / 11×8-down seed**, while the shipped runtime is all-BFP4 / 11×6-down; the capture script froze the saved-IR policy and later precision/geometry were swept on hardware.

## Per-op decode matmul table
One replay (device ≈ 2.083 ms, ~43 ops). All `LoFi BF16×BFP4→BF16`, 1D-mcast.

| Role (M×K×N) | Device µs | Cores | DRAM GB/s | DRAM % |
| --- | ---: | ---: | ---: | ---: |
| QKV 32×8192×10240 | 194 | 107 | 216 | 42.2% |
| O 32×8192×8192 | 169 | 86 | 198 | **38.7%** |
| gate 32×8192×28672 | 426 | 100 | 275 | 53.8% |
| up 32×8192×28672 | 425 | 100 | 276 | 53.9% |
| down 32×28672×8192 | 423 | 64 | 278 | 54.3% |

SDPA-decode 74 µs/110c; NLPCreateQKVHeads 29 µs; 2× PagedUpdateCache 19.5+19.9 µs. **Entry ReshapeView 79.9 µs + exit ReshapeView 91.4 µs** (both 110c) — see Pattern A. Decode roofline 40.1% / 205 GB/s; MLP (gate+up+down) ≈ 1,274 µs dominates.

## Pattern A — segment-boundary reverts
**Present; two large boundary `ReshapeView` ops (not a Copy).** Entry ~79.9 µs (`DRAM_INTERLEAVED` → l1/interleaved feeding rms_norm) + exit ~91.4 µs (`L1_WIDTH_SHARDED` residual → l1/interleaved layer output, named `(output revert)` in final-IR). Combined **≈171 µs/replay ≈ 8.2%** of decode. Residual is width-sharded L1 *inside* the block; the block enters/exits via interleaved to satisfy the public single-layer contract. In a full-model L1-resident-residual stack these two reshapes are the fuse-away candidate; the exact lower-movement chain that removes them was measured and lost by 3.7%.

## Pattern B — strategy × precision
**Partially frozen early — a genuine gap (as in Llama-3.1-8B).** DRAM-sharded-vs-1D was adjudicated at **BFP8-attention/BFP4-MLP** (repaired 32-core DRAM-sharded 2.600 vs advisor 1D 2.454 ms at that precision); the subsequent all-BFP4 precision + geometry sweeps ran **only on the 1D family** — no all-BFP4 DRAM-sharded re-run. The **O matmul at 38.7% DRAM (<40%)** and QKV at 42.2% are the weakest rows and hint a DRAM-sharded/alternate config was never retried at final precision. **Mitigating:** DRAM 16/8-core variants hit a hard L1 CB limit (2.35 MB > 1.57 MB), so lower-core DRAM-sharded is infeasible regardless — the residual risk is a same-core (32) all-BFP4 DRAM-sharded trial for QKV/O.

## Notable
The standout is the **TP=4 / 2×2-mesh → dense single-device collapse** (original all-reduces folded into dense full-weight matmuls; multi-device items correctly N/A). GQA 8:1; packed QKV; split gate/up (full-phase packing exceeds L1 by 690 KB/core). Two split `paged_update_cache` retained after a fused rewrite failed the validator + tripped a watcher NoC fault. AutoDebug fixed a DRAM-width-sharded prefill corruption (8-bank shards disagreeing with the 11-column `per_core_N`).
