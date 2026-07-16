# GPT-OSS-20B (MoE) — optimized decoder analysis

**Verdict:** The interesting outlier. MoE with `sparse_matmul` experts (**outside shard-advise coverage**). Ships **1D-mcast for QKV/O at only ~12% DRAM util** — but that is **correctness-bounded** (DRAM-sharded QKV/O was tried and rejected on sliding-window PCC, not perf). **Pattern A does *not* bite** (0 Copy ops, 1 reshard/replay). Pattern B done right.

_Analyzed read-only from committed artifacts on `mvasiljevic/model/openai-gpt-oss-20b`._

## Model & shape
| | |
| --- | --- |
| hidden | 2880 |
| heads / kv | 64 / 8 (GQA), head_dim 64 |
| experts | 32 total, **top_k=4 active** (`sparse_matmul`) |
| expert intermediate | 2880 (gate/up packed to 2×2880) |
| layers | 24 (rep: one per attn kind — sliding + full) |
| batch / seq | **1** (hard contract) / 17 |
| device | Blackhole P300 |
| KV cache | **BF16** (BFP8 rejected: sliding pos-129 PCC 0.863) |
| attention | alternating sliding-window (128) + full, **attention sinks** |

## Headline decode perf
| Path | Warmed prefill | Traced decode | vs functional |
| --- | ---: | ---: | ---: |
| functional | 7.709 | 6.139 | 1.0× |
| **final selected** | **3.875** | **0.928** | **6.61×** decode |
| full dense advisor A/B | 7.727 | 5.910 | rejected (6.4× slower) |

Harness: `test_optimized_decoder_perf`, 10 prefill / 100 replay; 0 host ops.

## Precision policy shipped
- Attention (QKV/O): **BF16, HiFi4** (BFP4/global-LoFi rejected on PCC).
- Router: **FP32 compute** (norm BF16 → FP32 linear → BF16 for topk/softmax/scatter).
- Experts (gate/up/down): **BFP8 weights & intermediates, LoFi** (BFP4/LoFi failed the 0.99 gate).
- KV cache: **BF16** linear paged.

## Decode matmul strategy shipped
| Role | Shape | Strategy |
| --- | --- | --- |
| QKV | 32×2880×5120 | **1D-mcast** `@11x8`, 80 cores, L1 width-sharded |
| O | 32×4096×2880 | **1D-mcast** `@11x9`, 90 cores |
| Router | 32×2880×32 | 1D-mcast `@1x1`, 1 core, DRAM |
| Experts gate/up/down | active 4/32 ×32×2880×2880 | **`ttnn.sparse_matmul`**, 90 cores, block 45, **nnz unset** (nnz=4 unsafe — routing weights can flush to 0) |

## Shard-advise: applied vs rejected
- Run this pass; captured the packed-attention + **dense**-MLP graph (46 ops). Applied: 10-core block-sharded norm; 80-core QKV (11x8); 90-core O/residual (11x9); DRAM cache + SDPA retained (decode 0.984 → 0.928 ms).
- **Rejected (with evidence):** full dense-MoE advisor chain — PCC-correct (0.9996) but **7.73 / 5.91 ms** → performance reject. First-bad tensor localized to router scatter.
- **Sparse limitation:** the advisor cannot trace `ttnn.sparse_matmul`, so the shipped routed experts have **no shard-advisor coverage** (tuned manually) and are 6.4× faster than the dense chain it recommended. No "unfixable op"; 0 spills.

## Per-op decode matmul table
Per replay (from committed decode_perf_report):

| Role | Device µs | Cores | DRAM % | Fidelity/dtype |
| --- | ---: | ---: | ---: | --- |
| QKV 32×2880×5120 | 77 | 80 | **12.2%** | HiFi4 BF16→BF16 (1D, L1) |
| O 32×4096×2880 | 62 | 90 | **12.2%** | HiFi4 BF16→BF16 (1D, L1) |
| Router 32×2880×32 | 23 | 1 | 0.25% | HiFi4 FP32×BF16→FP32 |
| SparseMatmul gate | 117 | 90 | 18.2% | LoFi BF16×BFP8→BFP8 |
| SparseMatmul up | 116 | 90 | 18.3% | LoFi BF16×BFP8→BFP8 |
| SparseMatmul down | 114 | 90 | 18.6% | LoFi BF16×BFP8→BFP8 |

**0 Copy ops; 1 reshard/replay** (~1.8 µs). One 46 µs single-core post-attn LayerNorm stands out. Decode ≈ 0.905 ms/replay, ~89 ops/replay; sparse 38%, dense matmul 18% of time; 33% aggregate roofline.

## Pattern A — segment-boundary reverts
**Present in the capture, does NOT manifest.** `final_ir.mlir` ends with `add → return: l1/width_sharded → dram/interleaved (output revert)` and entry DRAM→block conversions — but the shipped runtime shows **0 Copy ops and 1 reshard/replay** (sharded↔interleaved reverts ~1–3 µs). The DRAM boundary here is the genuine decoder-layer output/next-layer contract, not an artifact. This is the model where Pattern A's "expensive per-step Copy" is **not observed** — a useful counter-example.

## Pattern B — strategy × precision
**Not stale.** Sparse grid/block and SDPA K were swept at the final BFP8/LoFi precision. **Low-DRAM% dominant rows present** (QKV/O at 12.2%, sparse experts ~18%), and the advisor/report explicitly advises "try DRAM-sharded" on QKV/O. But DRAM-sharded QKV/O **was tried and rejected on correctness** (sliding-window pos-130 PCC 0.913 with the required bias-after-matmul repair), not perf — so the low util is a **correctness-bounded ceiling**, not a missed config. The router (0.25% DRAM, 1 core, FP32, SLOW) and the 46 µs single-core LayerNorm are the clearest small remaining rows.

## Notable
MoE routing FP32→topk→softmax→scatter; `nnz` intentionally unset for safety (Blackhole zero-flush). Attention sinks preserved; native sliding-window kernel rejected on PCC → device-built explicit 128-token mask + BF16 dest accumulation. RoPE AutoFix (host scalar position absent from program hash → device-gathered cos/sin). Context held at 21,248 tokens (DRAM-limited). No shard-advise coverage of the sparse experts is the structural gap for MoE.
