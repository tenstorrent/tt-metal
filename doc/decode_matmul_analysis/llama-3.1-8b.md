# Llama-3.1-8B-Instruct — optimized decoder analysis (on-device verified)

**Verdict:** The reference case, and the **sole Pattern-B miss** across the fleet. Shipped **advisor 1D-mcast** matmuls, but at the shipped all-BFP4 precision a plain **DRAM-sharded** path is **~2% faster with identical PCC** — the strategy was frozen at an earlier precision and never re-decided. The advisor's norm/residual chain was correctly rejected (Pattern A: a ~33 µs boundary copy).

This model was **re-run on a Blackhole P300** with real layer-16 weights (the others are analyzed from committed artifacts). Full detail with side-by-side `tt-perf-report`:

- [`exact_vs_mixed_decode_perf.html`](../../models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/shard_advise/exact_vs_mixed_decode_perf.html) — advisor exact residual chain vs shipped mixed 32-way (Pattern A).
- [`dram_sharded_vs_advisor_matmul.html`](../../models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/shard_advise/dram_sharded_vs_advisor_matmul.html) — DRAM-sharded vs advisor 1D at final precision (Pattern B).

## Model & shape
hidden 4096 · 32 q / 8 kv (GQA 4:1), head_dim 128 · intermediate 14336 · 32 layers · rep layer 16 · batch 32 / seq 18 · Blackhole P300 · BFP8 cache.

## Headline decode perf (BFP8 cache, 10 prefill / 100 replay)
| Path | Traced decode ms |
| --- | ---: |
| functional BF16 | 36.95 |
| shipped optimized (advisor_1d) | 0.710 |
| **DRAM-sharded, 32c (untried at final precision)** | **0.6965** |
| advisor exact residual chain (rejected) | 0.744 |

Functional→optimized ≈ 49× traced decode.

## Shard-advise: applied vs rejected
- **Applied (verified advisor-caused):** the 5 matmul 1D program configs (grids 11×9/11×6, `in0_block_w`, `per_core_N`, output subblock 5) and projection output-shard geometry — drives all 5 dominant matmuls.
- **Coincides / independent (not advisor):** projection outputs being L1 width-sharded (baseline already did), the 32-way residual grid, and BFP4/LoFi precision (separate sweep; advisor doesn't do dtype).
- **Rejected:** the exact norm/residual chain — PCC-fine but ~5% slower.
- **Unfixable op:** `ttnn.nlp_concat_heads_decode`.

## Per-op decode matmul table (advisor_1d shipped)
| Role | Device µs | Cores | DRAM % |
| --- | ---: | ---: | ---: |
| QKV 32×4096×6144 | 91 | 96 | **27%** |
| O 32×4096×4096 | 33 | 64 | 50% |
| gate 32×4096×14336 | 107 | 90 | 54% |
| up 32×4096×14336 | 108 | 90 | 53% |
| down 32×14336×4096 | 111 | 64 | 52% |

DRAM-sharded QKV instead hits **46% / 53 µs** (vs 27% / 91 µs) — the driver of the ~14 µs win.

## Pattern A — segment-boundary reverts
**Present, and bites in the rejected variant.** The advisor's exact chain holds the residual L1-interleaved (110 cores) and reverts to DRAM at the boundary → a ~33 µs 110-core `Copy` per step. The shipped mixed 32-way chain keeps the residual width-sharded and avoids it (metadata-only reshapes).

## Pattern B — strategy × precision
**Stale (the miss).** advisor_1d beat DRAM-sharded at *original* precision, was "retained as search base" (`work_log:131`), and all BFP4 reductions were layered only onto it. DRAM-sharded was never re-measured at the shipped BFP4 precision, where it wins. QKV at 27% DRAM was the tell. Filed as skill fixes (OPT-014/OPT-015) on `mvasiljevic/optimize-skill-notes`.
