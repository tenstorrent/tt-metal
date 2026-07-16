# Qwen2.5-Coder-32B-Instruct — optimized decoder analysis

**Verdict:** The Pattern-B **validation case**. It's the one dense model **forced to stay BFP8** (BFP4/LoFi all failed the PCC bar), and at that final precision it **correctly re-decided the strategy and 1D-mcast won** (DRAM-sharded 2.29 vs 1D 1.94 ms) — with a **batch-aware default** (1D-mcast for batch-32, DRAM-sharded for batch-1). Pattern A present in the capture but avoided at runtime.

_Analyzed read-only from committed artifacts on `mvasiljevic/model/qwen-qwen2.5-coder-32b-instruct`._

## Model & shape
| | |
| --- | --- |
| hidden | 5120 |
| heads / kv | 40 / 8 (GQA 5:1), head_dim 128 |
| intermediate | 27648 (packed gate/up N=55296) |
| layers | 64 (dense) · rep layer 32 |
| batch / seq | 32 (+batch 1) / single-token decode, prefill 17 |
| device | Blackhole P300 |
| KV cache | BFP8 |
| special | **QKV bias** (Qwen2.5), separate BF16 add; RMSNorm HiFi2+FP32; no QK-norm |

## Headline decode perf
| Path | Traced decode ms (batch 32) |
| --- | ---: |
| functional BF16 | 82.374 |
| **optimized (advisor 1D-mcast, BFP8/HiFi2)** | **1.941** |
| speedup | **42.4×** |

Beats the strongest earlier correct candidate (2.291 ms packed gate/up DRAM-40c) by 15.3%. Batch-1 default (DRAM-sharded 40c): 2.903 → 2.150. Harness: batch 32, 50 warmed reps, traced replay; PERF_DECODE window = 44 ops, 1.908 ms device.

## Precision policy shipped
**BFP8 weights / HiFi2** for all four matmul groups; KV cache **BFP8**; activations BF16; RMSNorm HiFi2 + FP32 accum. **All BFP4/LoFi families rejected on real-weight PCC** (BFP8/LoFi 0.9987/0.9986 < bar; BFP4 gate/up 0.9957; BFP4 MLP 0.9935; BFP4 attn 0.9948). Only BFP8/HiFi2 clears 0.99878/0.99892 — the only dense model here that is **not** on BFP4.

## Decode matmul strategy shipped
**Batch-aware.** Batch-32 default `advisor_packed_bfp8_hifi2_1d` = **1D-mcast, interleaved weights** (`MatmulMultiCoreReuseMultiCast1DProgramConfig`) at the advisor final-IR fields: QKV 11×7/75c, O 11×8/80c, packed gate/up 11×10/108c, down 87c→80c (all `in0_block_w=2`). Batch-1 default = **DRAM-sharded** weights, gate/up 40c. No LM-head in scope.

## Shard-advise: applied vs rejected
- Run this pass on the rewritten packed dense decode graph (28 ops, 25 choices, 1 spill). **Applied exactly:** 11-core block-sharded RMSNorm; QKV 11×7/75c; O 11×8/80c; packed gate/up 11×10/108c; multiply on 108c + down input reshard to 87c. The **packed gate/up spill to interleaved L1 before the static-split consumers is critical** — omitting it dropped synthetic decode PCC to 0.000351; applying it restored 0.999839.
- **Unfixable op:** `ttnn.nlp_concat_heads_decode` ("Input tensor must be sharded") — dedicated head helper (DRAM-interleaved) retained on whole-path evidence.
- **Rejected (with evidence):** "keep every advisor field without local search" — block 4/8/16 + non-power trials, DRAM-sharded alternatives, and precision candidates all measured; block 2 is the legal maximum (input shard is exactly 2 tiles). First advisor split chain was correct but 11.748 ms → advisor rerun on the packed graph.

## Per-op decode matmul table
PERF_DECODE window (device 1.908 ms, 44 ops). All `HiFi2 BF16×BFP8→BF16`, 1D-mcast.

| Role (M×K×N) | Device µs | % | Cores | DRAM GB/s | DRAM % |
| --- | ---: | ---: | ---: | ---: | ---: |
| QKV 32×5120×7168 | 110 | 5.7 | 75 | 333 | 65.0% |
| O 32×5120×5120 | 102 | 5.3 | 80 | 258 | 50.3% |
| packed gate/up 32×5120×55296 | 821 | 42.4 | 108 | 345 | 67.4% |
| down 32×27648×5120 | 583 | 30.1 | 87→80 | 243 | 47.4% |

Four matmuls = **83.5%** of decode; overall 256 GB/s / **49.9%** roofline (higher than the BFP4 models — BFP8 streams 2× the bytes so it stays DRAM-bound at good util). SDPA-decode 50 µs/64c; fused PagedUpdateCache 21 µs. **No large Copy or ReshapeView row**; the only sizable movement is two attention head Transposes (29 + 42 µs = 71 µs / 3.7%).

## Pattern A — segment-boundary reverts
**Present in capture, avoided at runtime.** report.txt has the `reshape → return: l1/interleaved/10x11 → dram/interleaved (output revert)`. But the shipped path keeps the residual **L1-resident/width-sharded through attention and MLP** and applies the revert **only at the model-visible decoder return** ("no immediate old-contract restore inside attention/MLP"); residual adds are ~1.3 µs on-device. No large boundary Copy/ReshapeView survives — only the once-per-layer cheap `to_memory_config` return + ~3.7% attention head permutes (inherent to the concat-heads contract, not the DRAM revert).

## Pattern B — strategy × precision
**Done right — the validation case.** Precision was locked to **BFP8** first (BFP4 rejected on PCC), then DRAM-sharded vs advisor 1D-mcast were compared **both at BFP8/HiFi2**: DRAM-sharded 32c 2.324, DRAM-sharded/packed 40c 2.291, **advisor 1D packed 1.941** — 1D wins for batch-32. This is the flip side of Llama-8B: at **BFP8** (2× the weight bytes of BFP4) 1D-mcast's utilization advantage holds, so 1D is genuinely fastest; the batch-1 path correctly switches to DRAM-sharded. No dominant matmul below 40% DRAM (lowest: down 47%, O 50%); the two "SLOW" rows triggered an exhaustive block/shard search (block 2 is the legal maximum). No untried config.

## Notable
QKV **bias** kept as a separate BF16 add (folding into the BFP8 linear damaged Q/K/V PCC; separate add → 0.999995). Packed gate/up + fused SiLU-multiply beats split whole-layer. Fused `paged_fused_update_cache` shipped. **Batch-aware strategy default** (1D for batch 32, DRAM-sharded 40c for batch 1–31; advisor geometry guarded to batch 32). No embedding/LM-head (out of scope).
