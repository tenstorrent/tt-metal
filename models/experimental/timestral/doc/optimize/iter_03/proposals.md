# Round 4 proposals (from `iter_03/analysis.md`, mapped to the optimize skill)

Current best (round 3, paper / accuracy): **540.7 ms** steady wall and **470.2 ms** device (round 0: 1445.3 / 1307.2; round 2: 679.6 / 530.9).
- Matmuls take 60.1 % of device time at a weighted 43.6 % FLOPs (110-core basis). Non-matmul ops take 39.9 %, and their own DRAM floor is 146.7 ms, so the "non-matmul < 15 %" exit criterion cannot be met with this op set.
- The 30 pad tokens per row (31 % of all work) are the largest structural cost left. They exist only because 2 decoder tokens push 64 context tokens over a tile boundary.
- From round 3 on, `device_s` is measured after a 0.3 s idle (back-to-back forwards slow the device by up to ≈ 35 ms; see `candidates.md`).

Savings overlap and are not additive. Estimates are full-model device ms at the paper shape unless marked otherwise. The gates are unchanged:
- the PCC suite (17 tests);
- paper normalized PCC ≥ 0.999;
- a paper median improvement of ≥ 2 % (device time is the primary signal);
- no regression > 3 % on `long_context`, `multivariate` or the `performance()` precision.

| # | Proposal | Finding | Est. savings | Confidence | Risk / complexity |
|---|---|---|---|---|---|
| R4-1 | Context / decoder split stream: context rows run unpadded (S = 64), the 2 decoder tokens per row run as a 2048-row stream that attends to the context K / V | F1 | 60–100 ms (+ ≈ 90 ms multivariate; long_context keeps one stream) | low | high (gate on a decoder-attention micro-benchmark ≤ 2 ms per layer) |
| R4-2 | Pack wg into the split QKV matmul (`[Wq \| Wk \| Wv \| Wg]`, 4 chunks; `[Wv \| Wg]` on the univariate V-path), sigmoid applied in the gate MUL | F3 | 11–19 ms | medium-low | medium (sweep the 4-chunk split first) |
| R4-3 | Last layer: q, SDPA and concat only for the decoder queries; K / V from a `[Wk \| Wv]` split on all rows | F6 | 3.5–4.5 ms | medium-low | medium (also the cheap test of R4-1's SDPA risks) |
| R4-4 | Subblock search for the split QKV (8x1 measured −3.5 %), the minimal 8/8/8 matmuls (wo, w_down, wv) and the SwiGLU | F7, F9 | 2–6 ms | medium | low (sweep scripts exist) |
| R4-5 | Embed cleanup: main + skip ADD folded into the out matmul's epilogue; MUL keep + ADD query as one op | F8 | 1.5–2.5 ms | medium | low |
| R4-6 | Host: cache `unit_stats` and the repeated cos / sin per shape and positions key | F12 | 2–4 ms wall | low-medium | low |
| R4-7 | [ttnn] Batch-capable fused head-split op (per-batch head stride, cos / sin indexed mod S), so SDPA writes the concatenated heads and `nlp_concat_heads` goes away | F2 | 15–17 ms | medium | medium (shared ttnn op outside the timestral tree; its unit tests) |
| R4-8 | [ttnn] Deeper reader pipeline in the fused head-split op and the RMSNorm reader | F4 | 10–18 ms | low-medium | medium (shared kernels; zone profile first) |
| R4-9 | Performance precision only: bf8 SwiGLU output (halves the hidden write and the w_down read) | F10 | 9–14 ms (performance only) | low-medium | low (PCC margin 0.99970) |
| R4-10 | Multivariate only: variate attention as a free (B, V·S, D) view with a same-slot mask (no permutes, no 4x padded SDPA); profile first | F11 | 150–250 ms on multivariate (guard shape) | low | high |

Dropped or deferred:
- F5 (fuse the gate MUL): the kernel version needs SDPA or `nlp_concat_heads` changes, and the model-only version (≈ 2.5 ms) conflicts with R4-2.
- F9's silu approximation mode: unmeasured; it is folded into R4-4's SwiGLU sweep as one more axis with a PCC check.
- Trace capture: host issue (≈ 46 ms per forward) is hidden behind 470 ms of device time.

## R4-1. Context / decoder split stream

- **Hypothesis:** the forecast time mask never lets a context token read a decoder token (`build_forecast_time_mask`: `enc_dec` all zero), and variate attention is slot-local. Context tokens therefore never depend on decoder tokens. Run the context as 1024 rows x 64 tokens (65536 rows, no padding) and the decoder as 2048 dense rows. In each time attention, the decoder queries attend to [their row's context K / V | the decoder K / V] through free views of the head-major tensors and one masked SDPA.
- **Expected:** `time` layer 18.3 → 14.2–15.7 ms, `time_variate` 23.8 → 18.0–19.5 ms, embed −3 ms: ≈ −60 to −100 ms. Multivariate gains ≈ 90 ms more. long_context has no padding and would lose ≈ 10 ms, so it keeps the single stream (gate on a pad fraction ≥ 15 %).
- **Optimize-skill rules:** "Optimize the operation topology before fine-tuning individual op configs"; "Keep logical ... batch separate from tile padding" (OPT-005); "Use SDPA ... when the target model fits their contracts".
- **Risks:** SDPA with Sq = 32 ≠ Sk = 1056 on a packed view with a block mask is unmeasured; K / V concat costs; missing patches need per-row key validity on both streams; ≈ 12 more ops per layer.
- **Experiment:** micro-benchmark the decoder attention chain (2 concats + masked SDPA + views) at the paper shape; proceed if ≤ 2 ms per layer. R4-3 tests the same mechanism on one layer.

## R4-2. Pack wg into the split QKV matmul

- **Hypothesis:** wg + sigmoid runs on 80 cores at 28 % FLOPs (2.30x its lower bound), 28 x 1363 µs = 38.2 ms. `minimal_matmul_split(xn, [Wq | Wk | Wv | Wg], chunks=4)` keeps today's 24-tile chunk width. The sigmoid moves into the gate MUL (`input_tensor_b_activations`, measured free in R2-5). On the univariate V-path, `[Wv | Wg]` with 2 chunks.
- **Expected:** QKVG ≈ 3.3–3.6 ms against 2616 + 1363 µs (−0.4 to −0.7 ms x 23), V-path −0.55 ms x 5: −11 to −19 ms.
- **Optimize-skill rule:** "When several matmuls consume the same input activation, first ask whether they should be one packed/fused projection ... Compare the packed candidate against a well-tuned separate candidate."
- **Experiment:** extend `tests/perf/sweep_qkv_split.py` to N = 96 tiles / 4 chunks and to the 2-chunk V-path; integrate the winner; PCC suite and benchmark.

## R4-3. Last layer: decoder-only queries

- **Hypothesis:** after the last layer's SDPA only 2048 of 98304 query rows matter, but q, its post op, SDPA and the concat run on every row, and three tensors are cut. Cut x first (the residual needs it anyway), compute q on 2048 rows, K / V from `[Wk | Wv]` on all rows, and run a masked SDPA with 32 decoder queries per 16-row block against its 1536 keys.
- **Expected:** 10.16 → 6.1–6.6 ms: −3.5 to −4.0 ms.
- **Optimize-skill rule:** "Count material matmuls ... in the measured path".
- **Experiment:** implement in `TtAttention` behind the existing `decoder` argument; the last-block PCC test covers it; benchmark.

## R4-4. Subblock search

- **Hypothesis:** the QKV split ships subblock 2x4, while the sweep measured 8x1 at −3.5 % (2.693 vs 2.790 ms). The minimal 8/8/8 configs (wo, w_down, wv: 1x8) and the SwiGLU (4x2) were never swept for subblock shape. The silu approximation mode is one more SwiGLU axis.
- **Expected:** ≈ 2 ms (QKV) plus 0–4 ms elsewhere.
- **Optimize-skill rules:** "Output subblock size should usually be at least 2x1 or 1x2 when legal ... try larger valid sizes"; "local program-config search is required".
- **Experiment:** rerun the split sweep at 3 repeats; a subblock sweep for the addcmul and SwiGLU configs; keep only device-confirmed wins.

## R4-5. Embed cleanup

- **Hypothesis:** the embed has an ADD of main + skip (1006 µs) and a MUL keep + ADD query pair (1011 + 1000 µs). The first folds into the out matmul as an addcmul with the skip as the residual. The pair becomes one `addcmul(add, embedded, keep)`.
- **Expected:** −1.5 to −2.5 ms.
- **Experiment:** implement; the patch-embedding and predict_block PCC tests; benchmark.

## R4-6. Host: cache the per-shape upload constants

- **Hypothesis:** `unit_stats` (`ttnn.full` [1, 1, 98304, 32]) and the `ttnn.repeat` of shared cos / sin are rebuilt on every call, though they depend only on the shape and positions.
- **Expected:** 2–4 ms wall; no device change.
- **Experiment:** cache per (rows, padded_seq) and per positions key; benchmark.

## R4-7 to R4-10

See `analysis.md` §7 (F2, F4, F10, F11) for evidence and mechanisms. R4-7 and R4-8 change shared ttnn kernels outside `models/experimental/timestral` and need those ops' unit tests. R4-10 needs a reduced multivariate profile before any change.
