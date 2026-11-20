<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Hy4-preview decode on one Galaxy: where it stands against the target

Target (tenstorrent/tt-metal#65884, set 2026-11-19): one Blackhole Galaxy, one SC1 engine, 8 users with 1M tokens
of context each, 75 output tokens per second per user. That is one decode step for the batch of 8 in 13.3 ms:
**0.171 ms per layer** over 78 layers, everything included.

Measured on one Blackhole Galaxy on 2026-11-19 with `tests/test_perf_decode.py` (8x4 mesh, torus fabric, the mesh
`tt/prefill_bridge.device_params()` opens, plus a 128 MB trace region; checkpoint weights from the decode weight
cache, BFP8 weights, BFP4 routed experts, BFP8 decode cache). Raw numbers: `generated/hy4_preview/logs/perf/*.json`,
logs `generated/hy4_preview/logs/perf_*.log`.

Every number below is marked: **(M)** measured on the device, **(A)** arithmetic on measured numbers (for example a
step of 78 layers from the three kinds of layer), **(E)** an estimate, with how it was made.

## 1. Where it stands

At the target shape, 8 users x 1M context:

| | ms per layer (mean over the 78) | step of 78 layers | tokens/s per user | x the budget |
|---|---|---|---|---|
| as the model runs today, untraced (A from M) | 89.1 | 6.96 s | 0.144 | 521 |
| the same ops replayed as a trace (A from M) | 42.1 | 3.28 s | 0.305 | 246 |
| target | 0.171 | 13.3 ms | 75 | 1 |

* Untraced, the host is the limit (20 to 30 ms per layer for one user at any context, about 95 us per ttnn call).
  Traced, one user at 4K is 3.1 to 4.5 ms per layer (M), 18 to 26 times the budget: a layer is 200 to 315 ttnn
  calls for one user, and a call costs 2 to 23 us on the device even when it computes next to nothing (section 4).
* At 1M with 8 users the device itself is the limit, traced or not: 52% of a traced step is the top-2048 selection
  of the 21 full-indexer layers and 26% is the row gather and collect, both run once per user (section 3.3).
* For a single user at 1M, 11 of the 24 stages of a full MoE layer are each above the whole 0.171 ms. The largest:
  the selection (10.3 ms per full layer), the routed experts (0.85 ms, all 8 experts of every chip for every
  token) and the four iHC mixes (0.73 ms together). For a batch of 8 the attention stages count 8 times,
  because attention takes one user per call.
* Tracing works on every kind of layer, bit-equal to the untraced forward, and follows the step's inputs when
  they are rewritten in place (section 5).
* Memory fits only with one copy of the cache and one set of weights (section 6): 15.14 + 13.27 GB of 34.18 GB per
  chip.
* The plan (section 7) takes a step from 6.96 s to an estimated 65 to 135 ms (7 to 15 tokens/s per user) with
  tracing, users as a batch dimension, fused ops and the new ops of section 8. The last factor of 5 to 10 is
  the cost of the collectives and of the expert weights themselves; nothing measured here says it is within
  reach as the step is structured.

## 2. How the numbers were taken

`tests/test_perf_decode.py`, one job per case through `tools/run_device_tests.sh`. A case builds layers 0, 1 and 2
(the three kinds of block: dense FFN + full indexer, 1 of 78; MoE + full indexer, 20 of 78; MoE + shared indexer,
57 of 78) with the embedding and the head, over a zero-initialised cache of `users` x `context` positions that
holds those three layers. Every user steps at the end of its cache, so a full layer scores the whole stripe. The
program cache is warm in every number.

| clock | what it is | tool |
|---|---|---|
| traced, per stage | the stage captured as its own trace and replayed 20 times back to back, one synchronize at the end: the stage's time on the device with no host work per op | `ttnn.begin_trace_capture` / `execute_trace` |
| traced, whole layer | `TtHy4DecoderLayer.forward` as the model calls it (attention looping over the users), captured and replayed the same way; output compared with the untraced forward | the same |
| untraced, staged | host time from a stage's first ttnn call to the end of a synchronize, an idle synchronize (0.09 to 0.13 ms) taken off | `time.perf_counter` + `ttnn.synchronize_device` |
| untraced, whole layer | the forward issued 8 times back to back, one synchronize at the end | the same |
| ttnn calls | operations a stage issues, counted on the host | a counting wrapper on the ttnn namespace |

Stages are the modules' own methods in the order the forward calls them. The staged chain is checked against the
layer's forward in every case (max difference 0.0 on all 18 layers), and the sum of the traced stages is 1 to 7%
above the traced whole layer.

The attention sublayer takes one user per call (`tt/attention_decode.py`). Its stages are measured for one user of
the cache; a batch of 8 is 8 such calls, which is what the whole-layer numbers of the 8-user cases run. Every
other stage takes the users as rows and is measured with all of them.

Tools that gave no number here:
* The realtime program profiler (`ttnn.device.RegisterProgramRealtimeProfilerCallback`, kernel time per program
  per chip, what the prefill stack's perf tests use) is not active on this host:
  `ttnn.device.IsProgramRealtimeProfilerActive()` is False. The 32 chips sit in identity IOMMU domains (kernel
  command line `iommu=pt`), and the profiler's device-to-host socket needs a translated domain
  (`tt_metal/distributed/realtime_profiler_manager.cpp`).
* The Tracy device profiler (`TT_METAL_DEVICE_PROFILER=1`) rebuilds every kernel, the fabric routers included. It
  was not tried: a fabric that fails to start takes this board down until it is reset by hand.
So there is no kernel time per op. What stands in for it is the cost of a traced program that has next to nothing
to compute (section 4).

## 3. Results

### 3.1 Layers and steps

Milliseconds per layer for the whole batch. "ttnn calls" is what the host issues for the layer (M).

| context, users | kind of layer | untraced | traced | untraced / traced | traced / budget | ttnn calls |
|---|---|---|---|---|---|---|
| 4K, 1 user | dense FFN + full indexer | 25.30 | 3.378 | 7.5 | 20 | 278 |
| 4K, 1 user | MoE + full indexer | 27.97 | 4.466 | 6.3 | 26 | 304 |
| 4K, 1 user | MoE + shared indexer | 20.38 | 3.066 | 6.6 | 18 | 203 |
| 128K, 1 user | dense FFN + full indexer | 26.86 | 9.911 | 2.7 | 58 | 289 |
| 128K, 1 user | MoE + full indexer | 29.60 | 11.001 | 2.7 | 64 | 315 |
| 128K, 1 user | MoE + shared indexer | 20.35 | 3.077 | 6.6 | 18 | 203 |
| 1M, 1 user | dense FFN + full indexer | 26.70 | 12.755 | 2.1 | 75 | 289 |
| 1M, 1 user | MoE + full indexer | 29.55 | 13.842 | 2.1 | 81 | 315 |
| 1M, 1 user | MoE + shared indexer | 20.18 | 3.212 | 6.3 | 19 | 203 |
| 4K, 8 users | dense FFN + full indexer | 124.92 | 20.422 | 6.1 | 119 | 1505 |
| 4K, 8 users | MoE + full indexer | 127.98 | 21.503 | 6.0 | 126 | 1531 |
| 4K, 8 users | MoE + shared indexer | 66.84 | 10.158 | 6.6 | 59 | 715 |
| 128K, 8 users | dense FFN + full indexer | 138.88 | 73.681 | 1.9 | 431 | 1593 |
| 128K, 8 users | MoE + full indexer | 140.62 | 74.738 | 1.9 | 437 | 1619 |
| 128K, 8 users | MoE + shared indexer | 67.31 | 11.269 | 6.0 | 66 | 715 |
| 1M, 8 users | dense FFN + full indexer | 141.73 | 103.391 | 1.4 | 605 | 1593 |
| 1M, 8 users | MoE + full indexer | 144.69 | 104.439 | 1.4 | 611 | 1619 |
| 1M, 8 users | MoE + shared indexer | 68.72 | 19.098 | 3.6 | 112 | 715 |

A step of 78 layers (A: 1 x dense + 20 x MoE-full + 57 x MoE-shared, plus the stages around the layers of 3.4;
the host uploads of a step, 1.2 to 8.6 ms, are not included):

| context, users | untraced: mean ms per layer | step of 78 layers, ms | tokens/s per user | x budget | traced: mean ms per layer | step of 78 layers, ms | tokens/s per user | x budget |
|---|---|---|---|---|---|---|---|---|
| 4K, 1 user | 22.39 | 1756 | 0.569 | 131 | 3.429 | 268.8 | 3.720 | 20 |
| 128K, 1 user | 22.81 | 1788 | 0.559 | 133 | 5.196 | 406.7 | 2.459 | 30 |
| 1M, 1 user | 22.67 | 1777 | 0.563 | 133 | 6.060 | 474.1 | 2.109 | 35 |
| 4K, 8 users | 83.26 | 6504 | 0.154 | 487 | 13.198 | 1030.8 | 0.970 | 77 |
| 128K, 8 users | 87.03 | 6798 | 0.147 | 509 | 28.344 | 2212.2 | 0.452 | 166 |
| 1M, 8 users | 89.14 | 6962 | 0.144 | 521 | 42.061 | 3282.2 | 0.305 | 246 |

The untraced one-user numbers reproduce the full-model runs (1.75 to 1.82 s per step measured over 78 real
layers).

### 3.2 Stages, traced

Milliseconds on the device (M), stage by stage, from layer 1 (MoE + full indexer); the two dense-FFN rows are
layer 0's. Layer 2 (shared indexer) has the same stages without the five indexer rows, with the same times.
"takes: one user" marks the attention stages, where a batch of 8 pays the number 8 times; the last column is the
stage's cost for the batch at 1M x 8 users over the 0.171 ms a whole layer has. ttnn calls are for one user.

| stage | takes | ttnn calls | 4K, 1 user | 128K, 1 user | 1M, 1 user | 4K, 8 users | 128K, 8 users | 1M, 8 users | untraced, 1M, 8 users | x budget at 1M, 8 users |
|---|---|---|---|---|---|---|---|---|---|---|
| iHC pre-mix (attention) | all users | 29 | 0.294 | 0.294 | 0.294 | 0.303 | 0.303 | 0.303 | 3.494 | 1.8 |
| norm (attention) | all users | 9 | 0.095 | 0.095 | 0.095 | 0.097 | 0.097 | 0.098 | 1.091 | 0.6 |
| latent projections (q_a, kv_a, norms, RoPE) | one user | 13 | 0.311 | 0.311 | 0.311 | 0.311 | 0.312 | 0.311 | 1.355 | 14.6 |
| indexer projections (key, queries, weights) | one user | 47 | 0.423 | 0.425 | 0.425 | 0.424 | 0.425 | 0.425 | 4.206 | 19.9 |
| cache append | one user | 6 | 0.020 | 0.020 | 0.020 | 0.057 | 0.058 | 0.058 | 0.611 | 2.7 |
| indexer scoring | one user | 11 | 0.037 | 0.070 | 0.234 | 0.040 | 0.075 | 0.249 | 0.749 | 11.7 |
| local top-k | one user | 11 | 0.000 | 0.814 | 3.371 | 0.000 | 0.814 | 3.368 | 4.085 | 157.6 |
| merge (all-gather, top-k) | one user | 20 | 0.890 | 6.581 | 6.586 | 0.890 | 6.581 | 6.580 | 7.502 | 307.9 |
| locate | one user | 20 | 0.118 | 0.117 | 0.117 | 0.117 | 0.117 | 0.117 | 1.217 | 5.5 |
| row gather | one user | 4 | 0.081 | 0.102 | 0.228 | 0.085 | 0.227 | 1.207 | 1.216 | 56.5 |
| collect | one user | 2 | 0.185 | 0.185 | 0.185 | 0.184 | 0.185 | 0.184 | 0.994 | 8.6 |
| query and gate projections (q_b, gate, RoPE) | one user | 16 | 0.119 | 0.119 | 0.118 | 0.118 | 0.119 | 0.118 | 1.558 | 5.5 |
| attention core | one user | 27 | 0.244 | 0.244 | 0.244 | 0.243 | 0.244 | 0.244 | 2.915 | 11.4 |
| output projection (gate, o_proj) | one user | 7 | 0.093 | 0.094 | 0.093 | 0.092 | 0.093 | 0.092 | 0.862 | 4.3 |
| iHC post-mix (attention) | all users | 10 | 0.073 | 0.074 | 0.073 | 0.086 | 0.086 | 0.085 | 0.847 | 0.5 |
| iHC pre-mix (MLP) | all users | 29 | 0.294 | 0.294 | 0.294 | 0.303 | 0.304 | 0.302 | 3.553 | 1.8 |
| norm (MLP) | all users | 9 | 0.095 | 0.095 | 0.096 | 0.097 | 0.098 | 0.096 | 1.125 | 0.6 |
| dense FFN gate/up | all users | 6 | 0.079 | 0.079 | 0.078 | 0.079 | 0.079 | 0.079 | 0.919 | 0.5 |
| dense FFN down | all users | 3 | 0.079 | 0.079 | 0.078 | 0.079 | 0.079 | 0.079 | 0.601 | 0.5 |
| MoE input all-gather | all users | 1 | 0.026 | 0.026 | 0.032 | 0.020 | 0.027 | 0.020 | 0.231 | 0.1 |
| router (scores, top-8, weights) | all users | 11 | 0.166 | 0.166 | 0.166 | 0.171 | 0.171 | 0.170 | 0.817 | 1.0 |
| dispatch (weights to chips) | all users | 1 | 0.052 | 0.052 | 0.052 | 0.051 | 0.051 | 0.051 | 0.113 | 0.3 |
| experts | all users | 9 | 0.847 | 0.848 | 0.849 | 0.850 | 0.847 | 0.848 | 1.105 | 5.0 |
| combine | all users | 3 | 0.084 | 0.084 | 0.084 | 0.083 | 0.084 | 0.083 | 0.922 | 0.5 |
| shared expert | all users | 10 | 0.098 | 0.098 | 0.098 | 0.097 | 0.098 | 0.097 | 1.495 | 0.6 |
| iHC post-mix (MLP) | all users | 10 | 0.073 | 0.074 | 0.073 | 0.086 | 0.086 | 0.085 | 0.834 | 0.5 |

Sums (M), and how many calls and collectives a layer issues:

| context, users | kind of layer | stages that take all users | attention stages, one user | sum for the batch | whole layer traced | ttnn calls: all-user stages + one user's attention | collectives (all_reduce + all_gather), one user |
|---|---|---|---|---|---|---|---|
| 4K, 1 user | dense FFN + full indexer | 1.080 | 2.504 | 3.584 | 3.378 | 105 + 173 | 14 + 4 |
| 4K, 1 user | MoE + full indexer | 2.198 | 2.522 | 4.720 | 4.466 | 131 + 173 | 16 + 5 |
| 4K, 1 user | MoE + shared indexer | 2.190 | 1.048 | 3.238 | 3.066 | 131 + 72 | 15 + 2 |
| 128K, 1 user | dense FFN + full indexer | 1.083 | 9.059 | 10.14 | 9.911 | 105 + 184 | 14 + 4 |
| 128K, 1 user | MoE + full indexer | 2.200 | 9.083 | 11.28 | 11.00 | 131 + 184 | 16 + 5 |
| 128K, 1 user | MoE + shared indexer | 2.193 | 1.075 | 3.268 | 3.077 | 131 + 72 | 15 + 2 |
| 1M, 1 user | dense FFN + full indexer | 1.076 | 11.89 | 12.96 | 12.75 | 105 + 184 | 14 + 4 |
| 1M, 1 user | MoE + full indexer | 2.207 | 11.93 | 14.14 | 13.84 | 131 + 184 | 16 + 5 |
| 1M, 1 user | MoE + shared indexer | 2.191 | 1.200 | 3.391 | 3.212 | 131 + 72 | 15 + 2 |
| 4K, 8 users | dense FFN + full indexer | 1.129 | 2.547 | 21.51 | 20.42 | 105 + 175 | 14 + 4 |
| 4K, 8 users | MoE + full indexer | 2.243 | 2.562 | 22.74 | 21.50 | 131 + 175 | 16 + 5 |
| 4K, 8 users | MoE + shared indexer | 2.245 | 1.073 | 10.83 | 10.16 | 131 + 73 | 15 + 2 |
| 128K, 8 users | dense FFN + full indexer | 1.131 | 9.227 | 74.95 | 73.68 | 105 + 186 | 14 + 4 |
| 128K, 8 users | MoE + full indexer | 2.253 | 9.250 | 76.25 | 74.74 | 131 + 186 | 16 + 5 |
| 128K, 8 users | MoE + shared indexer | 2.245 | 1.219 | 12.00 | 11.27 | 131 + 73 | 15 + 2 |
| 1M, 8 users | dense FFN + full indexer | 1.131 | 12.93 | 104.6 | 103.4 | 105 + 186 | 14 + 4 |
| 1M, 8 users | MoE + full indexer | 2.237 | 12.95 | 105.9 | 104.4 | 131 + 186 | 16 + 5 |
| 1M, 8 users | MoE + shared indexer | 2.244 | 2.201 | 19.85 | 19.10 | 131 + 73 | 15 + 2 |

What the table says:
* Stages that take the users as rows cost the same for 8 users as for 1 (2.19 against 2.24 ms for a MoE layer):
  one row of tiles either way.
* Nothing but the selection, the row gather and the cache append depends on the context length.
* `local top-k` and `merge` are not top-k time. Measured alone (section 4), `ttnn.topk` of 2048 from 32,768 or
  65,536 scores is 0.09 to 0.19 ms, and the `ttnn.gather` that follows it (the fp32 scores and the positions of
  the picks) is 1.4 to 3.9 ms a call: two gathers per stage are 2.8 of the 3.4 ms and 5.5 of the 6.6 ms.
* `row gather` reads through the whole table of the cache (users x slots rows per chip): 0.08 ms from 128 rows,
  0.10 ms from 4,096, 0.23 ms from 32,768, 1.21 ms from 262,144 (8 users x 1M), for 2,048 rows wanted.
* `experts` is the largest stage of a one-user step outside the selection: every chip multiplies through all
  8 of its experts for every token (0.17 GB of BFP4 weights per chip, 200 GB/s), whichever were picked.

### 3.3 Where a step at 8 users x 1M goes

Stage times of 3.2 times the number of layers that run them, attention stages times 8 users (A from M; the traced
whole layers add up to 3281 ms, these stages to 3353 ms):

| part of a layer | ms per step of 78 layers | share | mean per layer, ms | x the 0.171 ms budget | ttnn calls per step |
|---|---|---|---|---|---|
| selection: scoring, local top-k, merge, locate (21 layers) | 1733 | 51.7% | 22.213 | 129.9 | 10416 |
| row gather and collect | 869 | 25.9% | 11.147 | 65.2 | 3744 |
| attention projections, core, cache append, output | 577 | 17.2% | 7.402 | 43.3 | 50376 |
| MoE and dense FFN | 98 | 2.9% | 1.257 | 7.4 | 2704 |
| iHC, four mixes per layer | 61 | 1.8% | 0.776 | 4.5 | 6084 |
| the two norms | 15 | 0.5% | 0.195 | 1.1 | 1404 |
| sum of the stages | 3353 | 100% | 42.99 | 251 | 74728 |

### 3.4 Around the layers

Once per step (M):

| stage | ttnn calls | traced, 1 user | traced, 8 users | untraced, 8 users |
|---|---|---|---|---|
| embedding (lookup, spread) | 12 | 0.110 | 0.112 | 1.719 |
| iHC expand | 1 | 0.026 | 0.027 | 0.299 |
| head readout (iHC head) | 27 | 0.284 | 0.291 | 5.276 |
| final norm (all-gather, norm) | 9 | 0.121 | 0.122 | 1.101 |
| LM head | 1 | 0.174 | 0.173 | 0.209 |
| greedy (top-k per chip, all-gather, top-1) | 7 | 0.645 | 0.647 | 1.034 |
| sum | 57 | 1.360 | 1.373 | 9.638 |

| context, users | begin_decode_step for every user + token ids, ms per step | idle synchronize, ms | whole untraced step of the 3 layers, embedding, head, greedy, ms | L1_SMALL in use per core, bytes |
|---|---|---|---|---|
| 4K, 1 user | 1.32 | 0.092 | 83.0 | 6784 |
| 128K, 1 user | 1.25 | 0.111 | 86.2 | 6784 |
| 1M, 1 user | 1.22 | 0.129 | 86.2 | 6784 |
| 4K, 8 users | 8.42 | 0.105 | 343.7 | 7680 |
| 128K, 8 users | 8.50 | 0.122 | 368.9 | 7680 |
| 1M, 8 users | 8.57 | 0.113 | 378.5 | 7680 |

The host uploads of a step (cos and sin, append indices and query position per user, the token ids) are 1.2 to
1.3 ms per user, 8.5 ms for 8 users: 64% of the 13.3 ms a whole step has. The traced stages around the layers are
1.37 ms, 10% of it; greedy alone (a top-32 per chip, two pairs of all-gathers, a top-1) is 0.65 ms.

## 4. What one op costs

`test_op_floor`: each op issued 32 times, untraced and as one trace (M, microseconds per call on the 32 chips):

| one ttnn call on the 32 chips | untraced, us | traced, us |
|---|---|---|
| add, [32 x 32] bf16 | 44 | 5.9 |
| multiply by a scalar, [32 x 32] bf16 | 42 | 5.8 |
| typecast bf16 -> fp32, [32 x 32] | 34 | 1.7 |
| slice of 16 columns, [32 x 32] bf16 | 42 | 1.7 |
| concat of two, [32 x 32] bf16 | 40 | 1.7 |
| add, [1 x 1536] fp32 | 48 | 5.9 |
| add, [8 x 1536] fp32 | 49 | 5.9 |
| matmul, [1 x 1536] fp32 x [1536 x 32] bf16 | 97 | 14.3 |
| all_reduce over mesh axis 1 (4 chips, hidden), [32 x 32] bf16 | 426 | 13.8 |
| all_reduce over mesh axis 0 (8 chips, heads), [32 x 32] bf16 | 552 | 21.3 |
| all_gather over mesh axis 1 (4 chips), [32 x 32] bf16 | 86 | 9.6 |
| all_gather over mesh axis 0 (8 chips), [32 x 32] bf16 | 207 | 12.3 |
| all_reduce over mesh axis 1 (4 chips), [8 x 1536] fp32 | 503 | 22.8 |
| all_reduce over mesh axis 1 (4 chips), [2048 x 576] bf16 | 236 | 79.0 |
| all_reduce over mesh axis 0 (8 chips), [2048 x 576] bf16 | 235 | 85.8 |

`test_selection_ops`, the two ops of the selection's top-k, each alone (M, milliseconds per call):

| one call | untraced, ms | traced, ms |
|---|---|---|
| topk 2048 of bf16 [1 x 32768] | 0.175 | 0.091 |
| topk 2048 of bf16 [1 x 32768] with uint32 labels | 258.213 | 258.097 |
| gather 2048 columns of fp32 [1 x 32768] | 1.469 | 1.420 |
| topk 2048 of bf16 [8 x 32768] | 0.252 | 0.125 |
| topk 2048 of bf16 [8 x 32768] with uint32 labels | 258.309 | 258.095 |
| gather 2048 columns of fp32 [8 x 32768] | 3.407 | 3.193 |
| topk 2048 of bf16 [1 x 65536] | 0.290 | 0.158 |
| topk 2048 of bf16 [1 x 65536] with uint32 labels | 524.940 | 524.580 |
| gather 2048 columns of fp32 [1 x 65536] | 2.776 | 2.743 |
| topk 2048 of bf16 [8 x 65536] | 0.275 | 0.190 |
| topk 2048 of bf16 [8 x 65536] with uint32 labels | 524.933 | 524.584 |
| gather 2048 columns of fp32 [8 x 65536] | 3.904 | 3.868 |

* A traced program costs about 6 us when it computes next to nothing (1.7 us for a typecast, slice or concat of one
  tile), a small matmul 14 us, a small collective 10 to 23 us. A layer issues 203 to 315 calls and 17 to 21
  collectives per user, so a traced shared layer (3.07 ms for 203 calls) is 15 us per call: the layer is made of
  op floors, not of arithmetic. 17 collectives at these floors are 0.24 to 0.39 ms: more than the budget before
  anything is computed.
* Untraced, a call costs the host 34 to 49 us, a matmul 97 us and a composite `ttnn.all_reduce` 0.24 to 0.55 ms.
* The collect's all-reduce of the 2,048 selected rows (2.4 MB) is 79 and 86 us per axis: 65 us of payload on top
  of the floor.
* `ttnn.topk` with `indices_tensor` (labels carried along with the values) and k = 2048 takes 258 ms for 32,768
  columns and 525 ms for 65,536: three thousand times the plain call. It cannot be used to avoid the gathers.

## 5. Tracing

What was tried (M): the forward of each kind of layer captured as one trace, at every context and user count, with
the step's host inputs uploaded before the capture; every stage captured as a trace of its own; one layer replayed
for the next token.

| | result |
|---|---|
| capture and replay of a whole layer | works for all three kinds, 1 and 8 users, 4K to 1M: 18 of 18 |
| output of the replay against the untraced forward | max difference 0.0 in all 18 |
| untraced / traced | 6.0 to 7.5 where the host is the limit (4K; every shared layer but the one at 8 x 1M, 3.6); 1.4 to 2.7 on full layers at 128K and 1M, where the device is |
| a blocking replay against replays queued back to back | 0.08 to 0.25 ms more per replay (the synchronize) |
| the next token through the same trace (`test_traced_layer_follows_step_inputs`) | captured at position P, the step's four inputs rewritten in place for P + 1 with `ttnn.copy_host_to_device_tensor`, replayed: output, cached latent row and index key of P + 1 equal to an untraced step's (max difference 0.0); the rewrite took 1.17 ms for one user |

What a traced step needs that the step does not have today:
* **Step inputs as tensors that exist before the capture.** `begin_decode_step` uploads four new tensors per user
  per step. A capture refuses host writes, reads and synchronizes (`fd_mesh_command_queue.cpp`: "Writes are not
  supported during trace capture"); not provoked on the device, read from the source. Rewriting the existing
  tensors works (above) and still costs a host write each; computing them on the device removes it (section 9).
* **Fixed shapes.** The selection reads `cache.stripe(length=position + 1)`: the slice is a host integer baked into
  the trace. A replay at a position past the captured round of 4,096 would score a stripe that is too short and
  say nothing. A traced step has to score the whole stripe (the indexer masks by position anyway), which is the
  1M column of 3.2 at every context unless the stripe is cut in a few fixed sizes with a trace each. Not run:
  reasoned from the code.
* **Tables built on first use** (`TtHy4Indexer.slot_positions`, `TtHy4LMHead._token_ids`, the cache's shard
  configs) are host uploads the first time; they must have been built before the capture. Here every stage ran
  untraced first.
* **A fixed batch.** The users loop is unrolled into the trace: one trace per set of active users, or always 8.
* **No `record` / `profile`.** Both read back or synchronize inside the step.
* Not measured: one trace of all 78 layers (size of the trace buffer, capture time), and the DRAM the trace region
  has to reserve. The 128 MB region used here held one layer of 8 users (1,619 calls) at a time.

## 6. Memory at the target

DRAM per chip is 34.18 GB (34.04 GB with the 128 MB trace region of these runs).

Weights, BFP8 with BFP4 routed experts (M, the same in all six cases):

| | GB per chip |
|---|---|
| embedding + head readout + LM head | 0.0941 |
| layer 0, dense FFN + indexer | 0.0245 |
| a MoE layer with an indexer | 0.1956 |
| a MoE layer with a shared indexer | 0.1949 |
| 78 layers (A: 0.0941 + 0.0245 + 20 x 0.1956 + 57 x 0.1949) | 15.14 |

Decode cache, BFP8 tiles (M):

| cache of | measured GB per chip | KvCacheConfig.bytes_per_chip, GB | measured bytes per token (3 latent + 2 index-key stripes) |
|---|---|---|---|
| 4K, 1 user x 3 layers | 0.00035 | 0.00027 | 2748.0 |
| 128K, 1 user x 3 layers | 0.00919 | 0.00863 | 2244.0 |
| 1M, 1 user x 3 layers | 0.07330 | 0.06907 | 2237.0 |
| 4K, 8 users x 3 layers | 0.00222 | 0.00216 | 2172.0 |
| 128K, 8 users x 3 layers | 0.06963 | 0.06907 | 2125.0 |
| 1M, 8 users x 3 layers | 0.55684 | 0.55260 | 2124.2 |

One latent stripe alone at 8 users x 128K is 0.02063 GB per chip (M): 629.5 bytes per token, of which 16 are the
cache's fp32 slot-position table (one per cache whatever the layers and users: a row of fp32 padded to a tile,
128 bytes per slot, 4.2 MB per chip at 1M), so 613.5 for the latent against 612 by formula; the index key comes
out at 134 by difference against 136. At 1M the measured cache is the formula plus that table (4.2 MB) for one
user and for eight.

Extrapolated to 78 layers (A, 8 users x 1,048,576 tokens; with 1,000,000 tokens every cache figure is 4.6% lower):

| per chip, GB | |
|---|---|
| decode weights, 78 layers | 15.14 |
| decode cache, 78 latent + 21 index-key stripes: 50,592 B per token, + the position table | 13.27 |
| both | 28.41 |
| left of 34.18 | 5.77 |
| the prefill stack's storage instead of the decode cache: 57,768 B per token (704 x 78 + 136 x 21) | 15.14 |
| decode weights + prefill storage | 30.28, 3.90 left |
| decode cache + prefill storage + decode weights | 43.55: does not fit |
| prefill weights as the stack runs today (dev layout, 20.9) + decode weights | 36.0: does not fit, with no cache at all |
| prefill weights (dev layout) + its own storage for 8 x 1M | 36.0: does not fit either |

What has to be shared for 8 x 1M to fit:
* **One set of weights.** Decode's 2D-sharded set is 15.14 GB; the prefill stack's dev layout replicates the
  attention and indexer weights over the 8 mesh rows (5.5 GB per chip by its own report) and is a second copy of
  everything else. Its tp32 layout is the same 1/32 sharding as decode's, so one set of tensors can serve both;
  today tp32 is refused at start.
* **One copy of the cache.** Either container fits beside one set of weights; the two together do not. Decode
  reading the prefill storage in place needs the ops of section 8; prefill writing decode's BFP8 tile stripes
  needs the stack to store that format (612 instead of 704 bytes per token per layer).
* **BFP4 routed experts.** A BFP8-expert MoE layer is 0.347 GB per chip (measured by the full-model work): 26.8 GB
  of weights, 7.4 GB left, 4 users of 1M.
* **Room for what a step allocates.** Not measured at 78 layers. Today's row gather alone untilizes the cache
  table into a bf16 copy per call: 302 MB per chip at 8 x 1M (A: 262,144 rows x 576 x 2 B). L1_SMALL in use per
  core after the three layers and the head: 6,784 B for one user, 7,680 B for eight, of the 32,768 the shared mesh
  opens with.

## 7. Plan, ranked by what it removes from a step at 8 users x 1M

Start: 6.96 s untraced, 3.28 s traced (A from M; the traced stages of 3.3 add up to 3.35 s, which is the basis
of the table). Each line gives what the change removes from the traced step and how that was estimated. The
changes are listed in the order they would be done; each "step after" assumes the lines above it.

| # | change | removes, ms per step | how estimated | step after, ms |
|---|---|---|---|---|
| 1 | Trace the step: step inputs in persistent tensors, the whole stripe scored, tables built before capture, a fixed batch of 8 | 3,680 | (A from M) whole layers untraced against traced, section 3.1 | 3,280 (3,350 as the sum of the traced stages, the basis of the lines below) |
| 2 | Selection with the users as rows and without `ttnn.gather` (section 8, op A): scores of all users in one call, the top-k itself returning the fp32 scores and positions of its picks | 1,650 of 1,733 | (E) per full layer for the batch: scoring 8 x 0.25 (M per user, no gain assumed), two local top-k passes 2 x 0.125 (M, `[8 x 32768]`), candidate all-gathers and reshapes 1.25 (0.78 today by difference, more for 8 times the payload), merge top-k 2 x 0.19 (M), locate 0.12 (M): 4.1 ms x 21 layers = 86 | 1,710 |
| 2' | the same with `ttnn.gather` kept (users as rows only, no new op) | 1,350 | (E) local 2 x 3.19 + 0.25, merge 2 x 3.87 + 0.38 + 1.25, scoring 2.0, locate 0.12 (gathers and top-k M at 8 rows): 18.1 ms x 21 = 380 | |
| 3 | Row gather that reads K rows, not the table (section 8, op B) | 750 of 754 | (E) 8 x 2,048 rows x 612 B is 10 MB over the Galaxy per layer; one program at the floor. Interim with what exists: slice the user's stripe first, 0.23 ms per user (M at 32,768 rows): removes 610 | 950 |
| 4 | Users as a batch dimension in the rest of attention (projections, cache append, core, output, collect) | 500 of 692 | (E) stages at the op floor cost the same for 8 rows as for 1 (M on every stage that takes rows): 0.31 + 0.12 + 0.09 + 0.06 = 0.58 ms, + 0.43 on full layers; core 0.24 -> 0.5 (8 times the payload, guessed 2x); collect 0.185 -> 1.24 (the 0.15 ms above the collective floors x 8). 2.3 ms per layer | 450 |
| 5 | Attention over the rows where they are: every chip attends with all heads over its own picks and the partial results merge by log-sum-exp (issue 65889; `reference/attention_decode.py lse_merge`), or the prefill stack's `sparse_sdpa` on its storage (section 8, op C). No rows move, no row gather, no locate | 85 | (E) collect 1.24 -> 0.15: the absorbed queries of 64 heads for 8 users are 0.6 MB to gather over one axis, the partial outputs 0.5 to 1 MB to reduce over both; 2.4 MB costs 65 us per axis above the floor (M) | 365 |
| 6 | iHC and the norm after it as few programs: fn projection, sum of squares and the 4 x 4 Gram matrix of the streams in one pass, one small all-reduce, then mix and RMSNorm (the norm's sum of squares is a quadratic form of the Gram matrix, so its own all-reduce goes) | 65 of 76 | (E) 0.97 ms per layer today (M: 2 x 0.30 pre + 2 x 0.08 post + 2 x 0.10 norm, 96 calls, 4 collectives). With ttnn ops, about 10 programs and 1 collective per site: 2 x (10 x 6 + 18) us = 0.16 ms. With a fused op (section 8, op E) 0.08 | 300 |
| 7 | Routed experts only where a user picked one (section 8, op D) | 15 to 50 of 65 | (E) 0.85 ms reads all 8 experts of a chip. 8 users pick at most 64 of 256: 2 per chip on average (0.21 ms at the measured 200 GB/s), 5 to 6 on the busiest chip if picks are uniform (0.55 to 0.64 ms), and the busiest chip sets the time | 250 to 285 |
| 8 | What is left of a layer as few programs: one matmul per projection with its norm and RoPE folded in, the core as one program, shared expert and router fused; collectives sharing a payload where two follow each other on one axis | 110 | (E) left per layer after 7: attention 1.23 ms, MoE without the experts 0.42, iHC and norms 0.16, experts 0.21 to 0.64: 2.0 to 2.45 ms. As about 25 programs and 12 collectives: 25 x 6 + 12 x 18 us = 0.37 ms, plus experts and 0.1 ms of LSE payload: 0.7 to 1.1 ms per layer, 55 to 85 ms per step, plus the selection's 86 | 140 to 175 |
| 9 | Selection as a fused score and a threshold select (section 8, op A'): no candidates gathered, no merge | 40 to 75 of 86 | (E) scoring is 2.0 of the 4.1 ms per full layer and reads 36 MB of index keys per chip for 8 users: 0.2 ms at the rate the experts matmul reaches; a few rounds of counts over the fabric 0.2 to 0.3 ms. 0.5 to 2.2 ms per full layer | 65 to 135 |

Where that ends (E): 65 to 135 ms per step, 7 to 15 tokens per second per user, 5 to 10 times short of the
target. The parent issue's own estimate for plain decode was 30 to 45 tokens/s, from 0.25 to 0.33 ms per layer
on MiMo; the same count of programs and collectives at this board's measured floors gives 0.7 to 1.1 ms.

Other changes the target needs that do not shorten a step:
* **In-place reads of the prefill storage**: memory (section 6) and time to first token. The bridge copies every
  row today (141 s through the host for one user at 131K, measured by the full-model work).
* **Shared weights**: memory, and 354 to 473 s to the first token today because the two sets take turns on the
  chips.
* **BFP4 experts** are already what was measured. BFP8 experts were not timed; if the experts stage is bound by
  bytes read it would be about 1.9 times 0.85 ms (E).
* **Host inputs of a step on the device** (section 9): 8.5 ms per step for 8 users today (M).

## 8. What cannot reach the target without a new ttnn op

Per layer the budget is 0.171 ms. At the measured floors that is about 10 programs and 5 small collectives with
no time left to compute. A shared MoE layer issues 17 collectives per user today; laid out as SC1 lays the model
out (inputs split over 4 chips, outputs over 8, experts and cache rows over 32) it needs about 12 even when
everything else is fused (iHC 2, latent 2, q and gate 1, attention merge 3, o_proj 1, MoE input 1, experts
combine 2, shared expert 2 = 14, less where two share a payload).

| | op | contract | replaces (M) | why existing ops do not do it |
|---|---|---|---|---|
| A | top-k that returns what it picked | scores fp32 or bf16 `[1, 1, U, N]`, N up to 65,536, k = 2048, optional labels `[1, 1, U, N]` -> (values in the input's dtype, labels or column indices) `[1, 1, U, k]`, sorted, at the speed of today's plain call (0.1 to 0.2 ms) | two `ttnn.gather` per top-k: 2.8 to 7.7 ms | `ttnn.topk(indices_tensor=)` does this and takes 258 to 525 ms at k = 2048; `ttnn.gather` reads through the whole input |
| A' | distributed top-k by threshold (better than A) | per chip: masked scores `[1, 1, U, slots]` -> a 0/1 mask or compact list of the chip's own entries that are in the global top-k of each user, and their count; rounds of counts over the fabric inside the op | scoring to locate: 10.3 ms per user per full layer | no chip needs the other chips' picks once attention runs over own rows (op C); a gather-and-merge of 65,536 candidates on all 32 chips is then wasted |
| B | indexed row read from a tile cache | rows uint32 `[U, K]` + table `[users, 1, slots, W]` BFP8 TILE in DRAM -> `[1, U, K, W]` bf16, reading only the tiles that hold the rows | `ttnn.embedding`: 0.08 to 1.21 ms, linear in the table | `ttnn.embedding` untilizes the whole table into a bf16 copy first, and refuses the prefill stack's ND-sharded and packed storage |
| C | sparse attention at decode on the stored rows | the prefill stack's `sparse_sdpa` with S = users query rows: absorbed queries `[1, H, U, 576]`, the KV storage (packed scaled-FP8 or BFP8 tiles), per-chip picks, sink -> per head the partial output `[1, H, U, 512]` and its log-sum-exp, over the chip's own rows only; with it a **single-row append** into the packed storage | row gather, collect, locate, the core: 0.6 to 1.7 ms per user | `sparse_sdpa` and `sparse_kv_push` exist for prefill chunks; nobody has run them with a handful of query rows; `update_padded_kv_cache` writes 32-row windows only |
| D | routed experts for a few tokens | x `[1, 1, U, hidden]`, weights `[1, 1, U, E]` (zero where not picked), the chip's experts -> the weighted sum `[1, 1, U, hidden]`, skipping every expert whose weights are all zero | 0.85 ms whatever the picks | a plain matmul over the chip's 8 experts side by side cannot skip any |
| E | fused iHC site | streams `[1, 1, U, 4 x C]`, fn, base, scale, norm weight -> pass 1: projections, sum of squares, Gram matrix (one program, then one all-reduce); pass 2: sigmoid parametrisation, mix, RMSNorm, and the post-mix of the previous site (one program) | 0.37 + 0.10 ms per site (two mixes, one norm), 48 calls | possible with existing ops at about 10 programs per site; the fused op is the difference between 0.16 and 0.08 ms per layer |
| F | a collective for small payloads | one program that sums (or gathers) `[1, 1, U, W]`, W up to a few thousand, over both mesh axes at once, on persistent semaphores | 14 to 23 us per axis per call, 17 to 21 calls per layer; 0.24 to 0.55 ms per call untraced | `ttnn.all_reduce` is a reduce-scatter plus an all-gather per axis and keeps semaphores per program in L1_SMALL |
| G | unpack of a packed scaled-FP8 KVPE row | the inverse of `pack_scaled_fp8_kv_cache` (FP32 scales and BF16 k_rope out of the row) | the host round trip of the hand-over | only needed while decode keeps its own container |

Even with A to F the routed experts stay above the budget at the measured rate: the experts 8 users pick are
1.36 GB of BFP4 weights per layer over the Galaxy, 42 MB per chip when spread evenly, 0.21 ms at the 200 GB/s
per chip the experts matmul reaches here (M: 0.17 GB in 0.85 ms, one row of tiles of activations; whether DRAM or
the tile arithmetic bounds it is not known without a kernel profile). The bandwidth ceiling in the parent issue
(114 tokens/s for 8 users x 1M) corresponds to about 500 to 550 GB/s per chip (A, back-computed from its table).
At the measured rate the same ceiling would be about 43 tokens/s (E).

## 9. Host round trips left on the decode path

| per step | cost (M) | on the device instead |
|---|---|---|
| cos / sin of each user's position (2 uploads per user) | part of 1.2 to 1.3 ms per user, 8.5 ms for 8 users | position x inverse frequencies in fp32, `ttnn.cos` / `ttnn.sin`; their accuracy at arguments up to 1e6 is not checked |
| append indices (a host table, 1 upload per user) | the same | `kv_cache.locate_rows` on a position tensor: the slot where the chip owns it, -1 elsewhere |
| query position (1 upload per user) | the same | a counter tensor the step increments |
| token ids of the first step or a forced token | one upload | none needed after it: `greedy` output feeds the embedding on the device |
| the users loop in attention | 8 x 73 to 186 ttnn calls per layer | users as a batch dimension (plan 2 to 4) |
| the context length as a host integer | a new program every 4,096 positions, and no trace | score the whole stripe, or a few fixed stripe sizes |
| reading the token for the client | at least one synchronize, 0.09 to 0.13 ms, plus the read | unavoidable; once per step, not per layer |

Rewriting the four inputs of one user in place took 1.17 ms (M, with a synchronize); for 8 users they would be
four tensors of 8 rows, one write each.

## 10. Not measured

* Kernel time per op (no profiler, section 2); everything "device" here is trace replay time.
* One trace of all 78 layers, a traced step with the embedding and the head, the trace buffer size.
* Anything batched over users inside attention: the module takes one user per call. The 8-user numbers are 8
  calls; the batched estimates of section 7 are estimates.
* A cache with real content: the caches are zero-initialised (three layers; the next-token trace test fills 1,000
  random positions). The whole-model accuracy runs are in `tests/test_model_full.py`.
* BFP8 routed experts, `core="expand"`, `head_sum="matmul"` (the model's defaults were timed: absorb, fp32).
* All 99 cache stripes at 8 x 1M on the device: the 78-layer residency is arithmetic from three layers.
* The prefill stack, the bridge, and decode after a real prefill at 128K or 1M.
