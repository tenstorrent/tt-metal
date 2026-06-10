<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# dots.ocr — Pipeline Perf Notes (`ocr` use case, skills/perf)

## Setup

- Device: qb (4 x Blackhole p300c, 1x4 mesh, FABRIC_1D), fp32 decoder path.
- Workload: gate-size doc (`demo/inputs/invoice_total.png`, prompt_len 41,
  L=64, max_new_tokens=24, 11 decode steps to EOS), greedy decode.
- Harness: `tt/profile_ocr.py` (1 warmup + N timed calls, per-step
  callback kinds untraced/capture/replay).

## Baseline (untraced, full-seq recompute, tick-46 state)

| metric | value |
|---|---|
| steady_step_ms | 89.8 |
| total_ms (11 steps + vision + prefill) | 1067 |
| per-step host readback | 19.4 MB fp32 logits |

## Sub-pass 1 — single-call metal trace

The AR loop is fixed-shape full-seq recompute (L padded once), so the
whole decode body (28 layers -> final norm -> fixed lm_head window
slice -> lm_head) is trace-safe with a persistent `[1,1,L,H]` embeds
buffer overwritten per step via `copy_host_to_device_tensor`. Trace
lifetime is single-call (L is call-dependent); capture once at step 0
after an untraced compile pass, release at end of `ocr()`. A
paged-KV-cache token step remains the long-form follow-up.

Trace alone: steady_step_ms 89.8 -> 113.7 (SLOWER): the trace exposed
the per-step host floor — 39 MB fp32 logits readback (64-row window vs
32 untraced) + host argmax dominate; device kernel total under trace is
only 26.07 ms/step.

## Sub-pass 2 — traced tracy + ONE targeted optimization

Tracy traced run (10 replays): kernel sum per replay 33.85 ms, ops/replay 2239
(top-10, total device kernel time, dev0):

| ms/replay | % | calls | op |
|---|---|---|---|
| 14.02 | 41.4% | 383 | MatmulDeviceOperation |
| 5.92 | 17.5% | 2 | ArgMaxDeviceOperation (post-fix; 2 captures) |
| 4.39 | 13.0% | 114 | AllGatherDeviceOperation |
| 2.69 | 7.9% | 112 | ReduceScatterDeviceOperation |
| 2.33 | 6.9% | 616 | BinaryNgDeviceOperation |
| 0.85 | 2.5% | 104 | LayerNormDeviceOperation |
| 0.70 | 2.1% | 40 | NlpCreateHeadsDeviceOperation |
| 0.68 | 2.0% | 168 | UnaryDeviceOperation |
| 0.44 | 1.3% | 2 | UntilizeDeviceOperation |
| 0.37 | 1.1% | 226 | SliceDeviceOperation |

Applied optimization (ONE): greedy argmax inside the trace — all_gather
logits (dim=-1) -> untilize -> multicore argmax; readback drops 39 MB ->
4 KB/step. Exact (fp32 compare; per-chip max-combine was rejected:
`ttnn.max` values are not bit-exact). First-call corruption fixed by
compiling the per-step embed path before capture (compile-time allocs
are unsafe under an active trace).

| metric | baseline | traced+argmax | speedup |
|---|---|---|---|
| steady_step_ms | 89.8 | 18.4 | 4.9x |
| total_ms | 1067 | 385 | 2.8x |

Parity: same text across all calls; e2e gate (5 samples WER vs HF) re-run.

## Recommendations

- Paged-KV token step: full-seq recompute is O(L²); at real-doc L≈4900
  the trace + recompute will be compute-bound — KV cache is the next win.
- Matmul 41%: fp32 mandated for parity; bf8b lm_head + fp32 acc inside
  trace could shave ~3 ms/step but flipped near-ties before — gated.
- Argmax 17.5%: spans full vocab x 64 rows; per-row chunked argmax or
  bf16 argmax inputs would cut ~2 ms/step (parity risk near-ties).
- Untraced step _embed_tokens + host tilize is ~5 ms host/step; fold
  next-token embed into the trace with persistent token-id buffer.

Tracy artifact: `perf/traced/cpp_device_perf_report_traced.csv` — the
traced (METAL TRACE ID set) device-0 rows of tracy's
`cpp_device_perf_report.csv`, slimmed to op identity + kernel duration
to fit repo size limits; full logs in `generated/profiler/.logs/`.

---

# Perf REDO over the KV-cache decode (tick 49)

Tick 48 replaced the full-seq recompute with a KV-cached token step
(`paged_update_cache`, persistent fp32 caches, shared slot buffer);
the per-token cost became O(1) in context but the untraced step was
host-dispatch bound at ~90 ms/tok. This pass traces the token step.

## Sub-pass 1 — cross-call metal trace of the whole token step

The decode body is fixed-shape (one token row vs persistent caches), so
the ENTIRE step — token embedding (persistent uint32 id buffer) -> 28 x
forward_decode (persistent rope/mask/slot buffers, paged_update_cache)
-> final norm -> lm_head -> on-device greedy argmax (all_gather ->
untilize -> multicore argmax, 4-byte readback) — is captured ONCE
(untraced compile pass first; tick-47 recipe) and replayed across steps
AND across `ocr()` calls. Caches/buffers persist in `TtOCRModel`, so the
cross-call lifetime applies (no post-AR device-alloc stages; single-call
release not needed; harness `release_decode_trace()` only at shutdown).
Per-step host work: 4 small H2D copies + 1-int readback.

| metric (gate doc, 11 steps) | untraced kv | traced kv | speedup |
|---|---|---|---|
| steady_step_ms | 88.2 | 18.45 | 4.8x |
| total_ms | 1023 | 321 | 3.2x |
| @2240-token context | ~90 (tick 48) | 18.45 | flat in context |

## Sub-pass 2 — traced tracy + ONE optimization (bf16 MLP weights)

Traced tracy (29 replays, dev0): kernel sum 12.17 ms/replay, 948 ops;
host floor ~6.3 ms/step (4 H2D + readback + python).

| ms/replay | % | calls | op (fp32 baseline) |
|---|---|---|---|
| 6.81 | 55.9% | 157 | MatmulDeviceOperation |
| 0.99 | 8.1% | 308 | BinaryNgDeviceOperation |
| 0.86 | 7.1% | 28 | SoftmaxDeviceOperation |
| 0.85 | 7.0% | 38 | AllGatherDeviceOperation |
| 0.79 | 6.5% | 84 | TransposeDeviceOperation |
| 0.61 | 5.0% | 36 | ReduceScatterDeviceOperation |
| 0.45 | 3.7% | 22 | LayerNormDeviceOperation |
| 0.18 | 1.5% | 11 | NlpCreateHeadsDeviceOperation |
| 0.17 | 1.4% | 84 | UnaryDeviceOperation |
| 0.11 | 0.9% | 22 | PagedUpdateCacheDeviceOperation |

Single ops: lm_head matmul 627 us; 28 MLP-class matmuls at ~126 us.
Applied ONE optimization: gate/up/down weights fp32 -> bf16 (HF's own
weight dtype; activations + accumulate stay fp32, lm_head untouched).
Result: kernel sum 12.17 -> 11.92 ms, steady 18.45 -> 18.08 ms — the
~126 us matmuls are 1-row dispatch/compute dominated, not DRAM-bound;
honest small win, kept (e2e WER gate passes, all 5 samples exact).

| metric | tick-48 baseline | traced + bf16 MLP | speedup |
|---|---|---|---|
| steady_step_ms | 88.2 | 18.08 | 4.9x |

## Recommendations

- Host floor ~6 ms/step: fold rope/mask updates on-device (compute from
  the slot tensor) and chain steps device-side to cut the 4 H2D copies.
- 948 ops/step is dispatch-heavy inside the trace: fused rotary
  embedding (12 slice/neg/concat/mul/add per layer) and fused
  silu-mul (BinaryNg 8.3%) would cut op count ~40%.
- CCL 12.6% (all_gather + reduce_scatter x 56 + lm_head gather): an
  all-reduce primitive or async CCL overlap is the next-largest slice.
- Softmax over fixed 3200 slots 7.2%; chunked mask span at small
  context only helps short contexts.

Tracy artifact: `perf/traced/cpp_device_perf_report_kv_traced.csv`
(traced dev0 rows + op names; final bf16-MLP config, 29 replays).
