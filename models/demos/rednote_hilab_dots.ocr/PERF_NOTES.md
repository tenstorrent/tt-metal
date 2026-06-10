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

---

# Perf REDO 2 — decode weight dtype (tick 50)

Hypothesis: the traced step (18.08 ms wall) is DRAM-bound on fp32
weights; convert decode-path weights to bf16 (QKV/o_proj — bf16 holds
the bf16-checkpoint values exactly) and bf8 where the parity gate
allows, KEEPING fp32 attention activations/accumulation and fp32 KV
cache (the ±3122 attention-sink mitigation lives in activations, not
weight storage). lm_head stays fp32 (argmax near-tie exactness,
generation-phase decision).

Steps, each re-validated against the e2e gate (`tests/test_e2e_ocr.py`):

1. QKV/o_proj fp32 -> bf16 (`weight_dtype` knob in text_attention).
   Gate PASS, WER 0.0000 vs HF 0.0000, 5/5 exact.
2. gate/up bf16 -> bf8 (`gate_up_dtype` knob in text_mlp; down stays
   bf16). Gate PASS, WER 0.0000 vs HF 0.0000, 5/5 exact. Kept.

Wall-clock A/B (`profile_ocr.py --traced --bench-replays 200` — new
bench mode times 200 pure replays at advancing slots, host tiles
prebuilt; trace recaptured per run):

| metric (200 replays, dev0) | fp32 attn + bf16 MLP | bf16 attn + bf8 g/u | delta |
|---|---|---|---|
| median ms/replay | 18.03 | 18.07 | flat |
| mean ms/replay | 18.08 | 18.11 | flat |
| traced kernel sum (29 replays, per-GCC mean) | 17.14 | 16.14 | -1.0 |
| MatmulDeviceOperation | 8.70 ms (50.7%) | 8.64 ms (53.5%) | flat |

Top-10 traced (final config, per-GCC mean across 29 replays, dev0):

| us/replay | % | calls | op |
|---|---|---|---|
| 8636 | 53.5% | 197 | MatmulDeviceOperation |
| 1280 | 7.9% | 58 | AllGatherDeviceOperation |
| 1146 | 7.1% | 57 | LayerNormDeviceOperation |
| 995 | 6.2% | 56 | ReduceScatterDeviceOperation |
| 983 | 6.1% | 308 | BinaryNgDeviceOperation |
| 859 | 5.3% | 28 | SoftmaxDeviceOperation |
| 788 | 4.9% | 84 | TransposeDeviceOperation |
| 461 | 2.9% | 28 | NlpCreateHeadsDeviceOperation |
| 273 | 1.7% | 56 | PagedUpdateCacheDeviceOperation |
| 171 | 1.1% | 84 | UnaryDeviceOperation |

Verdict: hypothesis REJECTED with evidence — halving (bf16) and
quartering (bf8) the decode weight bytes moved wall 0%, matmul kernels
<1%. The 1-row matmuls are dispatch/latency dominated, not
weight-streaming dominated; the dtype lever is exhausted. (Note: prior
PERF_NOTES kernel sums 12.17/11.92 were per-session means over
TRUNCATED tracy sessions; the per-GCC method here is truncation-robust
— 1213 ops/replay, sums 17.14/16.14 baseline/after on the same basis.)
The dtype changes are kept (parity-neutral, strictly less DRAM
residency, frees ~250 MB/chip).

Target <= 12 ms/tok NOT reached: floor is matmul dispatch latency
(197 matmuls/step) + CCL 14% + host ~6 ms. Next levers (next pass):
fused rotary (12 host-built slices/layer -> 1 op), fused
silu-mul/residual (BinaryNg 308 calls), all-reduce primitive, folding
the 4 H2D copies + readback chain to cut the ~6 ms host floor.

Tracy artifact: `perf/traced/cpp_device_perf_report_kv_dtype_traced.csv`
(traced dev0 rows + names, final dtype config, replay sessions 1-8/29).

---

# Vision tower optimization REDO — bf16 + SDPA tuning + head-parallel TP (tick 51)

Component pass (`vision_transformer`), measured at the 11k-token
production shape (`/tmp/demo_image1_cropped.jpg`, seq 11224 -> padded
11264, grid [1,92,122]); old single-chip bf16 reference 754 ms, target
<= 900 ms. Parity gate is the E2E 5-sample WER test (block PCC is
informational).

## Step 1 — bf16 activations+weights

Tower dtype fp32 -> bf16 (HiFi4 + fp32_dest_acc on all matmuls).
Untraced wall: 3073 ms. Traced == untraced (3073) — compute-bound, not
dispatch. Tracy: WindowedSDPA 2630 ms = 86.9% of kernel time
(62.6 ms/call x 42, kernel defaults: 32x32 chunks, HiFi4+fp32acc).

Targeted change: SDPA chunks 256x256 on the full 11x10 grid + HiFi2 +
fp32_dest_acc (microbench: 62.7 -> 10.3 ms; bf16 dest acc 9.7 ms
rejected — softmax*V over 11k tokens accumulates in bf16). Chunks are
seq-adaptive (64 below 2048); fp32 high-precision path keeps kernel
defaults (PCC margin 0.9901). Result: 889 ms untraced, kernel sum 779
ms (SDPA 49.7%, Matmul 35.4%). Block PCC bf16 0.9821; E2E WER 0.0000
== HF, 5/5 exact.

## Step 2 — A/B head-parallel TP vs replicate

TP4: QKV columns re-blocked per chip (3 heads/chip, dim=-1 shard),
o_proj row-shard + reduce_scatter+all_gather; MLP col/row parallel,
all-reduce after down. CCL bench at [1,1,11264,1536] bf16: all_gather
1.31 ms, reduce_scatter 1.23 ms (~5 ms CCL/block vs ~12 ms compute
saved).

| 11k tokens, untraced wall | replicate | TP4 |
|---|---|---|
| median ms | 889 | 485.9 |

TP4 kept (`tp_degree=4` in ocr_model; degenerates to 1 on a single
device). Block PCC TP4 0.9852. E2E WER 0.0000 == HF, 5/5 exact, gate
re-passed after each step.

Final kernel profile (TP4, dev0): 470.6 ms — AllGather 23.4%,
WindowedSDPA 23.0% (2.57 ms/call), ReduceScatter 21.6%, Matmul 19.7%.
Next lever: async CCL (~45% CCL share).

Tracy artifacts: `perf/traced/vision_tower_bf16_11k_traced.csv` (step
1), `perf/traced/vision_tower_bf16_tp4_11k_traced.csv` (final).

---

# Perf REDO 3 — decode fused kernels (tick 52)

The decode step competes on op count (948-1213 ops/step inside the
trace, 1-row matmuls dispatch-bound — tick-50 verdict). This pass works
the decode fused-kernel checklist, ONE change at a time, e2e WER gate
(`tests/test_e2e_ocr.py`, 5 samples vs HF) re-run after each step.
Baseline re-measured: 18.17 ms/step (bench --traced --bench-replays 200).

## (a)+(b) Fused fp32 rope + nlp_create_qkv_heads_decode  — KEPT

- Rope collapsed from 14 ops/layer (2x slice/slice/neg/concat/mul/mul/add)
  to 4 (matmul/mul/mul/add) applied PRE-head-split on the fused QKV row:
  `rot = xqkv @ R` where R is the 0/±1 block-diagonal HF rotate-half
  matrix with a ZERO v-block, then `xqkv*cos_cat + rot*sin_cat` with
  cos=1/sin=0 over v. 0/±1 selection in an fp32 matmul is exact math —
  the fp32 attention-sink mandate is untouched. Decode-mode
  `rotary_embedding_llama` itself was REJECTED structurally: the kernel
  is bf16-only (all tensors), requires the META rope convention (its
  32x32 tile trans_mat cannot express the HF half-swap, so q/k weight
  rows + the fp32-prefill K cache would have to flip convention), and
  parallelizes over batch=1 -> 1 core here. The fp32 matmul-rope cuts
  more ops with zero numerics risk.
- `nlp_create_qkv_heads_decode` then hands k/v straight to
  `paged_update_cache` in its HEIGHT_SHARDED [1,1,heads,hd] layout —
  the old permute + interleaved_to_sharded pairs are gone.
- Framework bug found + worked around: with a bf16 DRAM-interleaved
  input, `nlp_create_qkv_heads_decode` silently ZEROES odd head rows on
  Blackhole (bf16 sub-tile lines are 32B; DRAM NoC alignment is 64B).
  fp32+DRAM and bf16+L1 are correct (measured with a synthetic-pattern
  probe; nq 3 or 4, nkv 1-3 all reproduce). The block stages the bf16
  input in L1.
- Step (a)-as-instructed (fp32, explicit core kept): 18.17 -> 17.57 ms.
  Gate PASS (WER 0.0000 == HF, 5/5 exact).

## (c) bf16 scaled_dot_product_attention_decode, Q pre-scaled — KEPT

- Q pre-scale: rope is linear in x, so 1/sqrt(head_dim) is folded into
  the cos/sin q-section ON HOST (zero device ops); the kernel runs
  scale=1.0 and every QK logit shrinks ±3122 -> ±276 before bf16
  softmax sees it.
- KV cache fp32 -> bf16 [1, 1, max_seq, hd] (kernel dtype requirement),
  and the q3|k3|v3 replicated per-chip packing became q3|k1|v1: the
  SDPA decode kernel maps the chip's 3 Q heads onto its single KV head
  natively (GQA) — 44% fewer QKV weight bytes/chip, 3x less cache
  traffic, and nkv=1 re-enables the kernel's sharded output for
  `nlp_concat_heads_decode`. Prefill keeps the fp32 explicit core
  (expands k1/v1 on device) and typecasts post-rope K/V into the cache.
- Causality from the runtime cur_pos tensor (the persistent slot buffer
  shared with paged_update_cache); the streamed [1,1,1,3200] decode
  mask + its per-step H2D copy are deleted.
- SDPA decode program config: 8x8 grid (kernel TT_FATALs >64
  cores/head), q_chunk 32, k_chunk 128, HiFi2 + fp32 dest acc,
  exp_approx_mode=False.
- 17.57 -> 10.76 ms/step. Gate PASS (WER 0.0000 == HF, 5/5 exact, all
  ~115 greedy argmax decisions match HF through the layer-0 sink —
  the near-tie validation). Single-layer harness PCC vs fp32 torch
  reference: 0.99977 (vs 0.99980 for the old fp32 explicit core).

## (d) all_gather_matmul on row-parallel o_proj — KEPT (Linear form)

- The fused `ttnn.experimental.all_gather_matmul_async` is
  Ring-topology-only (tt_transformers gates it on
  `ccl_topology == Topology.Ring`); qb's 1x4 mesh is a Linear line, so
  the applicable form is the gather->matmul restructure: wo REPLICATED
  full-width, the per-chip concat-heads row (384 wide, bf16) is
  all_gathered on dim=3 into full head order, ONE matmul produces the
  complete replicated output. The reduce_scatter+all_gather all-reduce
  of fp32 1536-wide partials disappears (2 CCL ops -> 1 per layer,
  prefill and decode). MLP down_proj keeps RS+AG: replicating its
  [8960,1536] weight would cost ~770 MB/chip.
- Wall flat (10.76 -> 10.75 ms — decode CCL is latency-bound), but
  traced CCL kernel time/step 2275 -> 1817 us (-20%) and 28 ops/step
  fewer. Exact-math change; kept. Weight cost +~100 MB/chip.
- Prefill side effect: prefill_ms 267 -> ~90 (lighter QKV + single-CCL
  o_proj).

## Result

| metric | tick-50 | REDO 3 | speedup |
|---|---|---|---|
| steady_step_ms (bench 200 replays) | 18.07 | 10.63 | 1.70x |
| ocr() steady_step_ms | 18.08 | 10.64 | 1.70x |
| total_ms (gate doc, 11 steps) | 321 | 237.8 | 1.35x |
| traced kernel sum / step (dev0) | 16.14 ms | 9.03 ms | 1.79x |
| ops/step in trace | 1213 | 679 | -44% |

Target <= 12 ms/tok MET. Top-10 traced (per-GCC mean over 29 replay
sessions, dev0, final config):

| us/replay | % | calls | op |
|---|---|---|---|
| 5375 | 59.5% | 169 | MatmulDeviceOperation |
| 1307 | 14.5% | 58 | AllGatherDeviceOperation |
| 747 | 8.3% | 37 | LayerNormDeviceOperation |
| 510 | 5.6% | 28 | ReduceScatterDeviceOperation |
| 470 | 5.2% | 168 | BinaryNgDeviceOperation |
| 116 | 1.3% | 28 | SdpaDecodeDeviceOperation |
| 114 | 1.3% | 28 | UnaryDeviceOperation |
| 102 | 1.1% | 36 | PagedUpdateCacheDeviceOperation |
| 58 | 0.6% | 1 | UntilizeDeviceOperation |
| 54 | 0.6% | 29 | TypecastDeviceOperation |

Parity: e2e gate re-run after EACH kept step and once more on the final
config — WER 0.0000 == HF 0.0000, 5/5 exact every time. Block tests:
text_attention PCC 0.999050 (was 0.999049), decoder_layer 0.999944
(was 0.999945).

Remaining levers (next pass): matmul 59.5% at 169 calls/step is
dispatch-latency floor — fusing gate/up into one wide matmul and
QKV+rope-rot into one (2x fewer dispatches on the hot path); MLP
all-reduce 1.8 ms CCL slice via async CCL; host floor ~1.6 ms/step
(3 H2D copies + 1-int readback) via on-device rope-row computation
from the slot tensor.

Tracy artifact: `perf/traced/cpp_device_perf_report_kv_redo3_traced.csv`
(traced dev0 rows, replay sessions 1-10/29, final config).
