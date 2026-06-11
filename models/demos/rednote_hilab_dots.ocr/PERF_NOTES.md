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
(traced dev0 rows, replay sessions 1-7/29, final config).

---

# Perf REDO 4 — precision budget (tick 53)

The parity gate is PCC >= 0.99 + e2e WER 0.0; the decode step was running
0.999+ everywhere — headroom paid for in DRAM traffic and kernel time.
This pass spends it down in descending bytes-touched order, ONE change at
a time, e2e WER gate (5 samples vs HF) re-run after each; baseline
re-measured 10.68 ms/step (bench --traced --bench-replays 200).

| step (decode path) | gate | bench ms/step |
|---|---|---|
| 0. baseline (tick-52 config) | — | 10.68 |
| 1. fp32 residual stream -> bf16 (norms/adds/MLP acts; lm_head re-entered fp32) | PASS 5/5 | — |
| 2a. lm_head weights fp32 -> bf8b (233 -> 58 MB/chip/step; logits pinned fp32 out) | PASS 5/5 | — |
| 2b. lm_head acts -> bf16 (drop fp32 re-entry; logits stay fp32) | PASS 5/5 | — |
| 3. rope/QK chain -> bf16 (xqkv/rot/cos/sin; typecast op deleted; add lands in L1) | PASS 5/5 | 9.41 |
| 4a. down_proj weights bf16 -> bf8b | PASS 5/5 | flat |
| 4b. QKV/o_proj weights bf16 -> bf8b | PASS 5/5 | 9.41 |
| (revert) norm gammas bf16 — wall REGRESSED 9.41 -> 9.78, reverted | PASS 5/5 | 9.78 -> 9.40 |

Kept fp32 (measured hazards, recorded): prefill attention activations +
softmax (Qwen2 ±3122 sink — bf16 core PCC ~0.92 on file); lm_head logits
output + fp32 dest acc (greedy argmax near-ties); decode-step accumulation
(HiFi4/HiFi2 + fp32 dest acc everywhere). 4a/4b were wall-flat (1-row
matmuls dispatch-bound, tick-50 verdict) but kept: parity-neutral and
free ~330 MB/chip.

| metric | tick-52 | REDO 4 | speedup |
|---|---|---|---|
| steady_step_ms (bench 200 replays) | 10.68 | 9.40 | 1.14x |
| ocr() steady_step_ms | 10.64 | 9.39 | 1.13x |
| total_ms (gate doc, 11 steps) | 237.8 | 226.3 | 1.05x |
| traced kernel sum / step (dev0, per-GCC mean, 29 sessions) | 9.03 ms | 8.35 ms | 1.08x |

Top-10 traced (final config, per-GCC mean over 29 replay sessions, dev0):

| us/replay | % | calls | op |
|---|---|---|---|
| 4544 | 54.4% | 169 | MatmulDeviceOperation |
| 1268 | 15.2% | 58 | AllGatherDeviceOperation |
| 1045 | 12.5% | 57 | LayerNormDeviceOperation |
| 447 | 5.4% | 28 | ReduceScatterDeviceOperation |
| 428 | 5.1% | 168 | BinaryNgDeviceOperation |
| 155 | 1.9% | 56 | PagedUpdateCacheDeviceOperation |
| 116 | 1.4% | 28 | SdpaDecodeDeviceOperation |
| 80 | 1.0% | 28 | UnaryDeviceOperation |
| 58 | 0.7% | 1 | UntilizeDeviceOperation |
| 57 | 0.7% | 28 | NLPCreateQKVHeadsDecodeDeviceOperation |

Block PCC on the touched blocks: text_attention 0.999050, lm_head
0.999969, decoder_layer 0.999944 — all >= 0.99 with margin.

Target <= 8 ms/tok NOT met (9.40): the dtype lever is now exhausted —
matmul 54% at 169 calls/step is dispatch-latency floor, CCL 20% is
latency-bound, host floor ~1 ms. Remaining levers (next pass, all
structural): fuse gate/up into one wide matmul and QKV+rope-rot into one
(-56 dispatches/step), async CCL / all-reduce primitive for the MLP
all-reduce, on-device rope-row compute from the slot tensor to drop the
cos/sin H2D copies. LayerNorm 12.5% on bf16 rows runs the fp32-gamma slow
path; bf16 gammas measured WORSE at wall (dispatch) — interleaved-norm
kernel work, not config.

Tracy artifact: `perf/traced/cpp_device_perf_report_kv_redo4_traced.csv`
(traced dev0 rows + names, final config, replay sessions 1-7/29).

---

# Perf REDO 5 — final pass over the occupancy-wave components (tick 67)

Final pipeline pass after the per-component occupancy redos (ticks 51-66:
sharded vision tower/attention/MLP/merger, DRAM-width-sharded decode
matmuls in text_attention/text_mlp, layer-scope width-sharded decode
residual in decoder_layer, vocab-sharded bf8b lm_head). Rebuilt the
cross-call decode trace over all the new sharded paths and re-profiled.

## Baseline (tick-66 config, traced)

| metric | value |
|---|---|
| steady_step_ms (bench --traced --bench-replays 200, gate doc) | 6.69 |
| traced kernel sum/step (dev0, per-GCC mean, 29 sessions) | 5.57 ms |
| ops/step in trace | 877 |

Top-5 traced hotspots vs the 13x10/110-core harvested grid (per-GCC mean,
29 replay sessions, dev0):

| us/replay | % | calls | op | cores (occupancy) |
|---|---|---|---|---|
| 2456.5 | 44.1 | 169 | Matmul | 80/110 hot bucket (lm_head 1351us) |
| 1221.7 | 21.9 | 57 | AllGather | 10/110 (latency-bound, 2/2 links) |
| 487.0 | 8.7 | 196 | BinaryNg | 110/110 |
| 460.2 | 8.3 | 28 | ReduceScatter | 10/110 (local-reduction bound) |
| 327.4 | 5.9 | 57 | LayerNorm | 12/110 (recorded sharded ceiling) |

The only un-waved hotspot was the CCL pair (AllGather 21.9% +
ReduceScatter 8.3% = 30.2%). Everything else is at a recorded ceiling
(matmuls DRAM-BW-bound at 220 GB/s with losing core A/Bs, LN at the 12c
sharded ceiling, SDPA at the 64-core kernel cap). The lever axis is the
sync CCL pair in the MLP decode all-reduce (the deferred item from REDO 3/4).

## Applied (ONE) — MLP decode all-reduce: RS+AG -> all_gather(dim=1)+fast_reduce_nc

The decode MLP down_proj produces per-chip PARTIAL [1,1,1,1536] bf16 sums
that need an all-reduce. The sync `reduce_scatter(dim=3) + all_gather(dim=3)`
pair has TWO fabric hops; at the 1-row decode shape the reduce_scatter hop
is latency-bound (local-reduction-bound, 10/110 cores), not wire-bound, so
halving its bytes does nothing (REDO 4 finding). Replaced with ONE
`all_gather(dim=1)` (replicas stacked on a new axis -> [1,4,32,1536]) +
`fast_reduce_nc(dims=[1])` local 4-way reduction.

Decode-shape CCL A/B (traced replay, num_links=2, perf tick 67):

| MLP all-reduce form | us/op-chain @ [1,1,1,1536] bf16 |
|---|---|
| sync reduce_scatter(d3) + all_gather(d3) | 30.80 |
| reduce_scatter_minimal_async + all_gather_async | 29.77 |
| all_gather(d1) + fast_reduce_nc | 24.02 |

Async CCL was also A/B'd and REJECTED (29.77 vs 30.80 — at the 1-row
shape the global-semaphore setup overhead cancels the pipelining win;
same verdict as vision_block tick 51 found at the 11k shape). The
all_gather+reduce form wins by 22%. Numerics identical to RS+AG within
bf16 rounding: max|diff| vs 4x host truth = 0.0625 for BOTH forms (the
reduction order differs but both accumulate in fp32 on-fabric / fp32
fast_reduce). Prefill keeps RS+AG (wire-bound regime — the dim=1 gather
would move 4x the bytes there).

| metric | tick-66 | REDO 5 | speedup |
|---|---|---|---|
| steady_step_ms (bench 200 replays, gate doc) | 6.69 | 6.66 | 1.00x |
| traced kernel sum/step (dev0, per-GCC mean, 29 sessions) | 5.57 ms | 5.37 ms | 1.04x |
| ReduceScatter calls/step | 28 | 0 (folded into AG+reduce) | — |

Wall delta is noise-level (1-row decode is dispatch/latency-bound at the
floor, not CCL-throughput-bound — the same regime REDO 3/4 documented),
but the change removes the 28 ReduceScatter dispatches/step (877 -> 905
ops shows the AG+reduce pair is 2 ops vs RS+AG's 2, net even at op count
but the RS local-reduction kernel time disappears: ReduceScatter 460us +
AllGather portion -> AllGather 1404us + FastReduceNC 67us; the MLP slice
specifically drops ~6us/step of the kernel sum). Parity-neutral and the
canonical decode all-reduce idiom; kept.

Top-5 traced after (per-GCC mean, 29 sessions, dev0; kernel sum 5.37 ms):

| us/replay | % | calls | op |
|---|---|---|---|
| 2458.3 | 45.8 | 169 | Matmul |
| 1403.9 | 26.2 | 57 | AllGather |
| 489.6 | 9.1 | 196 | BinaryNg |
| 307.0 | 5.7 | 57 | LayerNorm |
| 118.9 | 2.2 | 85 | Slice |
| 107.1 | 2.0 | 28 | SdpaDecode |
| 66.9 | 1.2 | 28 | FastReduceNC |

## Gates

- PCC: text_mlp block 0.999989, decode-row (worst of 8) 0.999893 (>= 0.99).
- e2e WER parity: ttnn_wer 0.0000 == HF 0.0000, tolerance 0.05, 5/5
  samples exact (`tests/test_e2e_ocr.py`).
- 180-token doc drift gate (production doc /tmp/demo_image1_cropped.jpg,
  2305-token context, eos disabled): traced == untraced TOKEN-EXACT over
  all 180 tokens, identical text. No precision drift from the new
  all-reduce reduction order.

## Final steady-state numbers

| metric | value |
|---|---|
| steady decode (gate doc, 200 traced replays) | 6.66 ms/tok |
| steady decode (180-tok doc, traced replay) | 6.92 ms/tok |
| steady decode (180-tok doc, untraced) | 86.99 ms/tok |
| warm doc e2e (vision + prefill + 180 decode, traced) | 2.257 s |
| prefill (2305-tok doc, once) | ~405 ms |

Target <= 8 ms/tok MET (6.66/6.92). The decode floor is now Matmul 46%
(169 1-row dispatches/step, DRAM-BW + dispatch-latency bound) + AllGather
26% (latency-bound at the 2/2-eth-link HW ceiling). Remaining structural
levers (all deferred, diminishing returns): fuse gate/up into one wide
matmul and QKV+rope-rot into one (-56 dispatches/step), on-device rope-row
compute from the slot tensor (drop the cos/sin H2D copies, ~1ms host
floor). Both are op-count plays against a dispatch-latency floor; the
per-op kernel time is already at recorded ceilings.

## Deferred (pre-existing, out of scope for this perf pass)

Multi-DOCUMENT L1 clash: a second full-resolution-document ocr() call on
the SAME model instance hits TT_THROW program.cpp:1335 — the persistent
L1 decode state (per-layer fp32 QKV bias rows + traced input shards) left
by the first call's decode, PLUS a ~1.3 KB/bank framework-internal L1
allocation that has no Python handle, fragment L1 so the doc-scale vision
tower's full-grid CBs (need L1 up to ~1.27 MB) no longer fit (largest
contiguous free drops to ~1.14 MB). Freeing the model's own decode L1
recovers most of it but the unfreeable internal allocation still blocks
the second call. Confirmed PRE-EXISTING at HEAD (commit 3c4e1f9, tick 66)
— independent of this tick's CCL change. A SINGLE doc generation (one
ocr() call, vision on empty L1) is unaffected, which is why the e2e gate
(gate-scale images) and the single-call 180-token doc drift gate both
pass. Fix belongs in the allocator / a vision-CB-shape reduction, not the
perf phase.

Tracy artifact: `perf/traced/cpp_device_perf_report_final_tick67_traced.csv`
(traced dev0 rows + op names joined from tracy_ops_data.csv, final config,
replay sessions 1-7/29).
