# Where the decode step goes, and what is recoverable

Continues `doc/decode_perf_handoff`. That handoff established that the step is
device time and left ~114 ms of 238 ms unattributed. This attributes all of it,
per op, and acts on what it found.

Everything below is measured on this host: 4x Blackhole p300c, `MeshShape(1,4)`,
`FABRIC_1D_RING`, real Qwen3.8-27B weights, all 64 layers, batch 32 — the shape
the server runs (`readiness_vllm/server.log` line 18: `max_num_seqs: 32`).

## Summary

| | ms/step | t/s/u | total t/s |
| --- | ---: | ---: | ---: |
| baseline (`38153c48c8a`) | 237.88 | 4.20 | 134.5 |
| `44e1aefadcc` (conv + matmul + mask gate) | 90.66 | 11.03 | 353.0 |
| **this branch** (+ typecast into the cache) | **88.76** | **11.27** | **360.5** |
| | **2.68x** | | |
| target (`perf_targets.tput_user`, unvalidated — see below) | 24.4 | 41.0 | — |

TTFT is unchanged: 17856 ms cold / 17644 ms warm against 17860 ms cold on the
baseline, same harness, same prompt. Nothing here touches prefill.

Confirmed under a real vLLM server, not just the harness — see "Serving".

Three things were wrong, in descending size, and none of them was a precision
or algorithm question:

0. (and a fourth, small one: the state was materialised twice on write, once
   for a typecast and once for a copy.)

1. **The four-tap causal convolution stores its tap axis on the tile
   dimension**, so every decode step shifts the window with a non-tile-aligned
   slice and concat, which ttnn implements by round-tripping the whole state
   through ROW_MAJOR. 49% of the step. A fused op that avoids this
   (`ttnn.experimental.kda.qkv_causal_conv1d_silu`) was already implemented,
   measured at 1.694x on this exact shape, and left switched off.
2. **The two recurrent state matmuls run on 4 of 110 cores.** 17% of the step,
   6.8x recoverable by changing the program class.
3. **The active-slot mask is applied by blending the whole recurrent state**,
   three broadcast ops over it plus a mask reshape the profile puts at 213 us —
   none of which the per-layer harnesses run, because they pass no mask.
   6% of the step.

And the handoff's missing 114 ms was not a fourth thing: it was the wrong
baseline row. See below.

## How the 114 ms was found

`tests/decode_profile_b32.py` (new) runs the served shape and brackets traced
replays with the signposts the tracy op report keys off. `full_model_perf.py`,
the only harness that emitted those signposts, pins `batch=1`; every profile in
`doc/` therefore described a model that is not the one being served.

Decode step time against layer count, all at batch 32 (`layer_types` repeats
`[linear, linear, linear, full]`, so every multiple of 4 keeps the 3:1 mix):

| layers | ms/step | linear fit |
| ---: | ---: | ---: |
| 4 | 16.72 | 16.56 |
| 8 | 31.57 | 31.40 |
| 16 | 60.99 | 61.09 |
| 32 | 120.01 | 120.46 |
| 64 | 239.42 | 239.20 |

`step(N) = 3.711 N + 1.715 ms`, residuals under 0.5 ms across a 240 ms range.

So the whole of decode outside the layer stack — embedding, final norm, LM head,
sampler, trace replay — is **1.7 ms**. The handoff's open question 1 named the LM
head, the 64 RMS norms and the embedding as the candidates for its 114 ms
residual; the intercept rules all three out.

The residual was never physics. It was a bookkeeping error in the baseline it
subtracted. `doc/ci_dispatch_qb2`'s "What must NOT change" table records

> decode multichip b32 / b1 linear | **2.376** / 0.793 ms

as the shipped default's per-layer time, and the handoff computed
`238.58 - 48 x 2.3697 - 16 x 0.6277 = 114 ms`. But 2.376 ms is the
**`linear_kda_conv` candidate**, not the default: `doc/kda_conv_swap` measured
the same layer at 4.3335 ms for the control and **2.3691 ms** for the candidate,
and the shipped precision config names the control. Measured on this branch with
the same harness at the same shape:

| GDN layer, TP4, batch 32, `multichip_traced_decode.py` | ms |
| --- | ---: |
| `--candidate default` (composite conv, the shipped one) | 4.3335 (`doc/kda_conv_swap` control) |
| `--candidate linear_kda_conv` | 2.3691 (`doc/kda_conv_swap` candidate) |
| `doc/ci_dispatch_qb2`, recorded as the default | 2.376 |

So the step closes exactly, with no residual, once the right row is used. Using
the shipped 4.3335 ms plus the active mask the harness does not pass:

`48 x 4.712 + 16 x 0.721 + 1.7 = 239.4 ms` against 239.42 measured.

(0.721 ms is this branch's measurement of the full-attention layer with the same
harness; the recorded 0.628 ms is a little optimistic. 4.712 ms is what the layer
sweep's slope implies for the GDN layer, and it exceeds the harness's 4.3335 ms
by 0.38 ms because `Qwen36Generator._capture_token_out_trace` always builds an
active mask — all ones when the caller passes nothing — while
`multichip_traced_decode.py` passes none. So every per-layer number in `doc/` is
of a graph the server does not quite execute, but that gap is 0.38 ms per layer,
19 ms of the step, not 114.)

## Per-op attribution

`python -m tracy -r -v --op-support-count 2000` on `--num-layers 8`
(6 GDN + 2 full attention, the same 3:1 mix), one traced replay between
signposts. Device time sums to 31.17 ms against a 31.60 ms measured step.

Two notes on getting this at all. With all 64 layers and a real prefill in the
window, the profiler DRAM buffer overflows and post-processing dies with
`Device data missing: Op N not present in cpp_device_perf_report.csv`; the
harness's `--skip-prefill` captures the decode trace against reset caches
instead, which changes values but not one program, shape or core grid (239.42 ms
prefilled vs 239.42 ms not). And `--op-support-count` sizes a per-RISC DRAM
buffer that is dumped in full: 400000 crashes the device run outright, and 20000
writes a 26 GB device log.

By region, of 31.17 ms:

| region | ops | ms | % |
| --- | ---: | ---: | ---: |
| causal conv window (`…x2560x…`) | 238 | 15.24 | **48.9** |
| recurrent state and head space | 72 | 6.73 | 21.6 |
| projections, norms, residual, CCL | 176 | 4.15 | 13.3 |
| LM head + sampler | 6 | 0.81 | 2.6 |
| everything else | 350 | 4.27 | 13.7 |

Overall DRAM roofline for the modelled ops: **5.8%, 30 GB/s of 512 GB/s.** The
step is not bandwidth bound. It is not compute bound either — it is layout
bound.

### The conv window, op by op

One GDN layer's convolution block, in issue order, on a tensor whose logical
volume is `32 x 2560 x 4` = 0.65 MB in BF16. Shapes are printed
`padded[logical]`:

| op | in -> out | us |
| --- | --- | ---: |
| Permute | `1x1x32x2560` TILE -> `1x32x2560x32[1]` | 117 |
| UntilizeWithUnpadding | `1x32x2560x32[4]` -> `1x32x2560x4` ROW | 261 |
| Slice | `1x32x2560x4` ROW -> `1x32x2560x3` | **615** |
| TilizeWithValPadding | `1x32x2560x3` ROW -> `1x32x2560x32[3]` | 192 |
| UntilizeWithUnpadding | `1x32x2560x32[3]` -> `1x32x2560x3` ROW | 319 |
| UntilizeWithUnpadding | `1x32x2560x32[1]` -> `1x32x2560x1` ROW | 286 |
| Permute | `1x32x2560x3` -> `1x32x3x2560` ROW | 359 |
| Permute | `1x32x2560x1` -> `1x32x1x2560` ROW | 244 |
| Concat | `+ 1x32x1x2560` -> `1x32x4x2560` ROW | 107 |
| Permute | `1x32x4x2560` -> `1x32x2560x4` ROW | 63 |
| TilizeWithValPadding | `1x32x2560x4` ROW -> `1x32x2560x32[4]` | 202 |
| BinaryNg (x conv taps) | `1x32x2560x32[4]` | 221 |
| FillPad / Reduce / Unary | | 148 |
| mask blend (5 ops) + Copy | | 205 |
| Permute | `1x32x2560x32[1]` -> `1x1x32x2560` | 30 |

**~3.0 ms per layer** — and eleven of those fifteen entries are pure layout
conversion. The cause is the state layout `[1, B, C, K]`: with `K = 4` on the
last, tile-padded axis, seven eighths of every tile is padding, `state[..., 1:]`
is a non-aligned slice of the tile dimension, and `concat(dim=-1)` of a 3-wide
and a 1-wide operand cannot stay tiled, so ttnn falls out to ROW_MAJOR and back.

This is not new information in the repository. `doc/kda_conv_swap/README.md`
diagnosed it exactly ("The reason decode was so much worse than prefill is
layout, not arithmetic"), implemented the fix, and measured the shipped shape:

| | control | candidate | |
| --- | ---: | ---: | ---: |
| decode, ms/token, batch 32, 64 layers | 237.87 | **140.40** | **1.694x** |
| TP4 layer trace replay, median | 4.3335 ms | 2.3691 ms | 1.83x |
| PCC vs the control | — | **1.0** | |

The knob is `linear_kda_conv`, and the doc says plainly that "the A/B is one word
in the precision config — `base_policy.linear_attention` from `linear_final` to
`linear_kda_conv`". **That word was never changed.** `OptimizationPolicy`
declares `linear_kda_conv: bool = False` ("Default-off until measured"), it was
measured, and `doc/datatype_sweep/selected_precision_config.json` — the file
`load_precision_config()` reads by default — still named `linear_final`.

Note the 4.3335 ms that doc records for the TP4 layer against the 2.376 ms in
`doc/ci_dispatch_qb2`: the kda_conv harness does pass an active mask. Its
control number is the honest one, and it agrees with the 4.738 ms this branch
derives from the layer sweep.

### The recurrent state matmuls

`output = query @ S` and `memory_value = key @ S_decayed`, per (slot, local
head): `left` is `[32, 12, 1, 128]`, `right` the state `[32, 12, 128, 128]`, so
the batch dimension collapses to 384 with one M tile. The shipped policy pins
this to `MatmulMultiCoreReuseMultiCast1DProgramConfig` on a `(4, 1)` grid and
loops all 384 batch items there. Measured in isolation at that shape:

| program | us |
| --- | ---: |
| auto (no program config) | 623.3 |
| 1D multicast grid 4x1, `in0_block_w=4` (shipped) | 438.9 |
| batched reuse, grid 8x1 (8 cores) | 156.6 |
| batched reuse, grid 8x2 (16 cores) | 100.5 |
| batched reuse, grid 4x8 (32 cores) | 67.7 |
| batched reuse, grid 8x6 (48 cores) | **65.0** |
| batched reuse, grid 8x8 (64 cores) | 66.7 |

**6.8x**, and flat past 32 cores: at 48 cores it reads the 12.58 MB state in
65 us, i.e. 194 GB/s, so it is bandwidth bound there and further grid is wasted.
In the profile these two ops are 434 us each on device, 5.21 ms of the 8-layer
step; across 48 layers, **41.7 ms**.

This is the decode twin of the prefill defect in
`doc/prefill_general_optimizations`: the same batched `[groups, 32, 128, 128]`
shape, the same wrong program class, found and fixed there, never applied here.
That doc blamed the optimize stage's core-grid checklist for being scoped to
"decode-time" consumers only and so missing prefill. The decode consumer was
swept too — `linear_recurrent_explicit_w1/w2/w4`, `linear_recurrent_subblock2` —
but every candidate in that sweep was a `(4, 1)` or `(2, 1)` 1D-multicast row.
The sweep varied `in0_block_w` and the subblock inside one program class and
never left it, so the checklist was satisfied and the 6.8x stayed hidden.
`grid4_w4` won that sweep and is 6.8x off.

Batched reuse hangs, rather than failing, when the batch does not divide the core
count — `_scan_matmul` records this — so `_batched_reuse_grid` picks the largest
grid whose product divides `groups` and leaves the caller on the old program when
the device grid admits none. At batch 32 that is 48 cores; at batch 1
(`groups = 12`) it is 12.

### The active-slot mask

Preserving an inactive slot's recurrent state by computing the new state for
every row and then selecting per row costs, on the full
`[32, 12, 128, 128]` state, two broadcast multiplies and an add — plus two
`ttnn.reshape` calls on the 32-element mask that the profile puts at **213 us
each**, because a ROW_MAJOR reshape that moves the last dimension is a real data
movement op.

`decay = 1` and `beta = 0` make the recurrence the exact identity for that row
instead: `delta` is zero so the rank-1 update is zero, and the decay multiply is
by one. Both are still `[1, 1, batch, heads]` when the gate is applied — one
tile — so it is two ops there plus one cheaper reshape of the mask. Inactive
rows still produce discarded outputs, which is what the previous code contracted
for as well ("inactive rows keep their old state while still producing
(discarded) outputs").

Measured alone, at 8 layers: 31.57 -> 29.72 ms, i.e. 0.28 ms per GDN layer,
**13.5 ms across 48** — consistent with the 0.38 ms/layer gap between the layer
sweep's slope and the mask-free per-layer harness.
`QWEN36_DECODE_STATE_MASK=blend` restores the old form.

The gate is not just close, it is inert on active rows: `beta * 1` and
`where(1, decay, 1.0)` are exact, and a full-model A/B holds per-slot logit PCC
at 0.9999999 with 288/288 argmax agreement across 9 steps at batch 32
(`tests/decode_logits_ab_b32.py`). On inactive rows the stored state is bit
identical, at batch 2 (`full_model_mixed_slots.py`, `INACTIVE_KV_EXACT`) and at
batch 32 with the fused conv live (`tests/inactive_slot_state_b32.py`, 28
inactive slots x 6 layers x 4 ranks, zero moved).

## What is not the problem

Adding to the handoff's list, from the profile:

- **The projections are healthy, and already optimal.** All 40 decode projections
  plus the two LM-head chunks are 42 ops and 3.02 ms of the 8-layer step, at
  48-77% of the FLOP roofline. Swept against the alternatives at the three shapes
  that matter (`doc/decode_perf/`, one device, BFP4 weights, 32 rows):

  | | 32x5120x4352 | 32x5120x4160 | 32x4352x5120 |
  | --- | ---: | ---: | ---: |
  | DRAM-sharded, 8 cores (shipped) | **47.8 us / 262 GB/s** | **45.1 / 265** | **46.9 / 267** |
  | 1D multicast 11x8 (88 cores), interleaved | 47.7 / 263 | 45.4 / 264 | 48.7 / 257 |
  | 1D multicast 11x10 (110 cores) | 49.8 / 252 | 47.5 / 252 | 50.9 / 246 |
  | 1D multicast 8x8 (64 cores) | 50.2 / 250 | 48.9 / 245 | 50.7 / 247 |
  | auto (no program config) | 98.6 / 127 | 96.2 / 125 | 92.6 / 135 |

  Nothing beats 8 DRAM-sharded cores, and ~265 GB/s (52% of peak) is what a
  32-row matmul reaches here. Unlike the two matmuls above, this one was tuned
  correctly. It is worth recording as a negative result: the obvious "only 8 of
  110 cores" reading of the decode projections is wrong.
- **The collectives are minor.** 16 reduce-scatters at 21.7 us and 16
  all-gathers at 33.7 us is 0.89 ms of 31.17 ms, i.e. 2.9%; across 64 layers
  ~7 ms. The `Fabric packet size 4352 B is suboptimal` warning the server logs
  is worth something, but not much.
- **The LM head and the sampler are 0.81 ms**, 2.6%. Consistent with the 1.715 ms
  intercept of the layer sweep.
- **The step is not weight-bandwidth bound.** With every projection at BFP4 and
  the KV cache at BFP8, one decode step reads about 3.8 GB per device — 7.4 ms at
  512 GB/s. The measured step runs the whole model at 30 GB/s.

## Expectations

Written before the post-change measurement, so the comparison is a check.

| change | mechanism | expected |
| --- | --- | ---: |
| fused KDA conv | removes ~2.5 of ~3.0 ms of conv-window layout per GDN layer | -97 ms (1.694x, already measured) |
| recurrent matmul program | 2 x (434 -> ~70) us x 48 | -35 ms |
| active-mask gate | measured 0.28 ms x 48 | -13.5 ms |

These do not simply add: the KDA number was measured with the other two absent,
and all three sit inside the same 4.738 ms layer. Taking the KDA measurement as
given (237.9 -> 140.4 ms) and applying the other two to what remains:

- 140.4 ms - 35 ms (matmul) - 13.5 ms (mask) = **~92 ms, 10.9 t/s/u**, 2.6x.

Measured: **90.66 ms, 11.03 t/s/u** at `44e1aefadcc`, and 88.76 ms / 11.27 t/s/u
once the state stopped being materialised twice on write.

The floor without touching state traffic is roughly 48 x (0.52 projections +
~0.1 conv + ~0.35 remaining recurrent) + 16 x 0.721 + 1.7 = **~60 ms**. Getting
below that needs the state traffic itself cut, which is the next section.

## What is left, and what the target is worth

Still on the table, unimplemented, in descending size. These are measured from
the post-change profile, not guessed.

**The step is now op-count bound, not layout bound.** 740 device ops per 8-layer
step scales to ~5900 for 64, and 88.76 ms / 5900 is 15 us per op. Measured on one
device at the real head-space shapes:

| op | `[32, 12, 1, 128]` (1.57 M elements, M padded 1 -> 32) | `[1, 1, 384, 128]` (49 K, no padding) |
| --- | ---: | ---: |
| `multiply(x, y)` | 35.8 us **(262 GB/s)** | 32.0 us (9 GB/s) |
| `multiply(x, per-head scalar)` | 34.6 | 32.9 |
| `sum(x, dim=-1)` | 38.0 | 12.3 |
| `rms_norm(x, weight)` | 25.2 | 10.7 |
| `silu(x)` | 23.8 | 16.0 |
| **total** | **157.3** | **104.0** |

The padded ops are already at the DRAM ceiling — 262 GB/s, the same rate the
projections reach — so the padding is real work, not stall. But the unpadded
versions are then *latency* bound at ~30 us apiece, so removing the padding buys
only 53 us per layer, and a reshape between the two layouts costs **21-24 us**.
The recurrence needs one M tile per (slot, head) for its matmul and its outer
product, so a packed head space needs five such conversions per layer: **~110 us
spent to save ~53 us.** Repacking head space is a measured net loss, and the
"31/32 of every head-space op is padding" reading -- which is what an earlier
draft of this document predicted was worth 17 ms -- is the wrong model of the
cost. What is left is dominated by the per-op floor, so the lever is *fewer* ops,
not smaller ones.

Which makes fusion the only real remaining lever, and the fused op that would do
it does not fit:

- `ttnn.experimental.kda.sigmoid_gated_rms_norm` computes
  `rms_norm(x) * weight * sigmoid(gate)` and repacks `[B*H, T, V] -> [B, T, H*V]`,
  which is exactly this layer's six-op tail (norm, reshape z, silu, multiply,
  permute, reshape). Qwen3.5 gates with swish rather than sigmoid, but
  `silu(z) = z * sigmoid(z)`, so passing `gate = z` and multiplying the output by
  `z` once -- in the packed layout the op already returns, where `z` natively
  lives -- is the same function. **It is blocked on shape, not on that
  identity**: the op wants `weight` rank 1 and a tile-aligned sequence, and decode
  has `T = 1`, so it fails on
  `attrs.sequence > 0 && attrs.sequence % TILE_HEIGHT == 0`. Its input is also
  head-major while the state matmul produces batch-major. That tail costs
  **136 us per layer, 6.5 ms of the step**, so the prize is real — but reaching it
  needs the same user-major packing the conv op needed plus a permute of the
  padded matmul output, and the net after those would be perhaps half of it.
  Recorded with a reproducible probe (`fused_gated_norm_probe.py`), not taken.

Then, in descending size and each a precision change rather than a rearrangement:

1. **State traffic, ~10 ms.** What is left of the recurrence moves the
   `[32, 12, 128, 128]` state about 90 MB per layer per device: a broadcast
   multiply by decay, one matmul read, an outer-product write, a full add, and a
   typecast into the cache. Read-once/write-once is 13.4 MB. BFP8 arithmetic
   instead of BF16 halves every one of those transfers, but multiplying straight
   off the BFP8 cache is measured *not* bit identical to typecasting first
   (max 0.03 on unit-scale values), so it needs its own validated cycle.
2. **One matmul instead of two, ~3 ms.** `q @ (D + kᵀδ) = q@D + (q·kᵀ)δ` where
   `q·kᵀ` is a scalar per (slot, head), so both projections read the same matrix
   `D = S·decay` and can share one matmul. Algebraically exact, not bit exact.
3. **The M-tile padding inside the state matmul.** One logical M row padded to a
   32-row tile, so 31/32 of its arithmetic is waste — but per the table above the
   op is at the bandwidth ceiling anyway, so this only pays with a batched
   mat-vec primitive, which ttnn does not have.
4. **Batch 1 gets none of the conv win.** The fused decode path needs `K*B`
   tile aligned, so `B` must be a multiple of 8; batch 1 keeps the composite
   conv. Batch-1 decode is already 50.1 ms/step (19.96 t/s/u) because the same
   layout thrash is 32x smaller there, but the composite is still what runs.
   `device_model_spec.max_num_seqs` in `doc/tti_release/autoport_release_spec.json`
   is **1** while the measured server ran 32; that inconsistency should be
   settled before either number is quoted as the product number.

On the 41 t/s/u target itself: it is not derived from this hardware.
`doc/qwen38_checkpoint_swap` records that
`model_performance_reference.json`'s entry for this model carries
`ttft_ms 62.0` and `tput_user 41.0` flagged **"ASSUMED, NOT VALIDATED"**,
extrapolated from Qwen3-32B on a **t3k (8 devices)** while this model runs on 4.
41 t/s/u is a 24.4 ms step. One decode step reads about 3.8 GB per device with
every projection at BFP4, which is 7.4 ms at 100% of DRAM peak — so the target is
not physically excluded. But the projections, the one part of the step that is
purely a weight read, already run at their measured ceiling of ~265 GB/s (52% of
peak) and contribute ~15 ms of the 88.76 ms on their own. With head-space
repacking measured as a net loss and the fused tail blocked on shape, what is
left is the ~13 ms of precision-dependent state work plus fusion that does not
exist yet: call it **~75 ms, ~13 t/s/u** without a precision trade, and ~60 ms
with one. Getting to 24.4 ms needs a fused gated-delta decode op or a batched
mat-vec primitive, not tuning. Quoting 11-14 t/s/u as the reachable range on 4
devices is more useful than carrying an unvalidated 10x gap against a t3k
extrapolation.

## Result

`tests/full_model_perf_batch.py`, batch 32, all 32 rows carrying a real
128-token prompt, 128 decode tokens over a captured token-out trace:

| | baseline | `44e1aefadcc` | this branch |
| --- | ---: | ---: | ---: |
| decode, ms/token | 237.88 | 90.66 | **88.76** |
| t/s/u | 4.20 | 11.03 | **11.27** |
| total throughput, t/s | 134.5 | 353.0 | **360.5** |
| speedup | 1.00x | 2.62x | **2.68x** |
| TTFT cold, ms | 17860 | 17856 | 17849 |
| TTFT warm, ms | — | 17644 | 17652 |

Against the expectations above: predicted ~92 ms and 10.9 t/s/u, measured
90.66 ms and 11.03 t/s/u. The per-layer decomposition also closes —
`multichip_traced_decode.py` at the same shape now reads **1.593 ms** for the
GDN layer (from 4.3335) and 0.721 ms for full attention, and
`48 x 1.64 + 16 x 0.721 + 1.7 = 91.9 ms` against 90.66 measured, where 1.64 ms
is the layer sweep's slope with the active mask the harness omits.

## Correctness

None of the three changes is a precision trade, and each was checked separately
rather than only in combination.

**Prefill is bit identical.** A full-model logits A/B at batch 32 with 32
distinct real prompts (`tests/decode_logits_ab_b32.py`, new) gives
`max|diff| = 0.0` and PCC exactly 1.0 on all 32 slots for the prefill logits.
The linear-attention state left behind by a batch-32 prefill is also bit
identical in both conv arms, every conv window and recurrent state, all 4 ranks
— so the fused path's `[B, K, C]` window and the composite's `[1, B, C, K]`
round-trip exactly across the prefill/decode seam.

**The recurrent matmul program is bit identical.** Real-weight decode PCC for the
linear-attention layer at batch 32 is `0.9982924342381899` under both
`linear_kda_conv` (batched reuse) and `linear_kda_conv_grid4` (the old 4-core
row) — the same value to all 16 digits. `multichip_traced_decode.py` also reports
PCC 1.0 against its single-chip baseline and 1.0 across repeated replays.

**The mask gate is inert on active rows and exact on inactive ones.** Gate versus
blend, everything else held: per-slot logit PCC 0.9999999 and argmax 288/288
across 9 steps at batch 32. Inactive rows keep their state bit for bit at batch 2
(`full_model_mixed_slots.py`, `INACTIVE_KV_EXACT`) and at batch 32 with the fused
conv live (`tests/inactive_slot_state_b32.py`, new: 28 inactive slots x 6 layers
x 4 ranks, zero slices moved, and 192 active slices did move so the check is not
vacuous).

**The fused conv does move decode logits, by less than this port already differs
from HF.** With a dense four-tap kernel and an active mask at batch 32 on TP4,
`doc/kda_conv_swap/check_conv_taps.py` gives `fused_vs_hf` 0.9999602 against
`composite_vs_hf` 0.9999603 over six steps — the fused path is as close to the
reference as the composite, not closer or further. At the model level, over 33
teacher-forced steps x 32 slots = 1056 comparisons:

| | value |
| --- | ---: |
| argmax agreement, fused vs composite | **1050/1056 (99.4%)** |
| median per-step worst-slot logit PCC | 0.9923 |
| worst single slot-step logit PCC | 0.7463 |

The comparison is teacher-forced deliberately. Letting each arm follow its own
argmax makes them diverge after the first slot that picks a different token, and
every later step is then computed from a different history; that measures
divergence amplification, not the change.

The 0.7463 outlier is one row at one step, and it is a saturated one: top-1
probability 0.964 (fused) against 0.990 (composite) on the same token, total
variation between the two distributions **0.030**. A typical row at PCC 0.9994
has total variation 0.078 — more than twice as much. PCC over a 248320-long
vector that is one spike plus a near-flat tail is dominated by the tail, which
is where relative error is largest and where sampling weight is negligible. The
outlier follows the prompt, not the slot: with `--reverse-slots` it moves from
slot 0 to slot 31 with the PCC values permuted and otherwise unchanged, so there
is no slot-indexed defect in the user-major packing.

For calibration, `doc/qwen38_checkpoint_swap` records this port's teacher-forced
agreement with HF at **top1 = 0.950**, top5 = 1.000. The two conv arms agree with
each other (99.4%) an order of magnitude more closely than either agrees with the
reference implementation, and the release config samples at temperature 0.

## Reproducing

```bash
export HF_HOME=... QWEN_AUTOPORT_MODEL_ID=Qwen/Qwen3.8-27B \
       QWEN_AUTOPORT_MODEL_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

# step time at the served shape
python models/autoports/qwen_qwen3_6_27b/tests/decode_profile_b32.py \
  --prompt PROMPT --output out.json --batch 32 --timed-tokens 48 --skip-prefill

# per-op attribution (8 layers keeps the profiler buffer and log sane)
python -m tracy -r -v --op-support-count 2000 -o tr8 \
  models/autoports/qwen_qwen3_6_27b/tests/decode_profile_b32.py \
  --prompt PROMPT --output tr8/perf.json --batch 32 --num-layers 8 \
  --decode-tokens 1 --timed-tokens 2 --skip-prefill
tt-perf-report tr8/reports/*/ops_perf_results_*.csv --arch blackhole \
  --start-signpost FULL_MODEL_DECODE --end-signpost FULL_MODEL_DECODE_END

# A/B either change back off
QWEN36_DECODE_STATE_MASK=blend ...        # mask blend instead of the gate
QWEN36_PRECISION_CONFIG=<json with base_policy.linear_attention=linear_final>

# program-config sweeps, one device, no model
python models/autoports/qwen_qwen3_6_27b/doc/decode_perf/recurrent_matmul_sweep.py
python models/autoports/qwen_qwen3_6_27b/doc/decode_perf/projection_matmul_sweep.py

# correctness
python models/autoports/qwen_qwen3_6_27b/tests/inactive_slot_state_b32.py --output out.json
python models/autoports/qwen_qwen3_6_27b/tests/decode_logits_ab_b32.py \
  --prompt PROMPT --output new.json --steps 32              # then again with the
                                                            # other config and
                                                            # --baseline new.json
python models/autoports/qwen_qwen3_6_27b/doc/kda_conv_swap/check_conv_taps.py \
  --multichip --active-mask --batch 32 --mode decode
python models/autoports/qwen_qwen3_6_27b/doc/decode_perf/prefill_state_seam_probe.py \
  --prompt PROMPT --output seam.pt                          # and --baseline seam.pt
```

`artifacts/` holds every number quoted above: the layer sweep, both per-op
reports (before and after), both program-config sweeps, the logits A/Bs, the
prefill-state seam comparison and the perf points.

## The layout audit, and three pre-existing failures

Enabling the fused conv changes `caches["conv"]` from the composite tiled
`[1, batch, channels, kernel]` to the fused path's row-major
`[batch, kernel, channels]` window — a different rank, a different layout, and a
different axis for the fixed slot. **Three** request-boundary paths were written
against the composite layout only, and CI found the first two before this audit
did:

| path | what it did on the window | found by |
| --- | --- | --- |
| `reset_slots` | in-place elementwise write; a preallocated row-major output is rejected | CI evals `34395194736` |
| `single_slot_prefill_view` | rank-4 slice start on a rank-3 tensor; axis 1 is the kernel, not the slot | CI evals `34404082346` |
| `remap_slots` | addressed the kernel axis instead of the slot axis | this audit |

The first two killed the vLLM EngineCore on the first request. All three now
borrow the composite layout for the duration of the call, reusing the
borrow/restore pair the prefill chunk already uses. Because the narrowed view
nests inside that chunk's own borrow, borrowing is now decided from the **live
layout** — rank 3 means window, anything else means already composite and
borrowing is a no-op — rather than from `linear_kda_decode_ready`, a static flag
every caller keys off. Decode is 88.74 ms/token before and after; these are
request boundaries, not the step.

**Why the suite missed them, and the general lesson.**
`full_model_mixed_slots.py` covers `reset_slots` and `remap_slots`, and
`vllm_reduced_target.py` covers the whole adapter lifecycle — but at batch 2, and
the fused decode path needs `kernel * batch` tile aligned, so at batch 2 it never
engages. Every test that touched this state exercised the one layout the served
batch does not use. That is the same shape of mistake as the per-layer harnesses
passing no active mask, which is what sent the previous handoff chasing a phantom
114 ms: **a test pinned to a batch or a call shape the model does not ship at is
not testing the model.**

Fixed by giving `vllm_reduced_target.py` a `--batch` (default 2, so its recorded
behaviour is unchanged; `--batch 32` reports `fused_kda_decode=3/3` and all five
checkpoints pass) and by `tests/slot_lifecycle_b32.py`, which asserts all three
paths exactly at batch 32 with the fused conv live. It refuses to pass vacuously:
it fails if no layer took the fused path, if prefill left the state at zero, or
if the narrowing view did not actually fire — it counts the invocations, because
the view only engages for exactly one active row with batch > 1.

### Pre-existing failures, not from this work

Found while sweeping every device test that touches per-slot state. Each was
re-run against the baseline commit `38153c48c8a` in an isolated sparse worktree
(models only, so `ttnn` still loads from the built tree) and reproduces
**bit-for-bit identically**, so none is caused by anything here:

| test | failure | note |
| --- | --- | --- |
| `prefill_active_row_pcc.py --batch 8` | slot 1 `argmax_match=False` | `doc/ci_dispatch_qb2` records "argmax match: True, every slot"; linear-state PCC has also drifted from the recorded 0.99857-0.99904 to 0.9996. Likely window is `25d269f6ed7`, which changed the multichip prefill recurrence. |
| `mixed_prompt_state.py` | `TT_FATAL: Invalid arguments to reshape`, `new_volume == old_volume`, in the composite prefill chunk's selector reshape | fused conv is off here (batch 2, `linear_final`), so it is unrelated to the conv work |
| `linear_recurrent_state_transition.py --batch 1 --real-weights` | decode_5 0.9846, decode_7 0.9682 against HF | matches `doc/ci_dispatch_qb2`'s own ranked risk #1, "recurrent-state divergence at long context is unmeasured" |

Everything else passed: `full_model_trace_lifecycle`,
`multichip_linear_attention_smoke`, `multichip_stacked_decoder_smoke`,
`full_attention_inactive_kv`, `linear_recurrent_state_transition --batch 32`
(PCC 0.9999), `full_model_mixed_slots`, `check_conv_taps --multichip
--active-mask --batch 32`, `multichip_traced_decode` (PCC 1.0), and both new
tests.

## Serving

Everything above drives the generator or the adapter directly. All three layout
bugs lived in code reached only through `generator_vllm` under a live scheduler,
so the harnesses could not have found them and did not. This is a local vLLM
server on the fixed code, configured to match `readiness_vllm/server.log`:
`max_num_seqs 32`, `block_size 64`, `sample_on_device_mode decode_only`,
`FABRIC_1D`, `trace_region_size 200000000`, the `qwen36_autoport` bundle via
`EXTRA_MODELS_DIR`, and `QWEN36_PREFILL_PER_REQUEST=1` so the narrowed prefill is
actually taken.

`vllm bench serve`, isl 128, osl 64, 8 prompts, concurrency 8, temperature 0 —
the same shape as the recorded readiness point:

| | recorded (pre-change) | this branch | |
| --- | ---: | ---: | ---: |
| median TPOT, ms | 236.30 | **88.37** | **2.67x** |
| mean TPOT, ms | 238.66 | 95.74 | 2.49x |
| decode t/s/u | 4.23 | **11.32** | 2.67x |
| completed / failed | 8 / 0 | **8 / 0** | |
| output throughput, tok/s | 14.03 | 17.41 | 1.24x |

The 88.37 ms median matches the 88.74 ms the standalone harness measures, so the
harness number was not an artifact of bypassing the server.

**But read the throughput row, not just the TPOT row.** This benchmark is
TTFT-bound: 23.4 s of its 29.4 s is prefill, which nothing here touches, so
end-to-end throughput moves 1.24x while decode latency moves 2.67x.
`doc/kda_conv_swap` predicted exactly this ("a serving benchmark at this batch is
prefill-bound, so end-to-end it would show far less than 1.694x until the prefill
slot-scaling is addressed"). At a decode-dominated shape the win does reach
aggregate throughput — isl 128, osl 512, concurrency 8 gives **59.51 tok/s**
output at 95.34 ms median TPOT. Anyone quoting a single serving number for this
model should say which shape it came from.

Two caveats on the comparison. `max_model_len` was 4096 here against 262144 in
the recorded run, to shorten startup; that changes cache and page-table setup, so
the TTFT column (21467 -> 23376 ms) is not strictly comparable and is not a
claim. And this is one run of each point, not a distribution.

## CI

### How long the benchmark sweep takes, and why the decode win barely moves it

The `benchmarks` workflow runs a sweep of `(isl, osl, max_concurrency)` points,
each with a **hard 7200 s timeout** — `"exceeded timeout of 7200s and was
killed"` appears three times in the baseline log, for the three points below. So
the sweep's wall time is set by how many points hit that cap and by the
prefill-bound long-ISL points, **not** by decode speed.

Measured per point, from both job logs (which *are* fetchable while the run is in
progress via `gh api .../actions/jobs/<id>/logs` — the handoff's claim that
in-progress logs return `BlobNotFound` holds for the run-level log, not this one):

| point | baseline `38153c48c8a` | this branch | speedup |
| --- | ---: | ---: | ---: |
| load + warmup | 8m17s | 9m24s | 0.88x |
| isl 128 / osl 128 / conc 1 | 6m19s | 3m25s | 1.85x |
| isl 128 / osl 128 / conc 32 | 15m54s | 13m06s | 1.21x |
| isl 128 / **osl 1024** / conc 1 | 21m35s | 8m35s | **2.51x** |
| isl 128 / **osl 1024** / conc 32 | 27m00s | 14m04s | **1.92x** |
| isl 1024 / osl 128 / conc 1 | 4m36s | 2m59s | 1.54x |
| isl 1024 / osl 128 / conc 32 | 45m57s | 44m43s | **1.03x** |
| isl 2048 / osl 128 / conc 1 | 6m14s | 4m39s | 1.34x |
| isl 2048 / osl 128 / conc 32 | 1h29m | *running* | |
| isl 4096 / osl 128 / conc 32 | **2h00m — KILLED at 33/128** | | |
| isl 8192 / osl 128 / conc 32 | **2h00m — KILLED at 1/64** | | |
| isl 8192 / osl 1024 / conc 32 | **2h00m — KILLED at 1/64** | | |
| **through isl 2048 / osl 128 / conc 1** | **2h15m** | **1h40m** | **1.35x** |

The per-point speedups line up exactly with the shape argument in "Serving":
**2.5x where osl is 1024 and 1.03x where osl is 128 and isl is long.** Decode
work is what got faster; prefill did not, and it dominates the long points.

**Estimate for the whole sweep: ~13.5 h on this branch against ~15 h on the
baseline**, both inside the 18 h job budget (`timeout-minutes: 1080`). Built from
the measured points above plus the remaining ones scaled by the measured
per-shape ratio, so the tail is an extrapolation, not a measurement:

| remaining | basis | estimate |
| --- | --- | ---: |
| isl 2048 / osl 128 / conc 32 | 1h29m at 1.03x | ~1h27m |
| isl 4096 / osl 128 / conc 1 | 9m34s at ~1.3x | ~7m |
| isl 4096 / osl 128 / conc 32 | killed at the cap either way | 2h00m |
| isl 8192 / osl 128 / conc 1 | 22m08s at ~1.3x | ~17m |
| isl 8192 / osl 128 / conc 32 | killed at the cap | 2h00m |
| isl 8192 / osl 1024 / conc 1 | 1h17m at ~1.5x | ~52m |
| isl 8192 / osl 1024 / conc 32 | killed at the cap | 2h00m |
| isl 10000 / osl 1024 / conc 1 | baseline still running at >1h27m | ~1h |
| isl 10000 / osl 1024 / conc 32 | expected to hit the cap | 2h00m |
| **total with the 1h40m already measured** | | **~13h25m** |

Four of the points are expected to be killed at 7200 s on this branch as well,
because the shape they fail on (`osl 128`, long `isl`, `conc 32`) is the one that
measured 1.03x. **8 hours of a ~13.5 h sweep is four capped points**, so the
sweep will not get materially shorter until prefill does — which is the same
conclusion the throughput row in "Serving" reaches from a different direction.

### Evals: quality is unchanged, and the eval does 1.6x more decode work

`34414429853` on `e970b4f966d` is the first **genuine** eval run of this branch:
1h40m, zero `EngineCore encountered a fatal error`, zero layout errors, 154k log
lines. Against the pre-change baseline `34360801790` on `38153c48c8a`, which was
also genuine (1h38m, zero fatal errors):

| | baseline `38153c48c8a` | this branch `e970b4f966d` |
| --- | ---: | ---: |
| `r1_gpqa_diamond` | **40** (ratio 0.4484, ❌ FAIL vs published 89.2) | **40** (ratio 0.4484, ❌ FAIL) |
| mean seconds per task | 541.6 | 541.5 |
| generation throughput, median | 16.8 tok/s | **29.6 tok/s** |
| concurrent requests, median | 5 | 4 |
| generated tokens in the window | ~93k | **~150k** |
| time decoding / prefilling / idle | 86 / 1 / 8 min | 86 / 1 / 8 min |

Two things to take from this. **Quality does not move**: identical score and
identical ratio, so the eval's failure against the published 89.2 is this port's
pre-existing quality gap — consistent with `doc/SAMPLING_TEXT_QUALITY.md`'s
recorded text-quality defect in long free-running generation — and not something
the conv, matmul or mask changes introduced. **And the speedup does reach the
eval**: it is decode-bound (86 of 94 minutes decoding, ~1 minute prefilling), and
the same 86 minutes of decoding produced ~150k tokens instead of ~93k, at a
*lower* median concurrency (4 against 5). Per request that is 7.4 against
3.4 tok/s, ~2.2x, which is the 2.67x TPOT gain discounted by prefill interleaving.

One thing this does **not** explain: `mean_seconds_per_task` is identical to
541.5 against 541.6 despite 1.6x more decode work in the same window. The
available job logs do not carry per-request completion lines at this verbosity,
so I cannot say from them whether the task set, the per-task token budget, or the
client sets that figure. Recorded as unexplained rather than guessed at.

**A crashed engine can report success.** Two evals runs reported job success with
a dead EngineCore — `run-evals` exited 0 after the traceback. The tell is
duration: 10 minutes, then 10 minutes, for a model that needs ~15 just to stage
and load weights. The second also emitted an `r1_gpqa_diamond` score of **20
against a published 89.2** — that is the dead engine, not a quality regression.
Do not read a green evals run as a passing model without checking the job
duration and grepping for `EngineCore encountered a fatal error`.

| run | workflow | tt-metal | outcome |
| --- | --- | --- | --- |
| `34360774551` | benchmarks | `38153c48c8a` | pre-change baseline |
| `34395177910` | benchmarks | `44e1aefadcc` | failure — `reset_slots` |
| `34395194736` | evals | `44e1aefadcc` | "success", engine dead — `reset_slots` |
| `34404067300` | benchmarks | `2092bf3424d` | failure — transient `git clone` TLS error in the image build, model never ran |
| `34404082346` | evals | `2092bf3424d` | "success", engine dead — `single_slot_prefill_view` |
| `34413443092` | benchmarks | `2092bf3424d` | cancelled once the same defect was known |
| `34360801790` | evals | `38153c48c8a` | genuine, `r1_gpqa_diamond` 40 — the baseline for quality |
| `34414429853` | evals | `e970b4f966d` | **genuine, `r1_gpqa_diamond` 40 — unchanged** |
| `34414440695` | benchmarks | `e970b4f966d` | dispatched after all three fixes |

`34404067300` is worth one note of its own: the image build died in a `git clone`
of tt-metal with `curl 56 GnuTLS recv error` / `fatal: early EOF`. The workflow's
`docker-image` input skips the build, so the retry reused the image the evals run
had just produced from the same SHA rather than rebuilding — which is also why
`34413443092` reached the model in four minutes instead of seventy.
