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

## CI

Benchmarks and evals were dispatched at `44e1aefadcc`, i.e. the 90.66 ms state,
before the typecast change landed:

| run | workflow |
| --- | --- |
| `34395177910` | benchmarks |
| `34395194736` | evals |

They queue behind `34360774551`, a benchmarks run dispatched earlier the same day
from `38153c48c8a`, which is the right pre-change CI baseline to compare against.
