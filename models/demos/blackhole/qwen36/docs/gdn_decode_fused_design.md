# Fused GDN decode kernel design

Status: implemented behind `QWEN36_GDN_FUSED_DECODE=1` (default off; composite path preserved
for A/B and PCC gating). Code: `models/demos/blackhole/qwen36/tt/gdn/fused_decode/`.

## Problem

The composite GDN decode layer issues ~80 tiny ttnn ops (conv shift register, head-prep
slices/reshapes/repeat_interleave, ~30-op fp32 recurrent delta-rule step, gated out-norm) at
~11 µs/op dispatch overhead — the dominant share of the ~55-60 ms decode step at B=8/TP=8
(48 GDN layers). See `~/.tt-buddy/notes/learn-qwen36-decode-anatomy.md`.

Shapes (Qwen3.6/3.8-27B, global 16 K-heads / 48 V-heads @ 128, conv K=4):

| | TP=8 | TP=4 |
|---|---|---|
| per-device Nk / Nv | 2 / 6 | 4 / 12 |
| qkv_dim_tp / value_dim_tp | 1280 / 768 | 2560 / 1536 |
| qkvzab width (tiles) | 2060 (65) | 4120 (129) |
| recurrence work units B·Nv | 48 | 96 |

## Op boundaries

Everything between the qkvzab in-projection matmul and the out-projection matmul becomes
**two generic_ops** (~55 composite ops → 2):

1. **`conv_shift_silu`** — channel-parallel FIR conv + SiLU + in-kernel shift-register
   writeback. Reads qkv straight out of qkvzab (no slice op).
2. **`recurrence`** — per-(b, v-head): GQA head selection, sigmoid/softplus gates, q/k L2
   norms, the fp32 delta-rule state update (in place), and the gated rmsnorm × silu(z)
   output gate. Reads z and the a/b columns straight out of qkvzab.

Conv is a separate op rather than fused into the recurrence because the two have orthogonal
parallelism: conv is channel-parallel (each channel computed once), the recurrence is
(batch, head)-parallel with GQA sharing (3 v-heads per k-head at rf=3). Fusing conv into
the recurrence cores would recompute each k-head conv 3× per batch row and multiply tap/state
DRAM traffic ~5× (≈4.6 MB/layer for taps alone); a cross-core producer/consumer split inside
one program would need semaphore barriers. Two ops keep every hazard local and still hit the
1-3-ops-per-layer target (in-proj, conv, recurrence, out-proj + 2 CCLs per layer).

## Semantics (must match the composite path)

From `recurrent_gated_delta_rule_decode_ttnn` (high_precision) + `tp.py:forward_decode`:

```
conv   = silu(tap0*st1_old + tap1*st2_old + tap2*st3_old + tap3*qkv)   # post-shift window
st     <- [st1_old, st2_old, st3_old, qkv]                              # raw pre-silu inputs
q,k    = conv blocks of k-head vh//rf ; v = conv block of vh            # GQA in the reader
beta   = sigmoid(b);  decay = exp(-exp(A_log) * softplus(a + dt_bias, beta=1, thr=20))
qn     = q * rsqrt(sum(q^2) + 1e-6) * Dk^-0.5 ;  kn likewise (no scale)
hd     = h * decay                                                       # decay BEFORE read
delta  = (v - kn @ hd) * beta
h_new  = hd + kn^T (x) delta                                             # in-place writeback
o      = qn @ h_new
out    = o * rsqrt(mean(o^2) + 1e-6) * norm_w * silu(z)                  # gated norm, no +1
```

Precision: state and the whole recurrence in fp32 (fp32 CBs, `fp32_dest_acc_en`, HiFi4 —
same as the validated `chunk_gated_delta_rule` scan phase; the composite ran HiFi2). The
conv accumulates in fp32 dest and hands off bf16, like the composite's bf16 conv output.
`o = qn @ h_new` is computed as `qn @ hd + qn @ outer` accumulated in one dest pass —
algebraically identical, one fp32 rounding fewer than materializing h_new first.

## Core / work mapping

- **conv**: work item = one tile column (40 at TP=8, 80 at TP=4), split contiguously over
  `min(Wt, grid)` cores (ceil/floor). Per column: 8 tile reads (3 states + qkvzab + 4 taps),
  4 bcast-row muls into dest 0..3, SFPU dest-to-dest adds, silu, 1 output + 4 state writes.
- **recurrence**: work item = one (b, vh), one per core (48 @ TP=8, 96 @ TP=4; asserts
  units ≤ grid, 130 on BH). K is never split (both reductions run over full Dk), mirroring
  the chunk kernel's scan distribution. Per core ≈ 37 tile reads (~110 KB, dominated by the
  64 KB fp32 state), ≈ 120 tile ops, 16 + 4 tile writes.

## Row semantics (the trick that removes masks and repeats)

All activation tiles hold every batch row (B ≤ 32 rows in one tile row). Every compute step
is row-wise — squares, ones-matmul row-sums (each row's Σ lands in every column, so column 0
is a per-row scalar for `bcast_cols`), per-row norms — so each row stays self-consistent for
its own batch. Batch b is only ever *selected*, twice:

- `mul_tiles_bcast_rows(..., bcast_row_idx=b)` (BH unpacker row-select, verified in
  `tt_llk_blackhole/llk_lib/llk_unpack_AB.h`) picks row b for the outer product and for
  scalar extraction — no one-hot mask tensors, no repeat_interleave.
- the writer emits only row b of each output tile (two 64 B face-row NoC writes per tile).

Scalar gate factors (decay, beta) are materialized as full-broadcast tiles with
`ones ×bcast_row(b) → transpose → ones ×bcast_row(col)` (three tile ops), then applied with
`bcast_scalar`. The all-ones tile is fabricated by the reader (no host constants).

## CB plan

- conv: 4 bf16 CBs — inputs ×2 (compute copy + writer copy for the shift), taps, output.
- recurrence: 29 CBs, ~48 KB bf16 inputs + ~440 KB fp32 intermediates per core (fits L1
  with wide margin). Everything is single-shot (CB capacity == total tiles), reader pushes
  all inputs up front; three CBs are sequentially reused (sq, colscale, and the
  delta/dm/vread trio for the output-gate stage) to stay within 32 CB indices.
- Dest: fp32 half-sync (4 tiles max); conv uses 0-3, recurrence only 0.

## State aliasing (in-place, trace-compatible)

generic_op imposes no read-only semantics: state tensors ride in `io_tensors` (keeps buffers
alive, orders the op) and kernels write them via `buffer_address()` + interleaved
`TensorAccessor` — the same mechanism as the deepseek_v3_b1 KV-cache update. Buffer
addresses are baked per launch, so under trace capture all replays see the persistent
`rec_state` / `conv_states` / scratch addresses (allocated in `reset_state`, before any
capture). Correctness of the in-place writes:

- recurrence: each (b, vh) core owns its 16 state tiles exclusively — plain full-tile writes.
- conv: cores own disjoint channel columns; within a core, the writer performs the shift
  only after (a) the conv output for that column arrives (⇒ compute consumed the reader's
  copies, ⇒ the reader's DRAM reads of the old state completed) and (b) its own `cb_shift`
  copies are in L1. No cross-core or cross-RISC hazard remains.
- gated output: per-row writes are disjoint across cores; the b==0 core zeros rows
  [B, 32) so the out-projection never reads uninitialized memory.

## generic_op vs dedicated C++ op

**generic_op** (deepseek_v3_b1 style: hand-built `ttnn.ProgramDescriptor`, JIT kernels).
Verified against the actual infrastructure before deciding:

- In-place/aliased IO: supported and precedented (KV cache in `fused_ops/kv_cache_branch`,
  `attention_block`) — the blocker that would have forced a C++ op does not exist.
- Trace capture: works (`test_lm_head_sampling.py` traces a generic_op); descriptor-build
  host cost is paid once at capture.
- JIT-from-file kernels + named compile-time args + per-core runtime args: all supported —
  zero-rebuild measure loop on the cluster.

A dedicated C++ op would buy host-side validation and lower untraced host overhead, at the
cost of a rebuild per kernel iteration. Decode runs traced, so the untraced overhead only
affects A/B experiments; not worth it for bringup. Revisit only if the descriptor-build cost
shows up in traced-capture time or the op graduates to production API.

## Integration

`TPGatedDeltaNet.forward_decode` branches to `_forward_decode_fused` when
`QWEN36_GDN_FUSED_DECODE=1`, the folded qkvzab weight is active (`_fuse_ab`), the step is
full-batch (B == max_batch_size), and the state is fp32. Partial-batch buckets and every
other configuration keep the composite path. The fused branch runs: qkvzab matmul (same
config as `_project_qkvzab`, minus the slices) → `conv_shift_silu` → `recurrence` →
`_row_proj` → all-reduce. Persistent scratch (`conv_out` bf16, `gated_out` fp32) is
allocated in `reset_state`.

## Expected impact

Per GDN layer the region between in-proj and out-proj goes from ~55 dispatches (~600 µs at
~11 µs/op) to 2 (~20-40 µs incl. ~8 MB/layer DRAM state traffic). Across 48 layers this is
the −20-30 ms/step lever identified in the decode anatomy note. Numbers to be confirmed by
Tracy on device.

## Verification

`models/demos/blackhole/qwen36/tests/test_gdn_fused_decode.py` (single BH card):

- conv op vs torch golden (PCC) + bit-exact shift-register writeback check;
- recurrence op vs torch fp32 golden AND vs the composite
  `recurrent_gated_delta_rule_decode_ttnn` chain (PCC ≥ 0.999 on output and state);
- 8-step state-evolution trajectory through both fused ops (in-place state), both TP=8 and
  TP=4 head configs, and DRAM + L1 qkvzab placements.
