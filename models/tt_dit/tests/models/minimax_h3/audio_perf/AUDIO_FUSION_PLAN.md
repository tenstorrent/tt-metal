# Audio decode — implementation plan for the remaining 5-10x

Everything here is sized from measurements in this branch. Each script that produced a number is
named, so any figure can be re-derived rather than trusted. `AUDIO_RESULTS.md` covers what is already
achieved (~1.3x); this covers what is not.

## The two facts the design must respect

**1. Cost is per-row, not per-byte, and rows are almost free to widen.** (`row_cost.py`)

Holding total elements constant at the s6 tail size and varying only the channel width:

| C | rows | fp32 ms | ns/row | GB/s | vs C=8 |
|---|---|---|---|---|---|
| 8 | 662 424 | 2.765 | 4.2 | 23.0 | 1.00x |
| 16 | 331 212 | 1.409 | 4.3 | 45.1 | 1.96x |
| 32 | 165 606 | 0.751 | 4.5 | 84.7 | 3.68x |
| 64 | 82 803 | 0.437 | 5.3 | 145.4 | 6.32x |
| 128 | 41 401 | 0.293 | 7.1 | 216.7 | 9.42x |
| 256 | 20 700 | 0.235 | 11.3 | 271.0 | 11.78x |

A row costs ~4.2 ns whatever it contains. The decode's tail runs at C=8/16/32, so it operates at
23-85 GB/s on a part that reaches **271 GB/s** when rows are wide. bf16 does not change the row cost
at all (2.772 ms at C=8, versus fp32's 2.765 ms) -- it only helps by halving bytes once rows are wide
enough for bytes to matter.

**2. Fixed per-op cost is ~142 us as the decode issues them.** (`op_floor.py`, `op_pipeline.py`)

~90 us perfectly pipelined, ~142 us chained (which is what the decode does), flat in tensor size below
~0.4 MB and unaffected by how many chips the cluster opens. 6955 ops x 142 us = 988 ms against a
~1170 ms wall.

**These multiply.** Op-count reduction attacks term 2; row-widening attacks term 1. Doing only one
leaves most of the factor on the table.

## Where the ops are (`band_ops.py`, over the committed Tracy capture)

127 bands hold **6862 of 6955 ops (99 %)**, median 53 ops per band, of which ~8 are convolution:

    Halo / Move / Conv2d               6.8 each
    Slice / PaddedSlice / SliceWrite   5.2-5.4 each
    UntilizeWithUnpadding              4.8
    Concat                             3.7
    ReshapeView / I2S / S2I            1.7-2.3
    Conv3d                             1.0

About 45 of every 53 ops are scaffolding around 8 real convolutions.

## Plan, in dependency order

**Step 1 — widen rows through the tail. DONE (`a805896d13e`), and close to exhausted.**

`SnakeBeta._fold_factor` capped the fold at one tile width, required a power of two, and bailed for
C >= 32. None of that was necessary -- the factor only has to divide T. Also note the padding idea
below was unnecessary and is dropped: T=165606 admits a fold of 21 directly (2 x 3 x 7 x 3943), so
nothing needs padding.

Measured on the snake, all bit-exact (maxdiff 0.0, pure re-indexing):

    shape              before    after   speedup
    s4 C32  T124206     2.65     1.87     1.42x
    s5 C16  T82806      1.85     0.79     2.34x
    s5 C16  T165606     3.52     1.29     2.73x
    s6 C8   T165606     3.57     0.80     4.46x
    s6up C8 T331212     6.98     1.34     5.21x

Tile occupancy is second-order: C=168 pads to 192 and still wins, having cut rows 21-fold.

**End to end it is worth ~1 %** -- 1.157 / 2.256 / 3.583 s against 1.169 / 2.260 / 3.623, inside the
+-8 % spread. The snake is 140 ms of a 1401 ms stage and the previous fold had already taken part of
it, so 5x on the snake cannot exceed ~8 % even in principle.

**Correction to this plan's original framing.** Step 1 was written as an independent cheap win before
the kernel. It is not. Row-widening only pays broadly if it reaches the FIR path -- ~2500 of the 6955
ops -- and a convolution along T cannot fold T into C without a polyphase rewrite. So row-widening is
**part of Step 2, not a precursor to it**, and there is no remaining low-hanging fruit before the
kernel. The elementwise residual adds and scale multiplies are the only unconverted per-channel ops
left, and they are a small share.

What Step 1 does leave behind is a validated technique the kernel should use internally, and the
knob to tune it (`MINIMAX_H3_AUDIO_FOLD_TARGET`, default 256).

**Step 2 — fuse the band.** The real kernel. **Re-scoped by measurement; read this first.**

Four things were measured before building it. One confirms the premise, three narrow what the kernel
has to be.

*The premise holds: arithmetic is free* (`compute_intensity.py`). At identical element counts,
`sin` -- a transcendental -- is **cheaper than `add`** (0.537 vs 2.860 ms), and the same elements cost
**11.6x** more at C=8 (331 212 rows) than at C=224 (11 829 rows). Cost tracks rows and op count, not
FLOPs, so a fused op will not pay for the extra arithmetic it does. This was flagged below as the
first thing to check and had never actually been measured.

*But the price list was wrong.* Sizing the win as (ops removed) x 142 us averages over ops that differ
by 20x. By time rather than count (PROFILE_2026_08_06.txt): **Concat 285.3 ms / 20.4 %** is the largest
single line item, all convolution (Conv3d + Conv2d) is only 292 ms / 21 %, and data movement totals
~900 ms of the ~1400 ms stage. Removing a convolution is nearly worthless; removing scaffolding is
where the 4-5x lives.

*Merging convolutions is confirmed worthless* (`branch_batch.py`). Batching the 3 parallel AMP branches
as one grouped conv -- called "nearly free, and it wins twice" below -- measures **0.94-1.11x**, and is
lossy at C >= 64 because `groups=3` misses the depthwise SFPU path. Those convs cost 2.8-4.2 ms each,
~20x the average, so they are past the flat-cost regime and merging saves dispatch only.

*Op-level rearrangement from Python is exhausted.* Re-measured under today's conditions (exact conv1d,
zero MAC fallbacks, wider fold), the existing algebraic fusion `MINIMAX_H3_AUDIO_FUSE_BAND=1` is
**1.131 s against 1.105 s, i.e. 2.4 % slower**, accuracy identical. It halves the rows each convolution
reads but doubles the convolution count and still concats to interleave the phases; with cost dominated
by dispatch and rows, those cancel. `_zero_pad_t` concat -> `ttnn.pad` is the same story: 2.2x on the op
(`tpad_bench.py`), exactly 0 % end to end.

*And the floor explains all of it* (`op_floor.py`). fp32 elementwise add, swept over four decades:

    rows        32      256     2048    16384   131072   331212
    ms       0.198    0.193    0.200    0.292    1.188    2.802

A 32-row tensor costs what a 2048-row one costs. Fitting `ms = fixed + bytes/bandwidth`:

    fp32   fixed 180.3 us/op  ->  6955 ops x 180.3 us = 1254 ms,  60 ms allows ~332 ops
    bf16   fixed 127.6 us/op  ->  888 ms,                         60 ms allows ~470 ops

**1254 ms against a decode measuring ~1.1 s: the stage is essentially pure per-op overhead.** Work is
not the binding term anywhere. That is why making an op 2.2x faster changed nothing (it was already at
the floor), why merging 2.8-4.2 ms convs changed nothing (they are 15-23x above it, work-bound), and
why swapping one concat for another changed nothing (no ops removed). Fusion pays exactly when it
deletes ops sitting *at* the floor -- PaddedSlice averages 172 us a call, UntilizeWithUnpadding 225 us.

**So the kernel must be a real device op** that runs up2 -> snake -> down2 without materialising
intermediates, so the Halo/Slice/Concat/Untilize scaffolding is never issued. Rearranging which ttnn
ops are called, in any combination, has now been measured four ways and moves nothing, and the path to
60 ms is arithmetically forced: from ~6955 ops to a few hundred.

*Step 2a is now priced, and it is small* (`fuse_saving.py`). Folding the activation into the conv, using
the scalar seam that already works so no plumbing is needed to measure it:

    case        conv     fused  separate   saving   x127 bands
    s5 C16     0.982     1.021     1.258    0.237     30.1 ms
    s6 C8      0.987     0.974     1.085    0.111     14.1 ms

**~20-30 ms, about 2 % of the decode** -- not the 169 ms the profile attributes to Ternary + Tilize.
relu is a weaker proxy than snake (1.1 ms a call in the profile against ~0.24 ms here), so the real
number could be several times larger, but that is extrapolation. Even optimistically Step 2a is
single-digit percent. **It is worth doing for its own sake and as the proving ground for per-channel
state riding the conv, but it is not a step toward 5x** -- so do not let its six-plus-file cost be
justified by the 5x. The order of magnitude has to come from deleting the ~45 scaffolding ops per band,
which only the full band op does.

(An earlier version of that harness reported 147 ms and had the fused conv beating the plain conv by
30 %, which is impossible. Cause: the raw weight tensor was passed every call, so conv1d re-prepared
weights each time -- 93 % of the measured 13.707 ms -- asymmetrically between arms. Prepare once and
reuse; there is now a guard that fails the case if fused beats plain by >5 %.)


One op per `Activation1d` band, implementing the decomposition already proven exact in
`audio_resample.py::Activation1d._forward_fused` (rel_rmse 8.5e-08 against the literal form,
`verify_fusion.py` checks the index algebra on CPU first). The trap it documents: replicate padding
does **not** decompose into per-phase replicate padding, because the pad region is a constant whose
parity alternates.

Sized at the measured 142 us/op:

| target | ops | saved | total |
|---|---|---|---|
| 53 -> 5 per band | 6862 -> 635 | 6227 | ~285 ms (4.1x) |
| 53 -> 1 per band | 6862 -> 127 | 6735 | ~215 ms (5.4x) |

Two design constraints from measurement, both of which would otherwise bite mid-implementation:

* **In-model L1 contention is real.** Shapes that take L1_FULL standalone are rejected inside a decode
  (`verify` mode: 1 of 42). Budget L1 against what is free mid-decode, not against a microbenchmark.
* **Prefer fewer, larger invocations.** Per-op cost is flat in size below ~0.4 MB, so batching the 3
  parallel resblock branches along the channel axis is nearly free -- and by fact 1 it also widens
  rows, so it wins twice.

### Step 2a — per-channel activation parameters (next increment, designed not built)

`1c11e1dd366` made the depthwise kernel honour fused activations, verified with GELU at 7.621e-08 for
0.9 % cost. `snake_beta` cannot use that seam yet: it is `y = x + (1/(beta+eps)) * sin(alpha*x)^2` with
alpha and beta **per channel**, while the unary seam is parameterised by compile-time scalars. Landing
this removes the snake op, its tilize and its untilize from every band -- 3 of ~53 ops, and more
importantly it is the step that proves per-channel state can ride the conv.

Design, in the order to build it:

1. **Host precomputes `inv_beta = 1/(beta+eps)`.** The kernel then needs no reciprocal, only two
   per-channel vectors: `alpha` and `inv_beta`.
2. **Shape them like the output tiles.** Replicate each value down all 32 rows of a tile, so the
   kernel can use plain `mul_binary_tile` with no broadcast. For C > 32 that is `ceil(C/32)` tiles per
   vector, ordered to match how the output tiles walk the channel axis.
3. **Two cheaper routes are ruled out; do not retry them.**
   * *Reuse the `bias` tensor.* It is already optional, already reaches the program factory, already
     has a CB the reader fills, and is per-output-channel -- exactly alpha's shape. It still fails:
     bias is `C_out` wide and snake needs `2 * C_out` values (alpha **and** inv_beta), and the CB is
     sized to `C_out`. inv_beta cannot be folded into the weights either, because it scales
     `sin(alpha*x)^2` rather than `x`, so the conv cannot absorb it.
   * *Reuse the `bias` CB by passing a 2*C-wide bias.* Verified dead, not assumed --
     `conv2d_op_program_factory_common.cpp:288-292` sizes it
     `num_pages = enable_bias ? per_core_out_matrix_width_ntiles : 0`, i.e. from the conv's output
     width, so a wider bias tensor gets no extra pages.

   **What is left is the optional-input-tensor route, and it is now the only one.** A CB is
   kernel-filled device scratch, not something the host populates at program build, so the parameters
   have to arrive as a tensor. That means `Conv2dInputs`, validate, compute_output_specs, the conv2d
   and conv1d invoke chains, pybind, the program factory and the reader -- six-plus files before a
   first compile. Budget for that rather than looking for a shortcut; three have now been eliminated
   by inspection and the search itself has cost more than the change will.

4. **The weights CB sizes itself from the weight tensor's shape, so no op-signature change is
   needed.** `conv2d_op_sharded_program_factory.cpp:260-262` takes

       weight_matrix_height = b.padded_shape()[2];
       weight_matrix_width  = b.padded_shape()[3];

   from the tensor -- **but that is only half the story, and the optimistic reading of it is wrong.**
   The CB's per-block content comes from the conv dimensions, not the tensor (same file, ~line 477):

       weight_block_h_ntiles  = act_block_h_ntiles * (coalesce ? filter_w : 1)
       weight_block_num_tiles = weight_block_w_ntiles * weight_block_h_ntiles

   So appending to the weight tensor grows the total height while the per-block fetch stays put, and
   the appended tiles are simply never read. Extending the weights CB therefore **does** require
   changing the host block sizing, which is most of the cost a separate CB would have carried. Weigh
   the two again before choosing; the weights-CB route is no longer obviously cheaper.

   The real risk is **CB bookkeeping**: the kernel must pop exactly what the reader pushed or the op
   desyncs, and the two accumulate paths pop differently (see below). Expect to need a compile-and-run
   cycle or two purely on that. Also check whether any TT_FATAL validates weight_matrix_height against
   the conv dimensions -- if one does, it has to learn about the appended tiles.

5. **Carrying them: extend the weights CB.** The weights CB is already
   indexed per channel-tile in exactly the pattern needed, and the reader already fills it. Appending
   the two parameter tiles per channel-tile to the prepared weight avoids a new CB, a new reader
   stream, and a second indexing scheme that could disagree with the first. Note the non-coalesced
   path consumes in1 with `wait_front(1)` / `pop_front(1)` per tap (kernel lines 59/100, 141/179) while
   the coalesced path takes the whole block (216/258), so the two consume differently and the append
   must respect that. A separate CB is the fallback if that proves awkward.
6. **Kernel, on the last tap only**, with the accumulator in DST_ACC:
   `copy alpha -> DST_A; mul_binary_tile(ACC, A, A); sin_tile(A); mul_binary_tile(A, A, A);
    copy inv_beta -> DST_B; mul_binary_tile(A, B, A); add_binary_tile(ACC, A, ACC)`.
   That is 4 DST slots; check against the fp32 DST budget in half-sync before assuming it fits.
7. **Gate it** the way the existing activation is, so nothing changes unless requested.

Verify with `fused_activation.py` extended to snake: the bar is rel_rmse ~1e-07 against a float64
golden, matching what GELU achieved, and cost within a few percent of the unfused conv.

**Step 3 — bf16.** Already runnable (`14572d860ce`), 0.959 s and 34.93 dB against a 28 dB gate. Worth
taking *after* steps 1-2, because bf16's benefit is halved bytes and bytes only become the binding
term once rows are wide.

## What not to do

Nine levers closed by measurement, listed with their numbers in `AUDIO_KERNELS_BENCH.md`: trace,
conv1d L1_FULL, operand splitting, algebraic band fusion, L1-sharded intermediates, cheaper padding
primitives, conv3d UnpackToDestFp32, the 32-chip hypothesis, and act_block_h tuning. They fail for one
of two reasons, both now understood: they optimise bytes (which barely register) or they rearrange ops
without removing them.

## Honest risk on the projection

The 285/215/130 ms figures are arithmetic: measured op counts times measured per-op cost. They assume
a fused op still costs ~142 us while doing far more compute per invocation. Total arithmetic in the
decode is ~20-40 GFLOP against a machine that should do that in single-digit ms, so compute should
stay negligible -- but that is an estimate, not a measurement, and it is the first thing to check with
a throwaway prototype before committing to the full op.
