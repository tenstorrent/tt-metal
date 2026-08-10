# Fused band op — design, grounded in where the 53 ops actually come from

Written before the implementation, because the op-count structure turned out not to be what
`AUDIO_FUSION_PLAN.md` assumed, and it changes the op's shape.

## Where the 53 ops per band come from

The plan treated the band as "8 convolutions plus 45 scaffolding ops". The profile disagrees about
what the scaffolding *is*. From PROFILE_2026_08_06.txt, over 127 bands:

    HaloDeviceOperation      870      Conv2dDeviceOperation   870      MoveDeviceOperation   870
    PaddedSliceDeviceOp      654      SliceWriteDeviceOp      654

**Halo, Move and Conv2d are all exactly 870.** One Halo and one Move per conv *invocation*, and
870 / 127 = **6.85 conv invocations per band — where the band contains only two convolutions.**

The reason is already written down in `audio_ops.py:375`: we hand `ttnn.conv1d` a DRAM-interleaved
tensor with no slice config, so every call takes the DRAM slicing loop, and that loop emits

    PaddedSlice -> Halo -> Move -> Conv2d -> SliceWrite

per slice. At ~3.4 slices per convolution that is 2 x 3.4 x 5 = ~34 ops, which is most of the 53.

So the band is not "2 convs surrounded by scaffolding". It is **2 convs multiplied ~3.4x by DRAM
slicing, each slice paying a 5-op wrapper**, plus the activation, its layout round trip, and the pads.

## What that implies for the op

The fused op cannot be one invocation per band -- the data does not fit L1, which is why the slicing
exists. It must be **one invocation per T-chunk**, each chunk doing the whole band:

    per chunk:  read window -> upsample x2 -> snake -> downsample x2 -> write

At the same ~3.4 chunks per band that is **~3.4 ops per band instead of ~53**, which is the 4-5x, and
it is the same arithmetic the plan reached by a different and partly wrong route. The win comes from
the wrapper collapsing: one PaddedSlice/Halo/Move/SliceWrite set per chunk instead of one per conv per
chunk, and no activation op or layout round trip at all.

## The math the kernel has to do

Every stage is per-channel, so the op is a 1-D nonlinear stencil applied independently per channel.
For ratio 2 with `up_kernel_size = down_kernel_size = K = 12`:

* **Upsample** is polyphase, so no zero-stuffed tensor is ever built. With taps `h[0..K-1]` scaled by
  the ratio, the two phases are the even and odd tap subsets, each length K/2 = 6:

      u[2n]   = sum_j h[2j]   * x[n - j + off]
      u[2n+1] = sum_j h[2j+1] * x[n - j + off]

* **Snake**, pointwise on the upsampled stream, with per-channel `alpha` and `inv_beta = 1/(beta+eps)`
  precomputed on the host so the kernel needs no reciprocal:

      s = u + inv_beta * sin(alpha * u)^2

* **Downsample** is a stride-2 FIR over `s`, so only even outputs are needed and only those are
  computed:

      y[n] = sum_k g[k] * s[2n - k + off]

Composing them, one output `y[n]` depends on a window of about `K/2 + K/2 = 12` input samples. The
kernel never materialises `u` or `s` beyond the small window it needs, which is the entire point --
those two intermediates are what currently cost a DRAM round trip each.

**The trap, carried over from `audio_resample.py::_forward_fused` and still true here:** replicate
padding does *not* decompose into per-phase replicate padding, because the pad region is a constant
whose parity alternates. The chunk edges must take their halo from real neighbouring samples, and only
the band's outermost chunks apply the replicate rule.

## Why this reuses the existing op rather than being a new one

A new ttnn op needs device operation, program factory, reader/writer/compute kernels, pybind and
CMake, then a full rebuild (~40 min, measured on the h3-prefix build). Most of that already exists and
works for `conv1d` depthwise: the halo gather, the sharding, the slicing loop, the weights CB, and --
from this branch -- SFPU fp32 tap accumulation and a fused-activation seam in
`compute_depthwise_conv1d.cpp`.

So the cheaper route is to extend that kernel with a band mode: same reader, same halo, same weights
CB carrying both tap sets, and a compute kernel that runs the three stages in DST before packing. The
per-channel snake parameters are the one genuinely new host-side piece, and they are the same
optional-input-tensor problem Step 2a already scoped.

## Carrying the per-channel parameters: weights CB, not an optional input tensor

`AUDIO_FUSION_PLAN.md` Step 2a chose the optional-input-tensor route and called the weights-CB route
"no longer obviously cheaper". Counted properly it is the cheaper one, by a lot, and the deciding
factor is not line count.

    optional input tensor   operation struct, validate, compute_output_specs, create_program,
                            the conv2d and conv1d invoke chains, pybind        -- 6+ files
    weights CB              per-block fetch width in the program factory (1 file),
                            kernel reads two extra tiles (done), host appends
                            the parameter tiles to the weight tensor (Python)  -- 2 files

An **op-signature change is all-or-nothing**: no subset of those six files compiles, so it cannot be
landed or handed over incrementally, and a half-applied signature change leaves the tree unbuildable.
The weights-CB route touches no signature, so each piece lands independently and the tree builds at
every point.

The plan's objection to the weights CB stands and is already handled: the per-block fetch comes from
the conv dimensions, not the weight tensor's shape

    weight_block_h_ntiles  = act_block_h_ntiles * (coalesce ? filter_w : 1)
    weight_block_num_tiles = weight_block_w_ntiles * weight_block_h_ntiles

so appending to the tensor alone leaves the extra tiles unread. Widening that count by 2 on the last
tap is the one host change required, and it is the same work the optional-tensor route would have
needed for its own CB anyway.

### Final answer on the carrier: a small persistent CB, not the weights CB

Two attempts at the weights-CB route each surfaced a further problem. Both are real and neither is
visible without writing the code, so they are recorded here rather than left to be rediscovered.

**1. The reserve count is a separate compile-time arg.** The reader's
`dfb_weight_obj.reserve_back(weight_block_num_tiles)` / `push_back(...)` take compile-time arg 7, which
is *not* the CB page count set in `conv2d_op_program_factory_common.cpp`. Enlarging the CB alone makes
the reader overflow its reservation. Bumping the shared `weight_block_num_tiles` instead is wrong the
other way: the compute kernel uses the same value for its tap loop bound and would run two taps long.
Solvable with a `#define` for the extra count (no arg re-indexing needed), but see 2.

**2. The killer: pop rate.** The compute kernel's non-coalesced path pops in1 **once per tile per
tap** -- `block_num_tiles` tiles x `num_taps` taps per block. `apply_snake_beta` popping two tiles at
the last tap therefore consumes `2 * block_num_tiles` per block, while the reader pushes the parameters
once per block. The counts cannot be reconciled by pushing more, either, because **the parameters are
the same for every tile in the block**: popping them at all destroys them for the tiles that follow.

The parameters are read-many, write-once data with a different lifetime from the streaming weights, so
they do not belong in a streaming CB. They want **a small dedicated CB, filled once and never popped**,
which the compute kernel reads with `wait_front` and leaves in place.

That also dissolves problem 1 -- no reserve arithmetic to reconcile, no interaction with the tap loop
-- and it makes `apply_snake_beta` simpler: drop the `pop_front(2)` and take the CB id from a define.

Cost: a new `Conv2dCb` enum value and its CB entry (2 files), the define in the sharded factory
(1 file), one read in the weights reader outside the block loop (1 file), the kernel change (small),
and host-side tiles. No op-signature change, no stride math, no arg re-indexing.

**Checked against the codebase's own conventions (Glean), which confirmed the shape and added one
risk.** The pattern is exactly as assumed -- `Conv2dCb` is an enum ending `OUT, COUNT` in
`conv2d_op_program_factory_common.hpp`, and a CB is added by appending an enum value and one
`cb_info.emplace_back(CBInfo{.name = ..., .num_pages = ..., .page_size = ..., .data_format = ...})`
in `get_cb_info`, exactly like `Conv2dCb::WEIGHTS`. Nothing bespoke is required.

The risk is `post_conv2d_op_memory_checks()` (conv2d_op_program_factory_common.cpp, ~line 749), which
validates the summed CB footprint against L1 and **is a known source of hard failures when a CB size
changes** -- see tt-metal issue #35207, where a 64->132 byte page-size bump on READER_INDICES tripped
it with a 16512-vs-16580 mismatch. Two extra tiles is small, but the audio tail already runs close
enough to L1 that `conv1d_l1_full_mode` measured only 1 of 42 shapes fitting L1_FULL, so this check is
a live failure mode here rather than a theoretical one. Budget a run against it, and if it trips, the
CB should be sized from the actual channel count rather than the block width.

**The two earlier commits on this (00553d225e3, 7aa264d7b53, b0b7c2ec862) reached the wrong carrier.**
The stride analysis in b0b7c2ec862 is still correct and still useful -- appending to the weight matrix
genuinely is safe -- it just is not the right place to put data with this lifetime.

### Correction to the correction: appending rows does *not* disturb the strides

The section below claimed appending parameter rows to the weight tensor shifts every block's
addressing. **That is wrong.** Reading the weights reader
(`reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp:216-246`):

    weight_row_start_tile_id = weight_current_block_start_tile_id + weight_h_offset
    for h in weight_block_height_ntiles:
        weight_tile_id = weight_row_start_tile_id
        for w in weight_block_width_ntiles:
            read page weight_tile_id
            weight_tile_id += 1
        weight_row_start_tile_id += weight_stride_h
    weight_current_block_start_tile_id += weight_next_block_stride_h

with `weight_stride_h = weight_matrix_width_ntiles` and
`weight_next_block_stride_h = weight_matrix_width_ntiles * weight_block_h_ntiles`. **Both strides are
functions of the matrix's width and the block height -- neither depends on the matrix's total
height.** And `weight_matrix_height` appears exactly twice in the program factory: where it is
assigned, and in a `% TILE_HEIGHT` assertion. Nothing else reads it.

So appending two tile-rows is safe: strides unchanged, the divisibility check still passes (64 % 32),
and the appended tiles occupy page ids past the last block where nothing currently looks. Better
still, the weight matrix's width *is* the channel axis, so one appended tile-row is exactly one
per-channel vector, and the tiles a given block needs sit at the same column offset that block already
computes -- `param_row_start + weight_col_offset + weight_tile_w_i`.

The error was conflating "the matrix gets taller" with "the strides change". They are independent.
The original 2-3 file estimate stands; what follows below overstated the cost.

### Superseded: "2 files" was too optimistic

That estimate came from reasoning about the route, not from reading the CB machinery. Reading it:

    conv2d_op_program_factory_common.cpp:207   Conv2dCb::WEIGHTS  num_pages = weight_block_num_tiles
    sharded_program_factory.cpp:477            weight_block_h_ntiles = act_block_h_ntiles
                                                                       * (coalesce ? filter_w : 1)
    sharded_program_factory.cpp:955            compute arg: weight_matrix_width_ntiles * weight_block_h_ntiles

The weights CB is sized and addressed **per block**, not per tap, and the reader walks the weight
matrix with `weight_matrix_width_ntiles * weight_block_h_ntiles` as a stride. So appending rows to the
weight tensor does not simply add two tiles at the end of what the kernel sees -- it shifts the stride
every block is addressed with, and the appended tiles land nowhere any block expects them.

Carrying the parameters in the weights CB therefore needs, at minimum:

1. program factory -- CB page count, the compute arg, and the stride
2. the weights reader kernel -- push the two extra tiles per block at the right offsets
3. the compute kernel -- pop them on the last tap (written)

plus host-side placement that matches the reader's block addressing rather than a plain append. Three
files with non-trivial stride math, not two with an append. **The optional-input-tensor route is worth
re-comparing on those terms** -- it is more files but each is mechanical, and a dedicated CB has no
stride to disturb. The all-or-nothing objection to it still stands; the choice is now genuinely close
rather than clear-cut, and should be made by whoever picks this up with a full build cycle available.

Remaining, in order:

1. Decide the carrier on the corrected comparison above.
2. Program factory: size the CB, emit `SNAKE_PARAMS_CB_ID`, fix the stride. Rebuild (~5-10 min
   incremental for one .cpp plus link; the 40 min figure elsewhere was a cold build of a fresh tree).
3. Reader: place the parameter tiles per block.
4. Host: build the alpha and inv_beta tiles -- per-channel value replicated down all 32 rows,
   `inv_beta = 1/(beta+eps)` precomputed.
5. Verify against `band_stencil_cpu.py`'s float64 golden at rel_rmse ~1e-07, matching what GELU
   achieved through the scalar seam. Expect a cycle or two purely on CB bookkeeping: the two accumulate
   paths consume in1 differently (one tile per tap non-coalesced, whole block coalesced), so the push
   and pop counts must agree in both.
6. Then the chunk-wise band mode on top, which is where the op-count win actually is.

## Cost, honestly

Step 2a was priced at ~20-30 ms (`fuse_saving.py`) and is *not* where the 5x is. This op is, because
it removes the slicing wrapper rather than one activation. But the projection rests on a fused
invocation costing about what a conv invocation costs today, and per `op_floor.py` the floor is 180 us
regardless of work, so that assumption is better supported here than anywhere else in this plan.

Expected: ~53 ops/band -> ~4, i.e. ~6700 ops -> ~500 for the band portion, against a measured floor of
180 us/op. That is the only remaining route to 60 ms.
