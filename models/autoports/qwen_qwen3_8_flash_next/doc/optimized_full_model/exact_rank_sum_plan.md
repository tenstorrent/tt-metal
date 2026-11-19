# Exact local rank-sum candidate — CPU preparation only

The test-only candidate is `tt/exact_rank_sum.py`, with three
`tt/kernels/exact_rank_sum_*.cpp` kernels and
`tests/probe_exact_rank_sum.py`. Production `Collectives.sum` and its synchronous
BF16 all-gather are unchanged. No device run, native compilation, latency or
full-model equivalence is claimed by this preparation.

## Source proof and preserved arithmetic

The selected path in `tt/multichip_decoder.py::Collectives.sum` is synchronous
BF16 all-gather(dim=1), FP32 typecast, accurate generic sum(dim=1, keepdim=True).
For local `[1,1,Q,2560]`, Q=1..32, the gathered shape is `[1,4,Q,2560]` and
physical rank-plane stride is 80 BF16 tiles.

1. `ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp:639`
   clones/fills input padding for Q<32. The outer-axis lowering at line209
   transposes HC, reduces H and transposes back. FP32 bypasses fast_reduce_nc.
2. `ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp:95`
   copies only logical token rows. Its line160 padding block writes the new
   rank-axis padding. For every live token/feature, the reduction tile contains
   rank0..rank3 in rows0..3 and positive zero in rows4..31. Poison in original
   token padding cannot reach a live output.
3. `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl:629` has Ht=1, so the
   first tile is copied without a tile fold; the accurate SFPU H finalizer is
   called at line664. The scalar is one, with no post-multiply effect.
4. Blackhole
   `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h:104`
   loads rows j,j+4,j+8,j+12 and j+16,j+20,j+24,j+28 into the two register
   groups. The replay initialized at line1366 produces
   `upper_j = rj + (+0 + (+0 + +0))` and
   `lower_j = +0 + (+0 + (+0 + +0))`.
   The line245 cross-face addition makes `u_j = upper_j + lower_j`.
   Transpose then the half replay produces `u0 + (u1 + (u2 + u3))`.

All zero-only subexpressions yield positive zero. The candidate therefore
retains **two explicit FP32 SFPU positive-zero additions per rank**, followed
by the identical right-nested rank sum. It does not replace this with a left
fold, all-reduce, or a mathematically equivalent reordered expression. The
FP32 binary SFPU implementation in `ckernel_sfpu_binary.h:104` loads operand0
and operand1 separately and computes `in0 + in1`; the params wrapper preserves
operand indices. Each intermediate remains FP32. BF16 unpack widens the
already-BF16 rank operands; the candidate returns FP32, and the probe retains
the caller's unchanged multiply0.25 and final BF16 cast.

This is a source argument for finite operands, including signed zero and
cancellation; it is not a hardware bitwise result. Extra FP32 DEST stores in
the binary-tile API do not narrow finite results. Live NaN payload equivalence
is not claimed. NaN/Inf in unused token padding is explicitly tested.
The module rejects non-Blackhole hardware, non-TP4 meshes, nonstandard tiles,
non-DRAM placement and shapes outside this exact geometry.

## Ownership, lifetime and measurement

There are 80 output tiles, each owned by exactly one worker. For feature tile
t, the reader reads gathered tile addresses `r*80+t`, r=0..3, into one four-tile
BF16 CB. The compute kernel uses three FP32 DEST tiles: accumulator, operand,
and a positive-zero tile initialized with `fill_tile`. Output is one complete
FP32 tile, with no partial writes or cross-worker sharing. Total CB storage
is16KiB/core. Arithmetic is elementwise across token rows; padding poison
cannot enter live rows. The collective itself is outside this local candidate.

The baseline/candidate timing includes their entire local post-gather paths
and common multiply/cast. It excludes the unchanged collective and makes no
network saving claim. The historical4.679354ms weighted post-gather budget in
`AUTODEBUG_functional_floor.md` is a zero-cost envelope, not an expected win;
this candidate overlaps clone/fill and transpose-removal budgets completely.
Because this implementation retains the common multiply/cast (historical
0.693393ms), its replaced-operation zero-cost ceiling is only3.985961ms.
The native kernel still has nonzero execution and data-movement cost.

The probe runs all Q1..32, four independently populated mesh devices, random
and rank-permuted exponent-gap/cancellation/negative-zero operands. Stress
live values are repeated with finite137, NaN and Inf padding. It verifies
padding was actually poisoned, compares raw FP32 bits and downstream BF16
bits, checks live-output padding invariance and physical input immutability.
Q1 and Q32 also compare against actual unchanged synchronous
`Collectives.sum`/gather. Each component receives12 changed-input replays and
three64-replay nonblocking timing windows, with only end-window synchronization
and correctness readbacks outside timing. Tracking runs include a retained
post-capture allocation positive control and acknowledge only the two trace
outputs. Only the probe's current trace is released. Runtime audit guards
eager, warm and capture bodies. Phase prints precede first native submission
and capture; source hashes include the generic reduction, transpose and LLK.

CPU checks passed: Python compilation/Black;256 pure-FP32 fixture comparisons
of the full32-row LLK expression against the candidate tree;172 cases
distinguish an incorrect left fold, and128 distinguish omission of zero adds.
Static ownership checks cover all80 output tiles and320 gathered input tiles
exactly once for worker counts1/8/64/80. These checks do not compile the native
kernel or establish device behavior.

## Owner-controlled commands

Run only after the current hardware owner releases the mesh. Projection and
residual-write experiments remain higher priority. From the checkout root,
after sourcing `doc/optimized_full_model/environment.sh`:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=0 timeout 1200 python_env/bin/python \
  -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_exact_rank_sum \
  --clean-timing \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/exact_rank_sum_clean_v1.json
TT_METAL_TRACE_ALLOC_TRACKING=1 timeout 1200 python_env/bin/python \
  -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_exact_rank_sum \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/exact_rank_sum_tracked_v1.json
```

Watcher must be a separate run under the hardware owner's environment.
Successful component checks would still require full-layer L0/L1/L3
equivalence, state/trace gates and new full-model evidence before adoption.

## Poison fixture repair after tracked v1

`exact_rank_sum_tracked_v1.json` failed before the first native submission at
the poison assertion. The v1 assertion did not record the failing family or
observed values, so the artifact cannot distinguish an implicit-padding issue
from NaN transformation. Source `ttnn/core/tensor/py_to_tt_tensor.cpp:82`
permits device tilize/typecast unless `preserve_nan_values=True`; its explicit
comment documents the NaN-preservation restriction. The original fixture left
that flag false and relied on `from_torch(pad_value=...)`.

The repaired fixture writes every physical row explicitly on CPU, uploads
aligned BF16 data with `preserve_nan_values=True`, then uses public two-shape
`ttnn.reshape(backing, logical_shape, padded_shape)` to expose logical Q with unchanged padded32
storage. It asserts unchanged device buffer addresses and reads back all
physical bits. Before the poison assertion it prints and records each family,
rank, physical shape, expected poison, observed BF16 bit values/counts,
NaN/Inf counts and exact full-buffer comparison. Both live values and poison
must survive exactly. Changed-input trace copies also verify the persistent
physical input once for each poison family outside timing.

The repair passed64 host-only TTNN construction/view/readback cases covering
Q1/3/17/32, four devices' independent fixture contents and all four poison
families, with exact live and physical BF16 bits. No mesh was opened for those
checks. Device upload, native compilation and replay remain for the owner's
new tracked artifact; v1 remains unchanged.

Actual passing device fixture is v4; v2 one-shape reshape failed device volume validation and v3 required FABRIC_1D_RING setup. See exact_rank_sum_tracked_v4.json, exact_rank_sum_clean_v2.json and exact_rank_sum_watcher_v1.json. The earlier plan records intentions, not test results.
