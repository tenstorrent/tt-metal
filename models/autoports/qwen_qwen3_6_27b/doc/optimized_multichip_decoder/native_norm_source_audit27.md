# Native minimal RMS / production decode owner: independent source audit

2026-11-13, source and existing artifacts only. No hardware, build, TTNN import,
or production edit. This is not a stage-review verdict. Scope: applied
`rms_allgather` diff, `worker_sync_utils.hpp`, current `tt/fused_decode_norm.py`,
norm routing in `multichip_decoder.py`, and validation25/owner26/build25 evidence.

## Finding requiring a bounded admission fix

`RMSAllGatherDeviceOperation::validate_on_program_cache_miss` validates
`args.output_mem_config`, but never checks `tensor_args.preallocated_output`.
`create_output_tensors` returns the caller's supplied tensor directly;
`compute_output_specs` also returns its own specification rather than an
independent expected specification. The factory then reads that actual
output's shard spec, dtype, buffer and layout to construct dynamic CBs and
write-back routes. A supplied tensor with incompatible shape, memory, shard
grid, dtype or device can therefore bypass the documented output contract.
An interleaved supplied tensor reaches `output.shard_spec().value()` rather
than a meaningful admission error. Wrong capacity is more serious than merely
a poor error message because native output CB sizing derives from the input
block geometry.

This is a native public-API admission gap, **not a demonstrated failure of the
selected production path**. `FusedDecodeNormOwner` constructs the expected
same-mesh BF16 TILE matching-shard outputs and shared-backing logical views.
The12 validation25 negatives concern stats/links and do not test persistent
output misuse. Parent requested a minimal unapplied patch plus extended native
admission tests. Required checks: allocated device output, same device, TILE,
requested dtype, requested memory/shard configuration, input logical shape,
expected padded shape, and standard output tiles. Preserve the legitimate
resharded-output calculation rather than requiring every output to equal the
input shard spec. On cache hits the adapter falls back to the same validator,
so one central check covers both admission paths. A new native rebuild and
focused hardware negatives/healthy rebound owners remain necessary.

## Inspected selected-path contracts

**Stats/transport.** The repaired global tile map is `rank/P` to stats core
and `rank%P` to local page. Selected TP4 stats have P4 on one sender core,
therefore every rank writes its unique local tile0..3 on the same logical
sender of every peer. New guards require one link, one stats core, correct
sender coordinate, FP32 stats when FP32 accumulation is enabled, TILE,
WIDTH_SHARDED L1, row-major shards, `[1,1,32,128]` padded stats shape and
sufficient full-tile capacity. The factory additionally bounds its producer
interval. Validation25 proves all12 expected messages before the first fused
launch, unchanged program-cache count, then healthy B1/B32 and fresh gamma/
input/stats/output/semaphore bindings. Generic API errors did not count.

The broader op still has inherited shape/factory assumptions beyond this
production envelope (nonrectangular/two-stage grids, nonstandard input tiles,
arbitrary alias patterns and output resharding). This audit does not certify
all of them. The selected wrapper uses a rectangular8×1 grid, standard32×32
tiles, distinct buffers, fixed one-link minimal norm and matching output
shards. The requested output guard closes a concrete exposed hole without
asserting that every other public configuration has now been tested.

**Compute phase.** The removed mid-kernel `compute_kernel_hw_startup` would
touch sender-only stats/variance CB metadata on non-sender workers and restart
the DST lifecycle. The repaired post-stats reduce is inside the sender-only
branch and uses `INPUT_AND_OUTPUT`, then the typed multiply chains configure
their own inputs/output for normalization and FP32 gamma. Other workers enter
the normal multiply path after their own valid global reciprocal RMS arrives.
The initial startup remains once per branch. No extra full-model conversion
or collective was inserted to hide precision differences.

**Physical row offsets.** `compute_core_layout` derives width/height from the
actual shard bounding box and records its origin. The factory first builds
relative sender/reduction ranges, then applies the origin; transport stats
coordinates and output storage routes use `worker_core_from_logical_core`.
Multicast endpoints include `start_core` before physical conversion. The
production sender is `(0, layer_idx % grid.y)`, with input/output cores0..7 on
that row and stats on its first core. It is not hard-coded to row0.
`minimal_norm_offset25_r1_b1/b32` and `r9_b1/b32` all pass with watcher and
allocation tracking, including trace/rebind checks. They complement the
row0 tests. Slots for64 layers distribute across available rows, rather than
accumulating all sender stats in one core. This is tensor-storage accounting,
not proof of arbitrary concurrent64-layer programs or every future L1 peak.

**Every-rank binding/cache identity.** The factory creates one workload
descriptor per mesh coordinate. Axis1 and mesh coordinates derive the rank
and forward/backward fabric peers separately for every rank. The cache key
includes epsilon, output memory, subblock/block widths, grid, compute config,
dtype, topology, links, ring size, axis, NoC policy, subdevice cores and tensor
specs. The semaphore address is intentionally excluded. The cache callback
rebinds semaphore, stats and gamma runtime words in every writer kernel and
all globally allocated input/residual/output/stats CBs by semantic role.
It visits all program runtime core entries; it does not patch rank0 alone.
The selected mesh allocator's shared addresses are the intended contract;
this does not support independently invented per-rank buffer addresses.

Owner26 tests both layer kinds and both norm families, two independent owners,
B1/3/17/32, changed inputs, poisoned original gamma, new gamma bindings,
untouched old storage,50 queued pairs per owner/batch, multiple batch traces,
and eager interleaving. Per-rank backing records and exact projection outputs
test stale-binding errors that simple norm PCC alone could miss. All pass.
`native_norm_source_audit27.receipt.json` compares the18 build25 pinned files
against current files. Owner26 records loaded native hashes matching the
build25 `_ttnncpp.so` receipt; this is evidence of the repaired binary, not
just the Python source being edited.

**Production owner/routing.** Installation occurs after packed-correction
setup, so canonical norm→matmul input memory references the actual selected
joint projection. Input/post are separate resource phases with separate
stats, persistent norm/output buffers and CCL semaphores. Projection keys
cover joint GDN or packed full attention and packed MLP. Construction rejects
noncompact/non-DRAM-sharded/unpacked MLP configurations, true minimal-SwiGLU
decode (which wants a different canonical layout), and non-BF16 CCL policy.
FP32 affine values are retained when the minimal family creates row-major
gamma. There is no host readback in forward.

All32 batch views are created at setup, checked to share each rank's base
allocation, and retain padded capacity32. Forward changes no persistent
device-buffer map. `warmed_batches` is a bounded host set, not a shape-keyed
device cache. Norm output is resharded/gathered to the projection's canonical
layout and the residual entering/leaving the decoder remains hidden-sharded.
The activation all-gather is inside the norm/projection boundary, not an
inter-layer restoration of replicated residuals.

## Concurrency, failure and documentation limits

The supported contract is complete ordered decoder calls on **CQ0**. No
concurrent queues, threads, overlapping same-owner partial phases, or independent
rank scheduling are supported by the wrapper. Its host phase latch validates
input then post with one batch and sets an invalid state before dispatch,
preventing reuse after an exception inside normalization. It is not a device
lock and does not run on trace replay. Trace safety relies on capturing full
phase pairs and replaying complete pairs in the same CQ order. Owner26
explicitly exercises that supported sequence and independent owners.

Native transport waits for remote stats credits, resets its global semaphore,
signals local workers and closes fabric traffic; the production activation
all-gather has an explicit barrier and separate input/post semaphore sets.
Sequential CQ0 operations plus that phase separation are necessary. Do not
infer safety for simultaneous same-owner CQ1 execution from50 queued CQ0
iterations. The owner API exposes no CQ argument; a future CQ or serving
scheduler extension must revisit buffers/semaphores and capture ordering.

The comment claiming a “failed partial forward remains unusable” is wider
than the actual latch scope: it guards failures inside normalize and between
input/post norms; after post normalize completes the host latch returns to0,
even if a later MLP operation fails. This does not establish a present unsafe
CQ0 reuse, but documentation should describe normalization-phase failure
rather than promise whole-forward transactional recovery.

`fused_decode_norm.py` still begins “Unapplied runtime candidate; patch
installs...” although it is now applied. This stale docstring should be
updated during final documentation cleanup. No behavior change is needed.

The shared `worker_sync_utils.hpp` diff selects a different full-slice schedule
only under `LINEAR_FULL_SLICE_V1` in `MatmulOpReceiver`. RMS kernels do not
instantiate that receiver; these edits do not alter minimal RMS stats routing.
The added unguarded positive-block assertion belongs to matmul validation and
its separate AGMM proof. This audit does not substitute for that family's
existing tests or final stage review.
