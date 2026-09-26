# AutoDebug: TP4 expert projection performance

2026-09-26. Source-only investigation; no device execution or implementation edits
were made for this report. The parent separately investigates projection configs
and profiler accounting. This report diagnoses measured v0/v1, not later v2 edits.

## Starting evidence

- `runtime_v0.py.txt` is the implementation behind `profile_v0/`.
- `profile_v0/decode_table.txt` identifies gate/up as active=8/128,
  32x2816x384, 12 cores, about 54 us, and down as active=8/128,
  32x192x2816, 8 cores, about 53 us. These are v0 observations.
- v1 keeps gate/up at 12 cores, moves down to 88 cores, and changes prefill
  K blocking from 1 to 11/6. Do not assign v0 down timing to this revision.
- `sliding_headline_v1.json`: median TP1/TP4 prefill host times are
  221200.545/225144.428 us; traced decode medians are
  825.485/898.609 us. Correctness passed; this is not a speedup.
- Source references below are repository-relative and identify symbols as well
  as approximate line numbers because the parent is editing the candidate.

## Findings and predictions

### 1. TP narrows the gate/up N dimension without distributing expert iterations

**Verified in source; performance causality still needs an experiment.**

`models/demos/gemma4/tt/experts/weights.py:70` pads intermediate width 704 to
768 for TP4, giving each rank width 192. The packed gate/up N is therefore 384,
or 12 tiles. `tt/multichip_decoder.py` uses a 6x2 gate grid with one output tile
per core. TP1's `OptimizedDecoder` defaults use width704, gate/up N=1408,
44 tiles, and an 11x4 grid. Every TP rank executes all eight selected experts.
Thus four chips supply only 48 gate/up workers versus 44 on TP1, and each worker
still traverses K=2816 for each of eight experts.

This is not a universal explanation of total latency: v0 also underutilizes
down, and attention, shared MLP, collectives and conversions remain material.
The sparse factory assigns work over M/N blocks, not the expert batch axis
(`sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:168-176,236`). Increasing
the gate grid alone cannot create more than 12 blocks for M=32,N=384.

### 2. Attention TP4 + MoE EP4 is feasible without a token-dispatch collective

**Verified at the source-contract level; device behavior unverified.**

The current public residual and expert input are replicated. Even the optional
hidden-sharded residual path gathers the normalized expert input before routing.
Keep the 128-way router replicated and unchanged. Each rank can own 32 complete
experts of width704, gather its 32 routing columns, compute only locally selected
experts, mix locally, and use the existing routed allreduce/reduce-scatter.
No token all-to-all is needed with this replicated input contract.

Candidate ownership is contiguous expert ranges [0,32), [32,64), [64,96),
[96,128). Upload gate/up `[1,128,2816,1408]` and down `[1,128,704,2816]`
using `ShardTensorToMesh(dim=1)`. Preserve packed `[gate,up]` order and the
baseline's quantization sequence/policies. Use a copied expert config with
`num_experts=32`; never change the router's 128-expert config. Local gate/up can
use 44 cores; local down can use 88 cores. K blocks must divide 88 and22,
respectively: the TP4 down block6 is invalid at full width704.

No expert weights are replicated across ranks. With equal dtype policy the
stored expert payload is 704/768 of the TP4 padded payload, excluding temporary
loader buffers. Avoid retaining both TP and EP weights in a capacity claim.

### 3. The existing indexed decode cannot simply be reused for EP

**Verified correctness hazard.**

`OptimizedExperts._chunk` (`tt/optimized_decoder.py:193-240`) uses either global
router indices, or `nnz=config.top_k`. Neither is correct for variable EP counts.
Local counts range from 0 through8 and can differ between ranks and trace replays.

- Global IDs are out of range for a 32-expert local weight tensor.
- Indexed mode derives a fixed loop count from `indices.logical_volume()`, ignores
  sparsity values, and executes every supplied ID. Padding to eight IDs with
  zero weights still executes inactive experts and is not the requested active
  expert path. Empty indices are also not a valid zero-count design: kernels
  use `num_active > 0` to enable indexed mode.
- Nonindexed `nnz=8` is an exact-count contract, not a maximum. On a rank with
  fewer than eight nonzero entries it can deadlock. `nnz=0` is rejected.
- Therefore the minimal candidate must omit **both** `indices` and `nnz`, and
  retain fixed 32-slot outputs. Do not only disable `enable_indexed_decode`:
  the inherited fallback still supplies `nnz=8`.

Evidence: sparse op validation at
`ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp:177-180,223-268`;
indexed reader behavior in
`device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp:298-315`;
factory `get_batch_from_reader` and `num_batch_compute` at lines94,292.

### 4. Dynamic mask scan has a source-supported zero-active path

**Source supports it; explicit 0/1/8 device test required.**

With no `nnz` and no indices, the factory sets `get_batch_from_reader=true` and
loops over all32 sparsity slots. In0 sender passes validity to every TRISC and
multicast receiver; invalid slots skip input transfer. Receivers pass the same
validity and skip data waits; compute skips its matmul loop; the weight reader
skips the weight block. All-zero masks therefore perform the validity scan and
no expert matmuls. Allocated expanded outputs are zero-filled before launch.

Evidence: in0 sender `:192-225`, in0 receiver `:44-66`, compute
`bmm_large_block_zm_fused_bias_activation.cpp:270-279`, in1 reader `:309-315`,
and sparse op `create_output_tensors`. Existing random sparse tests generally
leave at least one nonzero; this investigation did not find a proof of the
all-zero exact model shape in the existing tests.

Keep sparsity BF16 row major, because kernels inspect uint16 entries. A zero
local result must still participate in the same collective sequence on all
ranks. Do not branch on host readback or omit a collective on an empty rank.

### 5. Rank-local route gathering and prefill union need explicit shapes

Prepare a device-owned global-ID tensor by uploading `[0,...,127]` and sharding
its last dimension across ranks. Each local decode index tensor is
`[1,1,1,32]`, containing that rank's global expert IDs. These are **gather IDs**
into the replicated 128 routing columns, not sparse matmul IDs.

`ttnn.gather` output has the index tensor shape and does not broadcast token
rows (`operations/data_movement/gather/gather.cpp`,
`device/gather_device_operation.cpp:52-113`). For a32-token prefill chunk,
provide `[1,1,32,32]` indices with the ownership row repeated32 times. Gather
inside the existing 32-token chunk loop, or prepare every supported token-row
shape before capture. A one-row index tensor silently returns only one row.
Input and index must share TILE or ROW_MAJOR layout. The current C++ supports
row major despite the older Gather.md limitations paragraph.

For prefill, preserve the existing union semantics: sum the **local** routing
weights over32 token rows, convert to row major, and use dynamic sparsity for
both gate/up and down. Each expert in that local union processes the chunk;
per-token routing multiplication zeros the rows that did not select it. Keep
the original per-expert scaling and global top8 normalization; never normalize
the remaining local weights again. Route values are nonnegative under the
baseline policy; if that policy changes, union must use an explicit nonzero mask
instead of relying on sums to avoid cancellation.

## What could refute the expected speedup

Average ownership count2 is not the critical path. Under uniformly sampled
distinct top8 experts, a CPU combinatorial calculation gives expected busiest
rank count3.4859 and per-rank zero-count probability9.2747%. Real routing need
not be uniform. Record local counts and use busiest-rank time plus collective
completion. Test a skewed8/0/0/0 case as well as balanced2/2/2/2.

EP down K grows from6 to22 tiles. With equal88-core grids, approximate down
work on the busiest rank scales as `max_local_count*22/(8*6)`; at3.49 this is
about1.60, before overhead. Gate may improve while down regresses. EP also
expands decode output slots from8 indexed to32 masked, adds zero-fill and
elementwise traffic, scans32 validity entries, and adds a route gather.
Measure the entire local expert branch and collective, not gate alone. No
end-to-end improvement follows from the source analysis alone.

## Minimal discriminating experiment and implementation boundary

1. Add an opt-in `expert_parallel=False` factory option; preserve TP4 default.
   Put a small EP expert adapter/loader in `tt/multichip_decoder.py`. Avoid
   changing shared `OptimizedExperts`, router, attention, cache, normalization,
   collective topology, precision policy, or the public residual contract.
2. Add a stage-scoped component probe with real H2816/I704/E128/top8 weights.
   Test local counts0,1,8 with global patterns8/0/0/0,1/1/1/5,2/2/2/2 and
   rotated ownership; compare local weighted sums and allreduced outputs to a
   CPU/TP reference using the same input and routing. Require finite outputs
   and exact zero for an empty partition. Probe prefill unions including an
   empty rank and different experts across token rows. Run with bounded timeout
   and watcher before timing; a hang requires triage evidence before reset.
3. Warm fixed shape signatures, capture once, then update only routing/input
   buffers to switch ownership patterns, including0->8->0 on the same rank.
   Verify replay matches eager and no output from earlier active experts leaks.
   This tests runtime sparsity and full zero-fill independently of the router.
4. Time full expert branch plus unchanged collective for TP4 and EP4 using
   fixed real-route fixtures. Keep v2 attention/QKV settings fixed for decoder
   A/B. Run the original paired4096/128 command only after component correctness.
   Native profiling must identify dynamic sparse execution and per-rank imbalance.

Keep EP only if it improves the intended complete workload without correctness
loss. Even a winning component experiment does not complete Stage04: full
context262144, stack composition, cache ownership, logical batch, both attention
types, watcher, trace and capacity gates remain required. EP leaves the attention
TP4 cache layout and 1x4 FABRIC_1D Linear topology untouched; claiming this is the
best practical topology still needs comparative evidence.
