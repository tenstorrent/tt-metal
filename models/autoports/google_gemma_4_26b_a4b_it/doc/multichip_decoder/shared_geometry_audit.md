# Shared-MLP decode geometry proposal

Source/CPU preparation only, 2026-09-26. No runtime or runner edit, TTNN import,
device use, or latency measurement was performed. The optional
[shared_geometry.patch](shared_geometry.patch) applies to runtime SHA-256
`a102e006caea6b91bf233981a9f734c4a464ebb086a40bab110ff62e8700ac65`.
It was rebased onto the AGMM/topology options and bounded continuation helper.
Applying it produces SHA-256
`17376326fdf3d75c8710cc71cd66fd7cfb6c50a08762eeb80d9d9a29e1f9cb95`.
Apply this artifact before `grouped_moe_reduce.patch`; the latter now uses this
geometry candidate as its base. Both exact-base apply checks and AST checks pass.
It touches `_SharedMLP` and its factory option only, independently of collective
implementation. The parent must wire a runner CLI flag if it applies the patch.

## Optional 1D candidates

The factory option is `shared_geometry=0|1|2`, default zero (automatic config).
Nonzero values require `optimized_shared=True`; invalid values fail at setup.
Only the two single-token decode `linear` calls receive an explicit program
config. BF16 prefill projections, BFP8 sliding/BFP4 full decode weights, LoFi,
BF16 inputs/outputs, `fp32_dest_acc_en=False`, and `packer_l1_acc=True` remain
unchanged. No additional weight storage or activation conversion is introduced.

Current local dimensions are H=2816=88 tiles, padded shared I=544=17 tiles,
and packed `[up, gate]` N=1088=34 tiles. Global shared width is padded from 2112
to 2176 before its four-way TP split. Existing setup already performs this
padding and rank-local up/gate packing.

| Candidate/op | Grid envelope | Working cores | per-core M,N tiles | K block tiles | Output subblock H,W | K blocks |
| --- | --- | ---: | --- | ---: | --- | ---: |
| 1 gate/up | 11 x 4 | 34 | 1,1 | 44 | 1,1 | 2 |
| 2 gate/up | 9 x 2 | 17 | 1,2 | 88 | 1,2 | 1 |
| Both down | 11 x 4 | 44 | 1,2 | 17 | 1,2 | 1 |

Every config uses `fuse_batch=True`, `mcast_in0=True`, and explicit output block
equal to its per-core output dimensions. Grid envelope is an availability
limit, not a claim that all 44 or 18 cores perform gate/up work. The matmul
factory creates `ceil(M/per_M)*ceil(N/per_N)` working blocks.

`matmul_device_operation.cpp:493-565` checks that K divides the K block,
subblocks divide output blocks, output blocks divide per-core dimensions, and
the subblock fits destination registers. The proposed divisions are 88/44,
88/88 and 17/17; subblocks contain only one or two tiles. Lines 713-764 check
grid capacity and require one M block for multicast-in0. Both grids fit the
observed 11 x 10 available compute grid. The inputs, weights and output remain
interleaved, avoiding unrelated sharded-operand constraints.

The primary per-worker CB payloads below follow
`matmul_multicore_reuse_mcast_1d_program_factory.cpp:134-215`: input CBs are
double-buffered when the K loop has more than one block, otherwise single;
output/intermediate BF16 storage can alias for these unsharded outputs. BFP8
tiles occupy 1088 bytes, BFP4 576, and BF16 2048.

| Op | Sliding primary CB bytes/core | Full primary CB bytes/core |
| --- | ---: | ---: |
| Candidate 1 gate/up | 278,016 | 232,960 |
| Candidate 2 gate/up | 375,808 | 285,696 |
| Down, either candidate | 75,904 | 58,496 |

These totals cover input A, input B, and the aliased output/intermediate tile
buffers. They are not a complete live-L1 allocation report: resident tensors,
kernel/config storage, semaphores and other runtime allocations still need a
device check. None approaches the 1.5 MiB physical L1 budget by itself.

Changing K blocks also changes accumulation boundaries. The factory enables
effective packer accumulation only when `packer_l1_acc && num_blocks > 1`:
candidate 1 gate/up retains it, whereas candidate 2 and down naturally disable
it despite the unchanged requested compute policy. Existing automatic small-K
blocks may spill more often. Therefore same storage dtype/fidelity is not
proof of bitwise or PCC equivalence; check both real layer kinds before using
timings. Candidate 2 also sacrifices half the gate/up parallelism to reduce its
K-loop count. No speedup is established by this proposal.

CPU checks: proposed full Python source parses; all integer K/block, N/core,
grid and CB calculations pass; `git apply --check` passes against the recorded
base. Geometry 0 passes `program_config=None`, retaining automatic selection.

## Separate DRAM-width-sharded weight experiment, not implemented

The report's generic DRAM-sharded advice requires different weight placement,
width-sharded L1 activations and output, and matching storage geometry. Merely
changing the program-config type on the current interleaved tensors is invalid.
`matmul_device_operation.cpp:1302-1375` enforces these requirements, including
one physical M tile and a K block dividing each activation shard's K width.

For the target's eight-bank Blackhole configuration, the minimal-padding
one-reader-per-bank candidate is:

| Op | Per-rank physical weight K,N | DRAM shard per bank | L1 activation storage | Program per-core M,N | K block |
| --- | --- | --- | --- | --- | ---: |
| Gate/up | 2816,1280 | 2816,160 | 8 cores, each 32 x 352 | 1,5 | 11 |
| Down | 544,2816 | 544,352 | 1 core, 32 x 544 | 1,88 | 17 |

The number of activation/output storage cores is independent of the eight
DRAM reader/compute workers. Gate/up has K=88 tiles, storage K/core=11, and
N=40 tiles, storage N/core=5. Down has prime K=17 tiles: one activation storage
core avoids adding K padding, while the factory still assigns one reader to
each of eight banks, each computing 11 N tiles and writing into the single
88-tile-wide output storage shard. Output memory must be L1 WIDTH_SHARDED,
ROW_MAJOR orientation, matching the input storage core grid. Verify the real
bank count at setup; do not hardcode this geometry on seven-bank hardware.

Gate/up N needs bank padding from 1088 to `ceil(1088/(8*32))*8*32 = 1280`.
Append 192 zero columns **inside each rank's packed block**, after its
`[up544, gate544]`; then concatenate the four padded rank blocks into global
weight shape `[1,1,2816,5120]` before mesh sharding along N. Padding only the
end of the existing global packed matrix would misalign rank ownership.
After matmul, convert the output to interleaved L1 and crop to the original
1088 columns before the existing 544/544 split. Otherwise the current
`gu[..., self.width:]` gate slice would incorrectly include padding.

Down needs no additional padding: its existing mesh-level matrix is
`[1,1,2176,2816]`, sharded along K to `[1,1,544,2816]`; N=88 tiles already
divides eight banks. Reshard its BF16 hidden input onto the one storage core,
then convert output back to the current interleaved representation before the
existing collective. No mathematical reduction order or post-normalization
placement should be changed as part of this isolated experiment.

Using one activation storage core for down is a deliberate minimal-padding
control, not necessarily the fastest candidate. An alternative eight-storage-
core control would need K padded from 544 to at least 768 (24 tiles divisible
by eight), with matching zero padding of the hidden activation and weight rows.
That adds 41.18% down-weight tile traffic and should be a separately identified
experiment. Multiple DRAM readers per bank introduce further N padding because
each bank shard's tile width must divide the reader count; do not silently
reuse the single-reader layout.

The minimal-padding gate/up adds 528 weight tiles per rank: 574,464 bytes for
sliding or 304,128 for full, beyond current quantized decode weights. Replacing
the interleaved decode layouts with these DRAM layouts adds 15,882,240 bytes
over 30 layers. Retaining both layouts instead adds the entire new set and
requires different capacity accounting. The original BF16 prefill weights
remain retained in either experiment.

The DRAM factory's worker/storage distinction and output resharing are visible
in `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:120-167,742-887`.
This makes the outlined dimensions source-consistent, not hardware validated.
The imported Gemma4 shared-MLP helper at `shared_mlp.py:146-150` explicitly
disables its older DRAM-sharded MoE path after a reported full-layer PCC drop.
Accordingly this new candidate needs isolated real-weight GU/down comparison,
complete layer PCC, all-rank equality, trace replay and closure before any
whole-layer performance interpretation. Preserve the full-layer timing window
and compare against the same precision policy.
