# Local kernel and stack contracts

The selected implementation changes head ownership and activation layout,
not the model's batch or sequence semantics. The single-chip optimized module
is imported unchanged. Setup is the only host weight conversion path.

## GDN

Rank r owns K heads4r..4r+3 and V heads12r..12r+11. Each K head serves three
adjacent V heads. The packed convolution row is Q512/K512/V1536 BF16 values,
5120bytes. The local reader uses byte offsets0/1024/2048; recurrence state is
[B,12,128,128]. Core partitions can begin inside a three-head group, so Q/K
ownership uses the absolute local V-head index divided by3. Output and state
writers use local, not global48-head strides. Source partitions are checked
across630 cases in `host_validation.json` and device cases in
`compaction_local.json`.

For decode B>1, the activation is compact[1,B,C], while recurrence beta/g is
[B,1,12] with physical[B,32,32]. Four convolution tap products and their sum
are FP32, rounded to BF16 before SILU; the shifted ROW_MAJOR history remains
[B,3,2560]. B1 uses the baseline native convolution. Prefill uses the native
chunk GDN and retains FP32 live recurrence; final persistent state is BF16.
Readout and normalization preserve the optimized baseline's rounding order.

## Full attention

Packed local QKV has64 tiles: Q6*256, K256, V256. Staging assigns one user
per core in an8-column grid, outputs logical[1,B,6,256] Q and[1,B,1,256] K/V,
with physical32 head rows/core and height-shard[32,256]. Both RISC writers
zero their complete512-byte faces before64-byte aligned reads and32-byte
row scatters. Q/K/V rank, user, head, channel and padded-zero values are
checked exactly in `qkv_staging.json`, including user15/16/31 boundaries.

Q/K RMS norm is per256-wide head. RoPE touches64 channels, preserving the
remaining192. Cache updates use separate K/V core ranges; native local-head
paged SDPA consumes one KV head for six Q heads. Concatenation yields six
heads and a physical batch32; the logical batch is cropped before output
gating and the row-parallel output projection.

Caches are BFP8[physical_pages,1,32,256] per chip. The int32 ROW_MAJOR page
table and current position vector are replicated across ranks. Page IDs are
physical block addresses, not inferred from user order. Owned padded pages
are finite zeros. Tests reorder physical pages and matching data, change
per-user positions and RoPE, and verify the traced result against eager.

## Layer boundaries, collectives and trace

Every public layer input/output is BF16 TILE[B,S,1280] on each rank, sharded
along hidden5120. Internal decode[1,B,C] compaction restores that shape.
Each row-parallel projection sums partial contributions using Linear axis1
reduce-scatter. Prefill fuses matmul/reduce-scatter only for512<B*S<=8192 and pads only
internal rows; the following residual/norm consumes the sharded result.
No host conversion or replicated boundary adapter is needed between layers.

Norm may gather first or gather distributed RMS statistics, depending on the
measured phase policy. It then supplies a replicated normalized activation
to column-parallel projections. Two fabric links are used; global CCL
semaphores cover the full11x10 native worker-selection grid. Trace capture
keeps fixed output/CCL buffers alive; runtime tensor conversion sentinels and
allocation tracking include program-cache allocations. No trace allocation
acknowledgment or blanket exemption is installed.

Native chunk512 is internal. Logical lengths1,31,33,513,1025 and near-maximum
262143 are valid; finite load/prefill padding is sliced before public output.
The delivered stacked test chains checkpoint layers0..4 (GDN/GDN/GDN/full/GDN)
with independent states and checks shape/dtype, prefill/decode PCC and final
output equality for one original-input trace replay. Changed-input and full-state
trace equality are checked separately in the single-layer and64-owner tests.


## Narrow corrected projection

The GDN primary logical width4160 contains QKV2560, Z1536, and two32-wide
padded gates. Correction width2624 contains QKV2560 and BA64 only. Its DRAM
physical width3072 is independent of the primary4608 stride. The native fused
combine reads both true padded strides and emits4160 BF16 columns. It adds
correction to QKV/BA in FP32, preserves the primary Z, and rounds once to BF16.
Decode intermediates are L1 width-sharded FP32; prefill uses DRAM. Kernel
reader/compute source is specialized from the unchanged optimized baseline,
with exact integer-token occurrence assertions. `narrow_v1_*` checks the
wide/narrow and DRAM/L1 combinations on recorded activations. Final correctness,
stress and watcher use the selected narrow+L1 implementation.

## 64-owner resource capacity

`delivery_resources_b{1,32}` own all64 CCL managers and all48 GDN convolution
exchange/semaphore sets. Distinct per-layer state seeds and complete all-rank
state hashes catch aliases. Prefill and both changed-input decode variants
execute before capture. Late program-cache misses are prohibited; trace
allocation tracking includes program cache and has a deliberate unsafe
allocation positive control. Three replays preserve every output/state hash
without recapture. L1 persistent allocation is58,368 bytes/bank at B1 and60,416 at B32; traces use
17,104,896 bytes(B1) and29,818,880 bytes(B32) within32MiB. Watcher is enabled
throughout these runs. Serialized execution on one queue remains the contract.

### Safe row-linear prefill dispatch

With sharded residuals and MMRS enabled, flattened M<=512 delegates to the
baseline TTNN row matmul and the existing reduce-scatter consumer; M>8192
also takes that path. Intermediate M is flattened, tile-padded, partitioned into at most128 rows per native MMRS,
concatenated and cropped back to[B,S,1280]. This partition is only linear
algebra; recurrent history, attention positions and KV page ownership retain
their original logical sequence. The native thin-wide plan must keep weight
reads on NoC0 on this fixed1x4 target. Both NoC selection and watcher behavior
are checked under persistent64-owner resources. The unpartitioned NoC1 native
path is not an accepted policy override. Replicated-layout controls use
ordinary matmul/reduce/all-gather. All phase weight copies remain included in
resident-memory accounting even when a small prefill bypasses MMRS.
