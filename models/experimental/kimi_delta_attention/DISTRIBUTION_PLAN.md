# KDA work distribution plan

## Decision

Use one eight-device tensor-parallel line. Partition whole heads, keep sequence
replicated, and keep each head's complete FP32 `[K,V]` state on its owner.
There is no collective inside recurrence. Prefer fused output-matmul +
reduce-scatter when the caller accepts hidden sharding; retain all-reduce only
for a replicated-hidden boundary.

This preserves the distribution contract in `API_SPEC.md` and follows the
sparse-MLA convention that the model owns mesh/fabric setup while the layer
accepts configured collective resources.

## Tensor map at the target shape

For TP=8, device `d` owns heads `[4d,4d+4)`.

| Tensor/weight | Global | Per device |
|---|---|---|
| Hidden input | `[1,T,2304]` | replicated initially |
| Q/K/V projections | three `[2304,4096]` | output-sharded to three `[2304,512]` |
| Fused QKV activation | `[1,T,12288]` | `[1,T,1536]` |
| Conv taps/state | width 12288 | width 1536 |
| `f_a` rank | `[1,T,128]` | replicated; required by local decay outputs |
| Output gate | factorized `g_a/g_b` | aligned prefill precomposes `g_b @ g_a`, then output-shards `[1,T,4096]`; fallback keeps replicated rank |
| Beta projection | `[1,T,32]` | output-sharded `[1,T,4]` |
| `f_b` / `g_b` outputs | two `[1,T,4096]` | output-sharded to two `[1,T,512]` |
| Decay/bias/norm | 32 heads | four whole-head slices |
| Recurrent state | `[1,32,128,128]` | `[1,4,128,128]` |
| Pre-output activation | `[1,T,4096]` | `[1,T,512]` |
| Output projection | `[4096,2304]` | row-sharded `[512,2304]` |
| Output partial | `[1,T,2304]` | reduce-scatter or all-reduce input |

The fallback fused auxiliary projection is rebuilt per device as
`[f_a(full 128), g_a(full 128), b(local 4)]`, width 260. Aligned prefill uses
`[f_a(full 128), (g_b @ g_a)(local 512), b(local 4)]`, width 644. Both avoid
collectives before recurrence; the aligned form deliberately trades more
well-utilized input-GEMM FLOPs for one fewer small gate program.

## On-chip mapping

Prep has `local_heads * num_chunks` independent work items. At T=640, TP=8
gives `4 * 20 = 80`; assign exactly one item to each of 80 cores. The remaining
30 cores cannot take another independent item without splitting a 32-token
head/chunk and adding a K reduction. Do not add that communication until a
measured intra-item split beats the 80-core mapping.

Scan is sequential over chunks and parallel over heads and V blocks. The
single-chip H=32 result favored one full V block/core, but that result does not
transfer to TP=8. The measured T=640 crossover is:

| Local heads | Mapping | Scan time | Verdict |
|---:|---|---:|---|
| 4 | 4 V blocks/head, 16 cores | 96.132 us vs 148.009 us full-V | split, 35.1% faster |
| 8 | 4 V blocks/head, 32 cores | 138.286 us vs 149.206 us full-V | split, 7.3% faster |
| 16 | 1 full-V block/head, 16 cores | 153.672 us vs 260.952 us split | full-V, 41.1% faster |
| 32 | 1 full-V block/head, 32 cores | 181.788 us | full-V |

Production selection for K=V=128 is therefore:

```python
value_splits = 4 if local_heads <= 8 else 1
```

The H=4 split moves 26.739 MB and reaches 278.15 GB/s (54.33% DRAM roofline).
Its next kernel optimization is one DRAM read plus multicast of V-independent
prep tensors to the four V workers; the current reader redundantly fetches
those tensors four times. Do not increase V splitting beyond four because V has
only four tiles.

## Collective map and target

The standalone TP=8 all-reduce benchmark measured 219.169 us for a 2.949 MB
BF16 partial, or 23.5% of the two-link LoudBox fabric roofline. It misses the
40% aspiration and is not the final mapping.

Implementation order:

1. Land whole-head TP weights/state and local H=4 recurrence with the measured
   four-way V split.
2. Return hidden-sharded output through the existing fused
   matmul-reduce-scatter pattern where the caller permits it.
3. Provide a replicated-output fallback through configured all-reduce.
4. Reuse the sparse-MLA critical-path byte model and real-time profiler for
   both paths. Report slowest-chip program time.
5. Sweep a dedicated CCL core row/column only after the fused path exists.
   Reserve cores permanently only if overlap improves end-to-end layer time;
   the current evidence does not justify a fixed reservation.

The fused path now exists. At the target shape it uses an 8x8 matmul grid and
two reduce-scatter worker rows, reaches 31.1% of the effective two-link
roofline on the slowest device, and shows 140.349-166.069 us device-median
spread. Retain whole-head TP and the 80-core/16-core recurrence maps: measured
prep and scan are 84.295 us and 96.497 us and agree with their isolated
crossovers.

The controlled fused-layout sweep held Ring topology, two links, FP32 output,
and tensor ownership fixed:

| Matmul grid / RS offset | Slowest-chip median | Result |
|---|---:|---|
| 8x8 / `(0,8)` | 166.069 us | retain |
| 8x7 / `(0,7)` | 172.950 us | 4.1% slower |
| 9x8 / `(0,8)` | 191.184 us | 15.1% slower |
| 8x6 / `(0,6)` | 192.304 us | 15.8% slower |

Horizontal offset `(1,7)` is illegal for this CCL worker geometry and aborted
before yielding a timing. The safe wrapper recovered and reset all devices.

Trace capture removed the eager dispatch gap: ten target-shape replays have a
1.263 ms median slowest-device span and 1.213-1.216 ms median active
kernels/device, versus 5.749 ms eager. The retained map is therefore:

- TP=8 whole-head ownership, replicated sequence and low-rank inputs;
- 80 prep workers, one per local `(head,chunk)`;
- 16 scan workers, four V blocks per local head;
- 8x8 output matmul with two Ring reduce-scatter worker rows at offset `(0,8)`;
- model/layer-owned, mesh-replicated chunk constants so the full layer is
  trace-capturable without host uploads.

Under trace, the fused output program measures 148.023 us on the slowest device
and reaches 34.9% effective CCL roofline. Prep and scan remain 84.602 us and
96.252 us. This evidence rejects further recurrence-core redistribution as the
next step: the active-time ranking instead prioritizes untilize/tilize removal,
local matmul/layout fusion, then a fused-output kernel change capable of
crossing the 129.0 us CCL target.

The first layout reduction retains this map and replaces only aligned B=1
prefill convolution with native depthwise `ttnn.conv1d`; decode, short,
batched, and padded paths retain the general FIR. Target-shape trace latency
falls from 1.263 ms to 0.987 ms and active kernels from about 1.215 ms to
0.941 ms/device. The unchanged prep, scan, and fused-collective times confirm
that this is a local layout win, not a work-redistribution effect. Continue
with output-gate head-layout fusion before reconsidering core allocation.

Output-gate head-layout fusion is now measured: batched head weights emit
`[local_heads,T,V]` directly and reduce traced target-shape latency from
0.987 ms to 0.890 ms. The unchanged recurrence and collective medians again
retain the same 80-core/16-core/8x8 map. Further work should target program
fusion and the 148 us output collective, not redistribute KDA heads.

The input projection now uses the same ownership map directly: every physical
weight shard is `[Q_local|K_local|V_local|f_a|g_a|beta_local]`. Fusing these
columns reduces traced latency to 0.874 ms without a collective or ownership
change.

A controlled 1x3 fused-output subblock test also retains 1x1: slowest-chip
matmul + reduce-scatter regressed from 146.778 us to 147.706 us. The remaining
CCL gap is not solved by widening the local matmul subblock.

Fusing the aligned chunk decay bias into its projection reduces traced latency
to 0.855 ms without changing tensor ownership or core allocation. It removes
one binary program/device; softplus remains separate. This is further evidence
to retain the 80-core prep, 16-core scan, and 8x8 output/row-8 Ring
reduce-scatter map while pursuing launch and dataflow reductions.

An exact 16-core scan A/B also rejects sharing the six V-independent inputs
from one V worker to its three siblings. The L1-transfer version increased
scan time from 97.387 to 145.942 us and whole-layer latency from 0.85484 to
0.90400 ms. Keep four independent DRAM readers/head; reducing bytes alone does
not improve this synchronization-sensitive recurrence.

The output gate retains its local whole-head ownership and BF16 producer
precision through sigmoid. Its FP32 contract is applied by the consuming
multiply, eliminating one conversion program/device and reducing traced layer
latency from 0.85484 to 0.84802 ms. This does not justify any core or tensor
redistribution; retain the 80-core prep, 16-core independent-reader scan, and
8x8 output/Ring reduce-scatter map.

The decay-scale product likewise produces the prep-required FP32 tensor
directly. Removing that conversion reduces traced layer latency from 0.84802
to 0.84038 ms while prep and fused-collective medians remain 84.05 and
146.31 us. Retain the current map; this is launch-chain reduction rather than
evidence for redistributing recurrence or CCL workers.

The convolution wrapper now converts QKV and carry to row-major exactly once
before concat and derives the next carry from that row-major QKV. Traced layer
latency falls 0.84038 -> 0.70788 ms, while prep, scan, and fused output are
unchanged. This removes layout traffic without changing ownership: retain the
same TP and core map. Revisit a private tile-to-tile causal kernel only against
the remaining measured conv boundary.

The layer-owned convolution cache is row-major to match aligned prefill; the
legacy FIR fallback alone adapts it to tile layout. This removes two more
target-path programs and reduces traced latency 0.70788 -> 0.69876 ms without
changing tensor ownership, prep/scan workers, or collective placement.

Aligned prefill now precomposes `g_b @ g_a` on the host and output-shards the
direct gate columns in the fused input weight. It reduces traced latency
0.69876 -> 0.69072 ms and one program/device, while keeping each device's four
whole heads and adding no communication. Retain the 80-core prep, 16-core scan,
and 8x8 output/Ring reduce-scatter map.

Two CCL alternatives are now rejected. BF16 output partials fail correctness
at PCC 0.004862. Two workers/link require four CCL rows; the resulting
8x6/four-row placement regresses fused time 151.333 -> 166.374 us and layer
span 0.69072 -> 0.70600 ms. Keep FP32 partials, one worker/link, the 8x8
matmul grid, and two CCL rows at offset `(0,8)`.

Direct projected offsets for decay, gate, and beta remove one local slice
program and reduce layer span 0.69072 -> 0.68346 ms. This is another
launch/dataflow reduction with no ownership or core-map change.

Fusing the output-gate sigmoid into its consuming multiply removes one more
local program and reduces layer span 0.68346 -> 0.67934 ms. The approximately
147 us fused output collective is unchanged. Retain the four-head/device,
80-core prep, 16-core independent-reader scan, and 8x8 output/Ring
reduce-scatter distribution.

Sequence parallelism is rejected for this phase: prep would shard naturally,
but scan would need ordered state handoff at every sequence partition. TP
already removes weight pressure without placing a collective on the recurrence
dependency chain.

## Acceptance

- Eight-device output/state agrees with the same torch reference and
  single-device result.
- Local mapping is H=4, prep 80 cores, scan 16 cores at T=640.
- CCL benchmark reports payload, topology, link count, critical-path bytes,
  theoretical time, measured slowest-chip time, and utilization.
- End-to-end report separates compute, DRAM/layout, and CCL time.
- The 60% compute and 40% CCL goals remain aspirations; measurements are not
  renormalized when they miss.


The gated-RMS epilogue maps one work item to each `(head, 32-token block)`.
At the target shape this is `4 heads * 20 blocks = 80` independent work items
per device, each processing all four value tiles for its head. It writes
token-major output directly, removing separate RMSNorm and head-concat
programs and reducing layer span 0.67934 -> 0.66733 ms. It preserves
four-head/device ownership and leaves the 80-core prep, 16-core scan, and
8x8 output/Ring reduce-scatter distribution unchanged.


Fusing decay Softplus into its consuming scale multiply removes one local
program and reduces layer span 0.66733 -> 0.65896 ms. Prep, scan, and
collective medians remain 84.15, 97.31, and 148.78 us, respectively. This is
another launch-chain reduction; retain the existing head, core, and CCL map.


Scalar BinaryNg fusion of beta sigmoid and typecast is rejected: it reduced
program count but regressed layer span 0.65896 -> 0.69050 ms with unchanged
active time. It provides no evidence for changing prep ownership or core count.


At `T=5120` (160 chunks), head ownership and sequence replication remain
unchanged. Prep and the gated-RMS epilogue each expose 640 independent
`(head,chunk)` or `(head,token-block)` items/device. Their shared row-major
distributor uses all 110 compute cores: the first 90 cores receive six
contiguous items and the remaining 20 receive five. Scan remains 16 cores
(four V blocks for each of four local heads), with each core processing all
160 chunks sequentially. Measured prep/scan times are 335.954/677.885 us.

The output map also remains 8x8 matmul plus two Ring reduce-scatter rows at
offset `(0,8)`, FP32 partials, and one worker/link. Its 1043.920 us measured
time reaches 39.55% of the effective two-link roofline, 11.728 us short of the
40% target. This near-target result does not justify sacrificing matmul cores
for more CCL workers; the matched T=640 two-worker experiment already
regressed. Long-sequence work should instead preserve this placement and
reduce the scan serial chunk chain or overlap within the fused output
program.
