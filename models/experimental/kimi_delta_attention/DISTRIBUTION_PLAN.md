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
| `f_a` / `g_a` ranks | two `[1,T,128]` | replicated; required by local output shards |
| Beta projection | `[1,T,32]` | output-sharded `[1,T,4]` |
| `f_b` / `g_b` outputs | two `[1,T,4096]` | output-sharded to two `[1,T,512]` |
| Decay/bias/norm | 32 heads | four whole-head slices |
| Recurrent state | `[1,32,128,128]` | `[1,4,128,128]` |
| Pre-output activation | `[1,T,4096]` | `[1,T,512]` |
| Output projection | `[4096,2304]` | row-sharded `[512,2304]` |
| Output partial | `[1,T,2304]` | reduce-scatter or all-reduce input |

The fused auxiliary projection is rebuilt per device as
`[f_a(full 128), g_a(full 128), b(local 4)]`, width 260. Replicating the two
low-rank results is cheaper and simpler than introducing two collectives before
recurrence.

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
