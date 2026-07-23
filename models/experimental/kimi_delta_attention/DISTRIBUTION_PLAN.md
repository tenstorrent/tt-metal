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
