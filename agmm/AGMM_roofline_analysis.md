# AGMM (AllGatherMinimalMatmulAsyncOp) Roofline Analysis

Per-instance roofline analysis of the fused **AllGather + Matmul** op, extracted from two
32-device Blackhole op-perf dumps (`transformer_stage1` and `transformer_stage2`).
22 AGMM instances per stage, 44 total.

## Method

- **Performance = fastest device.** Each logical AGMM runs on all 32 devices; we align the
  *i*-th AGMM across devices and take the **minimum** kernel duration. Because AGMM is a
  ring collective, the slowest device's time is inflated by dispatch/collective wait
  (measured skew: median ~2-3x, up to ~6x, driven by dispatch order), so the fastest
  device is the cleanest estimate of true op cost.
- **All 44 instances are uniform:** ring_size=4, num_links=2, Ring topology, BF16 in/out,
  HiFi2, 12x9=108-core compute grid. The all-gather is along K, so
  `K_gathered = K_local x ring_size`.

### Peaks and formulas

| Resource | Peak | Notes |
|---|---|---|
| Compute | **298.6 TFLOP/s** | 108 cores x 2048 FLOP/cyc x 1.35 GHz. Per core 8x16x16 = 4096 FLOP at full rate; HiFi2 takes 2 cycles -> 2048 FLOP/cyc. |
| DRAM | **512 GB/s** | bytes read = 2 x (M x K_gathered + K_gathered x N), BF16. |
| Fabric | **50 GB/s per unidirectional link** | bidirectional ring all-gather, 2 links/direction. |

- `FLOPs = 2 x M x K_gathered x N`
- **Fabric (bidirectional ring all-gather):** each device contributes shard
  `S = M x K_local x 2 bytes`. Per unidirectional link the volume is
  `(ring_size - 1) x S / (2 x num_links)` (the `/2` is the bidirectional split).
  Fabric volume depends only on M, K_local, ring_size -- **independent of N** and of the
  addcmul/chunk fusions (weights are not gathered; FSDP is off here).
- **Utilization** = achieved / peak, using the fastest-device time. `bound` = whichever of
  the three utilizations is highest.

## Results

| stg | id | M | K_gat | N | fused | t (µs) | TFLOP/s | FLOP% | DRAM% | fab GB/s | FAB% | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stage1 | 0 | 1216 | 4096 | 8 | — | 136.1 | 0.6 | 0.2% | 14.4% | 13.7 | 27.4% | fabric |
| stage1 | 1 | 1216 | 4096 | 3072 |  chunks=3 | 220.6 | 138.7 | 46.5% | 31.1% | 8.5 | 16.9% | compute |
| stage1 | 2 | 1216 | 4096 | 1024 | addcmul | 146.1 | 69.8 | 23.4% | 24.5% | 12.8 | 25.6% | fabric |
| stage1 | 3 | 1216 | 4096 | 8 | — | 136.7 | 0.6 | 0.2% | 14.3% | 13.7 | 27.3% | fabric |
| stage1 | 4 | 1216 | 4096 | 1024 | — | 144.8 | 70.4 | 23.6% | 24.7% | 12.9 | 25.8% | fabric |
| stage1 | 5 | 1216 | 4096 | 1024 | addcmul | 147.6 | 69.1 | 23.1% | 24.3% | 12.7 | 25.3% | fabric |
| stage1 | 6 | 32 | 2048 | 8 | — | 37.3 | 0.0 | 0.0% | 0.9% | 0.7 | 1.3% | fabric |
| stage1 | 7 | 32 | 2048 | 1536 |  chunks=3 | 59.7 | 3.4 | 1.1% | 21.0% | 0.4 | 0.8% | dram |
| stage1 | 8 | 32 | 2048 | 512 | addcmul | 37.4 | 1.8 | 0.6% | 11.6% | 0.7 | 1.3% | dram |
| stage1 | 9 | 32 | 2048 | 8 | — | 37.3 | 0.0 | 0.0% | 0.9% | 0.7 | 1.3% | fabric |
| stage1 | 10 | 32 | 2048 | 512 | — | 37.5 | 1.8 | 0.6% | 11.6% | 0.7 | 1.3% | dram |
| stage1 | 11 | 32 | 2048 | 512 | — | 37.5 | 1.8 | 0.6% | 11.6% | 0.7 | 1.3% | dram |
| stage1 | 12 | 1216 | 4096 | 8 | — | 137.3 | 0.6 | 0.2% | 14.3% | 13.6 | 27.2% | fabric |
| stage1 | 13 | 1216 | 4096 | 512 | — | 136.8 | 37.3 | 12.5% | 20.2% | 13.7 | 27.3% | fabric |
| stage1 | 14 | 256 | 2048 | 1024 |  chunks=2 | 47.6 | 22.6 | 7.6% | 21.5% | 4.1 | 8.3% | dram |
| stage1 | 15 | 1216 | 2048 | 1024 | — | 92.4 | 55.2 | 18.5% | 19.4% | 10.1 | 20.2% | fabric |
| stage1 | 16 | 32 | 2048 | 8 | — | 37.3 | 0.0 | 0.0% | 0.9% | 0.7 | 1.3% | fabric |
| stage1 | 17 | 32 | 2048 | 512 | — | 37.5 | 1.8 | 0.6% | 11.6% | 0.7 | 1.3% | dram |
| stage1 | 18 | 1216 | 4096 | 1024 | — | 144.8 | 70.4 | 23.6% | 24.8% | 12.9 | 25.8% | fabric |
| stage1 | 19 | 32 | 2048 | 512 | — | 37.4 | 1.8 | 0.6% | 11.6% | 0.7 | 1.3% | dram |
| stage1 | 20 | 1216 | 4096 | 4096 | — | 486.6 | 83.8 | 28.1% | 17.5% | 3.8 | 7.7% | compute |
| stage1 | 21 | 32 | 2048 | 2048 | — | 110.2 | 2.4 | 0.8% | 15.1% | 0.2 | 0.4% | dram |
| stage2 | 0 | 4864 | 4096 | 8 | — | 456.3 | 0.7 | 0.2% | 17.1% | 16.4 | 32.7% | fabric |
| stage2 | 1 | 4864 | 4096 | 3072 |  chunks=3 | 684.1 | 178.9 | 59.9% | 18.6% | 10.9 | 21.8% | compute |
| stage2 | 2 | 4864 | 4096 | 1024 | addcmul | 513.5 | 79.5 | 26.6% | 18.3% | 14.5 | 29.1% | fabric |
| stage2 | 3 | 4864 | 4096 | 8 | — | 455.1 | 0.7 | 0.2% | 17.1% | 16.4 | 32.8% | fabric |
| stage2 | 4 | 4864 | 4096 | 1024 | — | 487.4 | 83.7 | 28.0% | 19.3% | 15.3 | 30.7% | fabric |
| stage2 | 5 | 4864 | 4096 | 1024 | addcmul | 513.4 | 79.5 | 26.6% | 18.3% | 14.6 | 29.1% | fabric |
| stage2 | 6 | 32 | 2048 | 8 | — | 37.3 | 0.0 | 0.0% | 0.9% | 0.7 | 1.3% | fabric |
| stage2 | 7 | 32 | 2048 | 1536 |  chunks=3 | 59.8 | 3.4 | 1.1% | 21.0% | 0.4 | 0.8% | dram |
| stage2 | 8 | 32 | 2048 | 512 | addcmul | 37.5 | 1.8 | 0.6% | 11.6% | 0.7 | 1.3% | dram |
| stage2 | 9 | 32 | 2048 | 8 | — | 37.2 | 0.0 | 0.0% | 0.9% | 0.7 | 1.3% | fabric |
| stage2 | 10 | 32 | 2048 | 512 | — | 37.5 | 1.8 | 0.6% | 11.6% | 0.7 | 1.3% | dram |
| stage2 | 11 | 32 | 2048 | 512 | — | 37.5 | 1.8 | 0.6% | 11.6% | 0.7 | 1.3% | dram |
| stage2 | 12 | 4864 | 4096 | 8 | — | 460.4 | 0.7 | 0.2% | 16.9% | 16.2 | 32.5% | fabric |
| stage2 | 13 | 4864 | 4096 | 512 | — | 496.1 | 41.1 | 13.8% | 17.3% | 15.1 | 30.1% | fabric |
| stage2 | 14 | 256 | 2048 | 1024 |  chunks=2 | 47.7 | 22.5 | 7.5% | 21.5% | 4.1 | 8.2% | dram |
| stage2 | 15 | 4864 | 2048 | 1024 | — | 289.9 | 70.4 | 23.6% | 16.2% | 12.9 | 25.8% | fabric |
| stage2 | 16 | 32 | 2048 | 8 | — | 37.3 | 0.0 | 0.0% | 0.9% | 0.7 | 1.3% | fabric |
| stage2 | 17 | 32 | 2048 | 512 | — | 37.5 | 1.8 | 0.6% | 11.6% | 0.7 | 1.3% | dram |
| stage2 | 18 | 4864 | 4096 | 1024 | — | 499.0 | 81.8 | 27.4% | 18.9% | 15.0 | 29.9% | fabric |
| stage2 | 19 | 32 | 2048 | 512 | — | 37.5 | 1.8 | 0.6% | 11.6% | 0.7 | 1.3% | dram |
| stage2 | 20 | 4864 | 4096 | 4096 | — | 1576.0 | 103.6 | 34.7% | 9.1% | 4.7 | 9.5% | compute |
| stage2 | 21 | 32 | 2048 | 2048 | — | 110.2 | 2.4 | 0.8% | 15.1% | 0.2 | 0.4% | dram |

## Takeaways

- **Nothing is saturated.** Across all 44 instances, fabric util peaks at ~33%, compute at
  ~60%, DRAM at ~31%. No single resource clears two-thirds of peak -- the signature of a
  **latency / dispatch-bound** workload with imperfect comm/compute overlap.
- **Small-N projections are fabric-dominated.** `N=8`/`N=512` ops are compute-trivial and
  are essentially "just an all-gather" -- fabric is the top resource (27-33%) and sets the
  runtime. Best payoff for reducing dispatch stagger / improving link efficiency.
- **Fabric BW/link is ~constant for fixed M** (volume is N-independent): M=1216 ops move
  ~13-14 GB/s/link, M=4864 ops ~15-16 GB/s/link. FAB% is highest at small N (nothing to
  hide the gather behind) and falls as N grows and compute stretches the op.
- **Large-N ops flip to compute-bound**, but even the biggest (s2/20, 1.58 ms) reaches only
  34.7% FLOP while fabric drops to 9.5% -- the gather is well overlapped there; the ceiling
  is matmul efficiency, not the ring.
- **Small-M (=32) ops are pure overhead** -- all three utils in the low single digits,
  pinned to a ~37 us floor. The op does almost nothing but pays a fixed dispatch tax.

> Caveat: fabric and compute **overlap** within AGMM by design, so these per-resource utils
> are "share of this op's wall-clock," not simultaneous occupancy. A combined
> `max(compute, fabric)` overlapped-ideal model would separate overlap inefficiency from
> dispatch overhead.

## Artifacts

- `agmm_instances.csv` / `.json` -- full per-instance extraction (shapes, dtypes, fusions, collective + matmul config, min/max/mean time, skew).
- `agmm_roofline.csv` -- the roofline table above (with shard/fabric byte columns).
