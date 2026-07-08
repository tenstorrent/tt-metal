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
- **Utilization** = achieved / peak, using the fastest-device time.

### Best-case projection (`ideal µs`, `limiter`, `speedup`)

Assume we can reach these fractions of peak: **50% FLOP util, 90% DRAM BW, 80% fabric BW**.
Compute, DRAM, and fabric **overlap** within AGMM, so the achievable time is set by the
slowest single resource:

```
t_compute = FLOPs        / (0.50 x 298.6 TFLOP/s)
t_dram    = bytes_read   / (0.90 x 512 GB/s)
t_fabric  = bytes/link   / (0.80 x 50 GB/s)
ideal     = max(t_compute, t_dram, t_fabric)     limiter = argmax
speedup   = measured / ideal
```

## Results

| stg | id | M | K_gat | N | fused | meas µs | FLOP% | DRAM% | FAB% | ideal µs | limiter | speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stage1 | 0 | 1216 | 4096 | 8 | — | 136.1 | 0.2% | 14.4% | 27.4% | 46.7 | fabric | 2.92x |
| stage1 | 1 | 1216 | 4096 | 3072 |  chunks=3 | 220.6 | 46.5% | 31.1% | 16.9% | 205.0 | compute | 1.08x |
| stage1 | 2 | 1216 | 4096 | 1024 | addcmul | 146.1 | 23.4% | 24.5% | 25.6% | 68.3 | compute | 2.14x |
| stage1 | 3 | 1216 | 4096 | 8 | — | 136.7 | 0.2% | 14.3% | 27.3% | 46.7 | fabric | 2.93x |
| stage1 | 4 | 1216 | 4096 | 1024 | — | 144.8 | 23.6% | 24.7% | 25.8% | 68.3 | compute | 2.12x |
| stage1 | 5 | 1216 | 4096 | 1024 | addcmul | 147.6 | 23.1% | 24.3% | 25.3% | 68.3 | compute | 2.16x |
| stage1 | 6 | 32 | 2048 | 8 | — | 37.3 | 0.0% | 0.9% | 1.3% | 0.6 | fabric | 60.68x |
| stage1 | 7 | 32 | 2048 | 1536 |  chunks=3 | 59.7 | 1.1% | 21.0% | 0.8% | 13.9 | dram | 4.29x |
| stage1 | 8 | 32 | 2048 | 512 | addcmul | 37.4 | 0.6% | 11.6% | 1.3% | 4.8 | dram | 7.74x |
| stage1 | 9 | 32 | 2048 | 8 | — | 37.3 | 0.0% | 0.9% | 1.3% | 0.6 | fabric | 60.71x |
| stage1 | 10 | 32 | 2048 | 512 | — | 37.5 | 0.6% | 11.6% | 1.3% | 4.8 | dram | 7.75x |
| stage1 | 11 | 32 | 2048 | 512 | — | 37.5 | 0.6% | 11.6% | 1.3% | 4.8 | dram | 7.76x |
| stage1 | 12 | 1216 | 4096 | 8 | — | 137.3 | 0.2% | 14.3% | 27.2% | 46.7 | fabric | 2.94x |
| stage1 | 13 | 1216 | 4096 | 512 | — | 136.8 | 12.5% | 20.2% | 27.3% | 46.7 | fabric | 2.93x |
| stage1 | 14 | 256 | 2048 | 1024 |  chunks=2 | 47.6 | 7.6% | 21.5% | 8.3% | 11.4 | dram | 4.18x |
| stage1 | 15 | 1216 | 2048 | 1024 | — | 92.4 | 18.5% | 19.4% | 20.2% | 34.2 | compute | 2.71x |
| stage1 | 16 | 32 | 2048 | 8 | — | 37.3 | 0.0% | 0.9% | 1.3% | 0.6 | fabric | 60.73x |
| stage1 | 17 | 32 | 2048 | 512 | — | 37.5 | 0.6% | 11.6% | 1.3% | 4.8 | dram | 7.75x |
| stage1 | 18 | 1216 | 4096 | 1024 | — | 144.8 | 23.6% | 24.8% | 25.8% | 68.3 | compute | 2.12x |
| stage1 | 19 | 32 | 2048 | 512 | — | 37.4 | 0.6% | 11.6% | 1.3% | 4.8 | dram | 7.74x |
| stage1 | 20 | 1216 | 4096 | 4096 | — | 486.6 | 28.1% | 17.5% | 7.7% | 273.3 | compute | 1.78x |
| stage1 | 21 | 32 | 2048 | 2048 | — | 110.2 | 0.8% | 15.1% | 0.4% | 18.5 | dram | 5.96x |
| stage2 | 0 | 4864 | 4096 | 8 | — | 456.3 | 0.2% | 17.1% | 32.7% | 186.8 | fabric | 2.44x |
| stage2 | 1 | 4864 | 4096 | 3072 |  chunks=3 | 684.1 | 59.9% | 18.6% | 21.8% | 819.9 | compute | 0.83x |
| stage2 | 2 | 4864 | 4096 | 1024 | addcmul | 513.5 | 26.6% | 18.3% | 29.1% | 273.3 | compute | 1.88x |
| stage2 | 3 | 4864 | 4096 | 8 | — | 455.1 | 0.2% | 17.1% | 32.8% | 186.8 | fabric | 2.44x |
| stage2 | 4 | 4864 | 4096 | 1024 | — | 487.4 | 28.0% | 19.3% | 30.7% | 273.3 | compute | 1.78x |
| stage2 | 5 | 4864 | 4096 | 1024 | addcmul | 513.4 | 26.6% | 18.3% | 29.1% | 273.3 | compute | 1.88x |
| stage2 | 6 | 32 | 2048 | 8 | — | 37.3 | 0.0% | 0.9% | 1.3% | 0.6 | fabric | 60.73x |
| stage2 | 7 | 32 | 2048 | 1536 |  chunks=3 | 59.8 | 1.1% | 21.0% | 0.8% | 13.9 | dram | 4.29x |
| stage2 | 8 | 32 | 2048 | 512 | addcmul | 37.5 | 0.6% | 11.6% | 1.3% | 4.8 | dram | 7.75x |
| stage2 | 9 | 32 | 2048 | 8 | — | 37.2 | 0.0% | 0.9% | 1.3% | 0.6 | fabric | 60.63x |
| stage2 | 10 | 32 | 2048 | 512 | — | 37.5 | 0.6% | 11.6% | 1.3% | 4.8 | dram | 7.75x |
| stage2 | 11 | 32 | 2048 | 512 | — | 37.5 | 0.6% | 11.6% | 1.3% | 4.8 | dram | 7.75x |
| stage2 | 12 | 4864 | 4096 | 8 | — | 460.4 | 0.2% | 16.9% | 32.5% | 186.8 | fabric | 2.46x |
| stage2 | 13 | 4864 | 4096 | 512 | — | 496.1 | 13.8% | 17.3% | 30.1% | 186.8 | fabric | 2.66x |
| stage2 | 14 | 256 | 2048 | 1024 |  chunks=2 | 47.7 | 7.5% | 21.5% | 8.2% | 11.4 | dram | 4.19x |
| stage2 | 15 | 4864 | 2048 | 1024 | — | 289.9 | 23.6% | 16.2% | 25.8% | 136.6 | compute | 2.12x |
| stage2 | 16 | 32 | 2048 | 8 | — | 37.3 | 0.0% | 0.9% | 1.3% | 0.6 | fabric | 60.73x |
| stage2 | 17 | 32 | 2048 | 512 | — | 37.5 | 0.6% | 11.6% | 1.3% | 4.8 | dram | 7.75x |
| stage2 | 18 | 4864 | 4096 | 1024 | — | 499.0 | 27.4% | 18.9% | 29.9% | 273.3 | compute | 1.83x |
| stage2 | 19 | 32 | 2048 | 512 | — | 37.5 | 0.6% | 11.6% | 1.3% | 4.8 | dram | 7.75x |
| stage2 | 20 | 4864 | 4096 | 4096 | — | 1576.0 | 34.7% | 9.1% | 9.5% | 1093.2 | compute | 1.44x |
| stage2 | 21 | 32 | 2048 | 2048 | — | 110.2 | 0.8% | 15.1% | 0.4% | 18.5 | dram | 5.96x |

**Aggregate (sum of all 44 instances): 9.4 ms measured -> 5.0 ms ideal = 1.88x.**

## Takeaways

- **~1.9x aggregate headroom** across the whole workload under these ceilings.
- **Small-M (=32) ops have the largest relative headroom** (up to ~60x on the N=8 cases,
  ~6-8x on the rest). They are pinned to a ~37 us dispatch floor while their ideal work is
  <1-5 us -- almost the entire runtime is overhead, not useful work. Fixing dispatch
  stagger / launch cost is the highest-leverage change.
- **Large-M mid-N ops (N=1024) are compute-limited at ~2x.** At M=1216/4864 the projection
  says these should run ~2x faster once compute hits 50% util.
- **Fabric-limited ops are the N=8/N=512 projections** (~2.4-2.9x) -- runtime is dominated
  by the all-gather; raising fabric efficiency to 80% roughly halves them.
- **The biggest matmuls are already close to the ceiling.** s1/1 (N=3072) = 1.08x and
  s2/1 (N=3072) = **0.83x** -- their *measured* FLOP util (46-60%) already meets/exceeds the
  assumed 50% compute ceiling, so the model predicts little-to-no gain. Interpretation: 50%
  is a conservative compute ceiling for large-N shapes; those ops are effectively optimized.
- **DRAM never binds by much** -- it is the limiter only on some small-M ops, and even then
  the real problem there is the fixed dispatch floor, not bandwidth.

> Caveat: the projection assumes perfect overlap of compute/DRAM/fabric (`max` model) and a
> fixed per-resource efficiency ceiling. Where measured util already exceeds a ceiling
> (e.g. N=3072), speedup < 1x is a model artifact indicating the assumed ceiling is too low
> for that shape, not a real regression.

## Artifacts

- `agmm_instances.csv` / `.json` -- full per-instance extraction (shapes, dtypes, fusions, collective + matmul config, min/max/mean time, skew).
- `agmm_roofline.csv` -- the roofline table above (utils, byte volumes, ideal time, limiter, speedup).
- `extract_agmm.py`, `roofline.py`, `skew.py` -- reproducible scripts.
