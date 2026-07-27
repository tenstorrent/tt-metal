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
| Fabric | **25 GB/s per unidirectional link** | bidirectional ring all-gather, 2 links/direction. |

- `FLOPs = 2 x M x K_gathered x N`
- **Fabric (bidirectional ring all-gather):** each device contributes shard
  `S = M x K_local x 2 bytes`. Per unidirectional link the volume is
  `(ring_size - 1) x S / (2 x num_links)` (the `/2` is the bidirectional split).
  Fabric volume depends only on M, K_local, ring_size -- **independent of N** and of the
  addcmul/chunk fusions (weights are not gathered; FSDP is off here).
- **Utilization** = achieved / peak, using the fastest-device time.

### Best-case projection (`ideal µs`, `limiter`, `speedup`)

`ideal` is the **hard 100%-of-peak roofline** — full FLOP throughput, full DRAM BW, full
fabric BW, no de-rating. Compute, DRAM, and fabric **overlap** within AGMM, so the
achievable time is set by the slowest single resource:

```
t_compute = FLOPs        / (298.6 TFLOP/s)
t_dram    = bytes_read   / (512 GB/s)
t_fabric  = bytes/link   / (25 GB/s)
ideal     = max(t_compute, t_dram, t_fabric)     limiter = argmax
speedup   = measured / ideal
```

## Results

| stg | id | M | K_gat | N | fused | meas µs | FLOP% | DRAM% | FAB% | ideal µs | limiter | speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stage1 | 0 | 1216 | 4096 | 8 | — | 136.1 | 0.2% | 14.4% | 54.9% | 74.7 | fabric | 1.82x |
| stage1 | 1 | 1216 | 4096 | 3072 | chunks=3 | 220.6 | 46.5% | 31.1% | 33.9% | 102.5 | compute | 2.15x |
| stage1 | 2 | 1216 | 4096 | 1024 | addcmul | 146.1 | 23.4% | 24.5% | 51.2% | 74.7 | fabric | 1.95x |
| stage1 | 3 | 1216 | 4096 | 8 | — | 136.7 | 0.2% | 14.3% | 54.6% | 74.7 | fabric | 1.83x |
| stage1 | 4 | 1216 | 4096 | 1024 | — | 144.8 | 23.6% | 24.7% | 51.6% | 74.7 | fabric | 1.94x |
| stage1 | 5 | 1216 | 4096 | 1024 | addcmul | 147.6 | 23.1% | 24.3% | 50.6% | 74.7 | fabric | 1.98x |
| stage1 | 6 | 32 | 2048 | 8 | — | 37.3 | 0.0% | 0.9% | 2.6% | 1.0 | fabric | 37.92x |
| stage1 | 7 | 32 | 2048 | 1536 | chunks=3 | 59.7 | 1.1% | 21.0% | 1.6% | 12.5 | dram | 4.76x |
| stage1 | 8 | 32 | 2048 | 512 | addcmul | 37.4 | 0.6% | 11.6% | 2.6% | 4.4 | dram | 8.60x |
| stage1 | 9 | 32 | 2048 | 8 | — | 37.3 | 0.0% | 0.9% | 2.6% | 1.0 | fabric | 37.94x |
| stage1 | 10 | 32 | 2048 | 512 | — | 37.5 | 0.6% | 11.6% | 2.6% | 4.4 | dram | 8.61x |
| stage1 | 11 | 32 | 2048 | 512 | — | 37.5 | 0.6% | 11.6% | 2.6% | 4.4 | dram | 8.62x |
| stage1 | 12 | 1216 | 4096 | 8 | — | 137.3 | 0.2% | 14.3% | 54.4% | 74.7 | fabric | 1.84x |
| stage1 | 13 | 1216 | 4096 | 512 | — | 136.8 | 12.5% | 20.2% | 54.6% | 74.7 | fabric | 1.83x |
| stage1 | 14 | 256 | 2048 | 1024 | chunks=2 | 47.6 | 7.6% | 21.5% | 16.5% | 10.2 | dram | 4.64x |
| stage1 | 15 | 1216 | 2048 | 1024 | — | 92.4 | 18.5% | 19.4% | 40.4% | 37.4 | fabric | 2.47x |
| stage1 | 16 | 32 | 2048 | 8 | — | 37.3 | 0.0% | 0.9% | 2.6% | 1.0 | fabric | 37.95x |
| stage1 | 17 | 32 | 2048 | 512 | — | 37.5 | 0.6% | 11.6% | 2.6% | 4.4 | dram | 8.61x |
| stage1 | 18 | 1216 | 4096 | 1024 | — | 144.8 | 23.6% | 24.7% | 51.6% | 74.7 | fabric | 1.94x |
| stage1 | 19 | 32 | 2048 | 512 | — | 37.4 | 0.6% | 11.6% | 2.6% | 4.4 | dram | 8.60x |
| stage1 | 20 | 1216 | 4096 | 4096 | — | 486.6 | 28.1% | 17.5% | 15.4% | 136.6 | compute | 3.56x |
| stage1 | 21 | 32 | 2048 | 2048 | — | 110.2 | 0.8% | 15.1% | 0.9% | 16.6 | dram | 6.62x |
| stage2 | 0 | 4864 | 4096 | 8 | — | 456.3 | 0.2% | 17.1% | 65.5% | 298.8 | fabric | 1.53x |
| stage2 | 1 | 4864 | 4096 | 3072 | chunks=3 | 684.1 | 59.9% | 18.6% | 43.7% | 409.9 | compute | 1.67x |
| stage2 | 2 | 4864 | 4096 | 1024 | addcmul | 513.5 | 26.6% | 18.3% | 58.2% | 298.8 | fabric | 1.72x |
| stage2 | 3 | 4864 | 4096 | 8 | — | 455.1 | 0.2% | 17.1% | 65.7% | 298.8 | fabric | 1.52x |
| stage2 | 4 | 4864 | 4096 | 1024 | — | 487.4 | 28.0% | 19.3% | 61.3% | 298.8 | fabric | 1.63x |
| stage2 | 5 | 4864 | 4096 | 1024 | addcmul | 513.4 | 26.6% | 18.3% | 58.2% | 298.8 | fabric | 1.72x |
| stage2 | 6 | 32 | 2048 | 8 | — | 37.3 | 0.0% | 0.9% | 2.6% | 1.0 | fabric | 37.95x |
| stage2 | 7 | 32 | 2048 | 1536 | chunks=3 | 59.8 | 1.1% | 21.0% | 1.6% | 12.5 | dram | 4.77x |
| stage2 | 8 | 32 | 2048 | 512 | addcmul | 37.5 | 0.6% | 11.6% | 2.6% | 4.4 | dram | 8.61x |
| stage2 | 9 | 32 | 2048 | 8 | — | 37.2 | 0.0% | 0.9% | 2.6% | 1.0 | fabric | 37.89x |
| stage2 | 10 | 32 | 2048 | 512 | — | 37.5 | 0.6% | 11.6% | 2.6% | 4.4 | dram | 8.61x |
| stage2 | 11 | 32 | 2048 | 512 | — | 37.5 | 0.6% | 11.6% | 2.6% | 4.4 | dram | 8.61x |
| stage2 | 12 | 4864 | 4096 | 8 | — | 460.4 | 0.2% | 16.9% | 64.9% | 298.8 | fabric | 1.54x |
| stage2 | 13 | 4864 | 4096 | 512 | — | 496.1 | 13.8% | 17.3% | 60.2% | 298.8 | fabric | 1.66x |
| stage2 | 14 | 256 | 2048 | 1024 | chunks=2 | 47.7 | 7.5% | 21.5% | 16.5% | 10.2 | dram | 4.66x |
| stage2 | 15 | 4864 | 2048 | 1024 | — | 289.9 | 23.6% | 16.2% | 51.5% | 149.4 | fabric | 1.94x |
| stage2 | 16 | 32 | 2048 | 8 | — | 37.3 | 0.0% | 0.9% | 2.6% | 1.0 | fabric | 37.95x |
| stage2 | 17 | 32 | 2048 | 512 | — | 37.5 | 0.6% | 11.6% | 2.6% | 4.4 | dram | 8.61x |
| stage2 | 18 | 4864 | 4096 | 1024 | — | 499.0 | 27.4% | 18.9% | 59.9% | 298.8 | fabric | 1.67x |
| stage2 | 19 | 32 | 2048 | 512 | — | 37.5 | 0.6% | 11.6% | 2.6% | 4.4 | dram | 8.61x |
| stage2 | 20 | 4864 | 4096 | 4096 | — | 1576.0 | 34.7% | 9.1% | 19.0% | 546.6 | compute | 2.88x |
| stage2 | 21 | 32 | 2048 | 2048 | — | 110.2 | 0.8% | 15.1% | 0.9% | 16.6 | dram | 6.62x |

**Aggregate (sum of all 44 instances): 9.4 ms measured -> 4.5 ms ideal = 2.09x.**

## Takeaways

- **~2.1x aggregate headroom** to the hard 100%-of-peak roofline across the whole workload.
- **Fabric is the dominant limiter** (24 of 44 instances). At **25 GB/s per link** the ring
  all-gather binds essentially every large-M shape (M=1216/4864) — N=8, N=512, and N=1024
  are all fabric-limited at ~1.5–2.0x. The gather moves `M x K_local` bytes and is
  independent of N, so widening N buys compute but no relief here; the lever is reducing the
  fabric transfer itself (more links, higher link BW, or a cheaper collective).
- **Small-M (=32) ops have the largest relative headroom** (up to ~38x on the N=8 cases,
  ~4.6–8.6x on the rest). They are pinned to a ~37 us dispatch floor while their ideal work
  is <1–5 us — almost the entire runtime is overhead. Fixing dispatch stagger / launch cost
  is the highest-leverage change for these.
- **The big matmuls (N=3072/4096) are compute-limited with real headroom.** s1/1 = 2.15x,
  s2/1 = 1.67x, s1/20 = 3.56x, s2/20 = 2.88x. Their measured FLOP util is 35–60% of physical
  peak, so there is genuine room before they hit the compute roofline (no sub-1x artifact:
  the ceiling is now the true 100%-of-peak).
- **DRAM never binds a large op** — it is the limiter only on some small-M shapes, where the
  real constraint is the fixed dispatch floor, not bandwidth.

> Caveat: `ideal` assumes perfect overlap of compute/DRAM/fabric (the `max` model) at the
> full physical peak of each resource. It is a hard lower bound on time, not an achievable
> target — real kernels never sustain 100% of any single resource, so a shape at ~1.5x is
> closer to practical peak than the raw number suggests. Use it to rank where the largest
> gaps are, not as a promised speedup.

## Artifacts

- `agmm_instances.csv` / `.json` -- full per-instance extraction (shapes, dtypes, fusions, collective + matmul config, min/max/mean time, skew).
- `agmm_roofline.csv` -- the roofline table above (utils, byte volumes, ideal time, limiter, speedup).
- `extract_agmm.py`, `roofline.py`, `skew.py` -- reproducible scripts.
