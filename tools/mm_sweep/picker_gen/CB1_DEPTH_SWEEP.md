# in1 CB depth sweep (TT_REGIME_A_CB1_DEPTH) at DEPLOYED configs

Tests the concurrency half of the in1-read hypothesis from `ABLATION_IN1_READ.md`: `cb1` holds
`depth * kb * N_sub` tiles, so `depth` (production 4) caps how many in1 blocks a reader keeps in flight. If
the read is bound by latency x concurrency rather than DRAM bandwidth, raising the depth should shrink the
wall.

## Mechanism

`cb1_depth` is an EXPLICIT env override (`TT_REGIME_A_CB1_DEPTH`, blocks), not a greedy leftover-L1 fill, so
each depth is measured exactly. It is stored as a hashed operation attribute, so every depth is its own cached
program and several depths can be swept in one device session. It deliberately does NOT feed
`auto_select_config`, so the picked config is identical at every depth and the sweep is a clean A/B of
buffering alone. A depth that overflows the L1 budget is rejected by the planner with an explicit error rather
than silently clamped (depth 64 is infeasible on `256x15360x768`).

**Bit-exact at every depth on every shape** (`bit_exact_vs_first = true` throughout) - pure buffering, no
numerical change. mask-0 / default-depth path unchanged: 111/111 regression.

## Results (median us; depth 4 = production)

| shape | d4 | d8 | d16 | d32 | d64 | verdict |
|---|---|---|---|---|---|---|
| **256x2048x6144** | 92.17 | +1.2% | +5.8% | **+7.7%** | +7.0% | **WIN, saturates at 32** |
| 256x6144x4608 | 141.81 | +0.7% | +0.7% | +0.2% | +0.4% | flat |
| 32x6144x1536 | 40.42 | -0.4% | -0.2% | -0.4% | -0.4% | flat (DRAM-bound) |
| 256x2048x2048 | 37.64 | -1.8% | -1.0% | -0.6% | -0.6% | slightly negative |
| 256x15360x768 | 95.26 | -1.4% | -1.9% | -0.7% | infeasible | slightly negative |
| 512x6144x4608 | 206.77 | -1.0% | -2.1% | **-3.0%** | -4.3% | monotone WORSE |
| 512x6144x2304 | 133.12 | -2.0% | -3.6% | **-4.6%** | -6.3% | monotone WORSE |

Reversed-order relaunch (depths swept 64 -> 4) reproduces both signs, so this is not an order artifact:
`256x2048x6144` +7.4% at d32 (three sessions: +8.6, +7.7, +7.4), `512x6144x2304` -3.1% at d32.

L1 is not the constraint on most shapes: at production depth `256x2048x6144` uses only 377 KB of the 1474 KB
budget (cb0 is small at deployed configs because Mt<=8 keeps `M_block` small), and depth 32 costs 492 KB.

## Causal confirmation on the winning shape

Re-running the `SKIP_IN1_READ` ablation at each depth isolates what changed:

| | base us | skip_in1_read us | in1 exposure |
|---|---|---|---|
| depth 4 | 93.30 | 57.62 | 35.7 us (38.2%) |
| depth 32 | **85.54** | 58.20 | **27.3 us (32.0%)** |

The skip-in1 floor is unchanged (57.6 -> 58.2 us) while the base drops 7.8 us. So the depth increase removed
8.4 us of in1-read exposure and affected nothing else - the concurrency half of the hypothesis is CONFIRMED,
and it accounts for ~24% of that shape's in1 exposure. The remaining 27.3 us is DRAM bandwidth plus latency;
the latency part is what the placement work would target.

## Why it hurts the in0-dominated shapes

The regressions are monotone in depth and land exactly on the shapes where in0 delivery is on the critical
path (`512x6144x2304`: ring +17.9%, in0 read +9.8%; `512x6144x4608`: +11.7% / +6.8%), while the DRAM-bound
shapes are flat and the in1-dominated one wins. Prefetching in1 harder puts more read traffic in flight
concurrently with the in0 own-shard read and the ring forward, so it steals DRAM and NoC bandwidth from the
in0 gather - which is the critical path there. That is the same competition the whole campaign keeps hitting:
in1-read exposure and in0-delivery exposure are anti-correlated, and a lever that helps one can charge the
other.

## Status: real but shape-selective, NOT ready to ship

- The win is mechanism-confirmed, bit-exact, reproduced in three sessions, and worth ~7.7% on one shape.
- It regresses two shapes by 3-4.6% at the same depth, so a global default depth change is off the table.
- A gate cannot be fitted yet: only ONE shape wins, so any rule would be fitted to n=1. Note that
  bytes-in-flight alone does not discriminate - `256x2048x6144` (win) and `512x6144x2304` (loss) both have
  nsb=1 and 16 KB in flight; the difference is whether in0 delivery is exposed (Mt=8 vs Mt=16, ring 6.9% vs
  17.9%).
- Next step to make it shippable: sweep depth {4, 16, 32} over the 60-shape corpus at deployed configs
  (~30-40 min) to find the win set, then fit a conservative picker gate on (in1 exposure high, in0 delivery
  light) - the same shape-selective discipline the reduce-scatter work used.

## Reproduction

```
export TT_METAL_DEVICE_PROFILER=1 ARCH_NAME=blackhole
W=tools/mm_sweep/picker_gen/cb1_depth_worker.py
python3 $W 256 2048 6144 4,8,16,32,64 1        # sweep, with PCC + bit-exactness per depth
TT_REGIME_A_CB1_DEPTH=32 python3 tools/mm_sweep/picker_gen/corpus_ab_worker.py 256 2048 6144 0,2048 0
```
