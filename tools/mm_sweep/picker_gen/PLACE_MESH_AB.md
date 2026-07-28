# 2D (bank x slice) mesh placement (diag bit13): +6.4 to +13.4% on 4 of 7 shapes

Placement aimed at in0 RING traffic, after the in1-focused attempt showed in1 reads are already DRAM-bound
(`IN1_PLACEMENT_AB.md`). Host-only, correctness-preserving (writes only `P.cores[i].coord`); PCC 0.999991 to
1.000144 on every shape; mask-0 unchanged, 111/111 regression.

## The structural idea

Two traffic classes want opposite groupings of the same 8 x preaders core array:

- the in0 **RING** connects the 8 cores of one SLICE (one per bank) -> wants slice-compact clusters
- the split-K **REDUCTION** chain connects the Pk cores of one BANK -> wants bank-compact clusters

They are orthogonal partitions, so no clustering makes both short. Production picks bank-compact blobs (short
reduction, long ring); "ring-compact blocks" just flips the sacrifice. A **2D embedding escapes the tension**:
put banks along x and slices along y, and then a ring step (bank -> bank) and a reduction step (kk -> kk+1)
are each ONE hop, in different dimensions. Layout: `(bank b, slice p) -> (x=b, y=p)`, with each overflow slice
(p >= grid.y) taking its own spare column at x >= 8, rows 0..7. Collision-free by construction, and
mm-siblings are consecutive in p so M-split slaves land next to their reader with no extra pass.

Offline (`in0_ring_place_search.py`, exact route model), vs production:

| shape | ring hops | reduction hops | in1 hops | whole-op peak | total link |
|---|---|---|---|---|---|
| 512x6144x2304 | 640 -> **194 (-70%)** | 875 -> **712 (-19%)** | 1296 -> 1338 (+3%) | 5.89 -> **5.26** | 1055 -> **812** |
| 256x15360x768 | 657 -> **194 (-70%)** | 867 -> **608 (-30%)** | 646 -> 653 | 5.47 -> **4.63** | 1135 -> **726** |
| 256x2048x6144 | 640 -> **194 (-70%)** | 960 -> **580 (-40%)** | 1296 -> 1338 | 4.15 -> **3.69** | 701 -> **559** |

Ring AND reduction improve together, which no clustering layout achieved.

## Measured (median us, 2 relaunches with mode order reversed)

| shape | ring exp | in1 exp | production | mesh | delta |
|---|---|---|---|---|---|
| 512x6144x2304 | +17.4% | +5.0% | 123.9 | **112.3** | **+9.4%** |
| 512x6144x4608 | +11.4% | +5.2% | 197.8 | **185.1** | **+6.4%** |
| 256x15360x768 | +21.7% | +14.3% | 95.9 | **88.1** | **+8.2%** |
| 256x2048x6144 | +6.9% | +36.9% | 92.7 | **80.3** | **+13.4%** |
| 256x2048x2048 | +5.0% | +19.9% | 39.1 | 41.4 | -5.9% |
| 32x6144x1536 | +0.2% | +70.7% | 41.2 | 49.1 | **-19.4%** |
| 256x6144x4608 | +5.2% | +32.1% | 143.5 | 175.7 | **-22.5%** |

The split is explained by which resource the shape is limited by. The mesh trades in1 read distance (+3%
hops, and cores are no longer next to their bank) for a 70% ring cut. Where the ring is exposed, that is a
large win. Where the shape is at the in1/DRAM wall it is a large loss - `32x6144x1536` (98% of peak on the
in1 read in isolation, ring exposure +0.2%) and `256x6144x4608` (97%) have no ring to win and pay the full in1
penalty. `256x2048x6144` is the one shape that wins despite an in1-dominated profile: its ring+reduction cut
(640 -> 194 and 960 -> 580 hops) outweighs the in1 cost, consistent with it also being the shape that
responded to cb1 depth (latency-bound rather than bandwidth-bound on in1).

## Cumulative, with the NoC-cache API fix

| shape | original | + API fix | + mesh | best total |
|---|---|---|---|---|
| 512x6144x2304 | 134.0 | 123.9 (+8.0%) | **112.3** | **+16.2%** |
| 512x6144x4608 | 207.5 | 197.8 (+4.5%) | **185.1** | **+10.8%** |
| 256x2048x6144 | 92.2 | 92.7 (-0.6%) | **80.3** | **+12.9%** |
| 256x15360x768 | 95.1 | 95.9 (-0.7%) | **88.1** | **+7.4%** |
| 256x6144x4608 | 141.4 | 143.5 (-1.5%) | 175.7 | -1.5% (keep production) |
| 32x6144x1536 | 40.5 | 41.3 (-1.9%) | 49.1 | -1.9% (keep production) |
| 256x2048x2048 | 37.9 | 39.3 (-3.7%) | 41.4 | -3.7% (keep production) |

## Status and next step

This is the first placement change that produces real speedup: **+7.4 to +16.2% on four of seven shapes**,
from cutting in0 ring traffic ~70% while also shortening the reduction chain. It is shape-selective, and the
discriminator is legible for once: adopt when the ring is exposed and the shape is not at the in1/DRAM wall
(e.g. ring-skip gain > in1-skip gain, or in1-read util below ~90%), both of which are measurable per shape and
derivable offline.

Before promotion: (1) fit the gate on the 60-shape corpus rather than these 7; (2) revisit the three API-fix
regressions (-0.6 to -3.7%), which are a placement side effect of the now-correct per-NoC targets and may be
recoverable by letting regime_a choose its NOC_1 target explicitly; (3) re-run the ring ORDER optimisation on
the mesh layout, since the ring cost model changes completely when edges are one hop.

## Reproduction

```
export TT_METAL_DEVICE_PROFILER=1 ARCH_NAME=blackhole
python3 tools/mm_sweep/picker_gen/in0_ring_place_search.py 16 192 72 1 12 1 2 1   # offline layout compare
python3 tools/mm_sweep/picker_gen/corpus_ab_worker.py 512 6144 2304 0,8192 1      # prod vs mesh
```
