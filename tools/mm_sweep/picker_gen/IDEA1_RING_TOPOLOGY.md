# Idea 1 (region-local in0 rings) — REFUTED, and what replaced it

Follow-on from `D1_RING_ATTRIBUTION.md` (ring forward is ~70% hop-distance-bound). Idea 1 was: since the in0
ring is purely an in0-delivery construct, any 8 cores sharing a `(kk, mm)` can form a ring — so for Ns>=2,
re-partition each group's 8*Ns cores into Ns PHYSICALLY COMPACT rings instead of "the 8 banks of one slice",
killing the two bisection crossings every ring pays today. Host-only, correctness-preserving.

Implemented as hashed diag **bit8 RING_REGIONAL** (no kernel define, no extra arg, valid output). PCC and
cached replay identical to mask 0 on all four golden shapes.

## Result: a consistent LOSS on 3 of 4 shapes

median us, two relaunches with reversed mode order; `mask 4` = SKIP_IN0_RING_FORWARD (the addressable cost).

| shape | cfg (Ns,Pk,Sm,kb,nsb) | baseline | bit8 regional | mask 4 ceiling |
|---|---|---|---|---|
| 256x2048x2048 | (2,2,3,4,4) | 36.92 / 36.58 | **-5.4% / -6.5%** | +13.0% |
| 256x2048x6144 | (3,2,2,2,4) | 85.28 / 84.23 | **+2.8% / +4.2%** | +3.6% / +2.2% |
| 512x6144x2304 | (2,6,1,2,1) | 170.23 / 170.52 | **-8.7% / -8.9%** | +30.9% |
| 512x6144x4608 | (2,6,1,4,1) | 224.31 / 224.52 | **-4.0% / -4.3%** | +18.9% |

## Why — an exact offline route model (the useful part of this experiment)

`ring_topology_probe.py` replicates the planner's `find_near` placement and both ring strategies, then
reconstructs the logical->physical torus frame empirically from device hop distances and models
dimension-ordered routing. The model reproduces `get_worker_noc_hop_distance` **exactly (0 of 277
mismatches)**, so per-link loads are trustworthy:

```
physical torus 17x12; logical col -> phys [0,1,2,3,4,5,10,11,12,13,14]  (the gap is between logical x=5 and 6)
```

For 512x6144x2304 (96 cores, 12 rings, every ring edge carrying 7 shards of 128 KB):

| strategy | total hops | bisection-crossing edges | **busiest link** | measured |
|---|---|---|---|---|
| production (per-slice bank ring) | 683 | 34 | 28 shards (4 edges deep) | baseline |
| (a) bit8 as shipped | 576 (0.84x) | 34 | **35 (1.25x)** | **-8.7%** |
| (b) manhattan clustering | 679 | 24 | 35 (1.25x) | – |
| (c) deterministic bank-half split | 667 | 12 | 28 (1.00x) | – |
| (d) production partition + link-balanced order | 678 | 30 | **21 (0.75x)** | **+4.2%** |
| (e) bank-half partition + link-balanced order | 631 | 8 | 28 (1.00x) | – |

Three findings:

1. **The wall tracks PEAK LINK LOAD, not total hops.** bit8 cut total hops 16% and shard-hops 16% but raised
   the busiest link 25%, and measured ~25% worse ring cost (-8.7% of a wall whose ring costs 30.9%). At ~86
   GB/s per link the busiest-link model predicts the absolute ring cost within ~20% (4 overlapping edges x
   128 KB x 7 steps ~= 42 us vs 52.7 us measured). This reproduces a lesson already in the project record
   ("the right metric is busiest-link bytes, not total hops") that idea 1 had ignored.
2. **My clustering metric was the bug, and fixing it does not save the idea.** It summed *directed* hop
   distances, which on a torus are wildly asymmetric (1 hop one way, 10 the other), so it ranked "cheap in
   the routing direction" above "physically adjacent" — the shipped rings still straddled the bisection (34
   crossings, unchanged). Variant (b) fixes the metric and (c) removes crossings by construction (34->12),
   but NEITHER improves the busiest link, so neither is expected to win.
3. **Compactness actively reduces balancing headroom.** (e) has the fewest crossings of all (8) yet cannot
   get below 28 on the busiest link, because a compact partition funnels every ring into the same narrow
   corridor. Spread, not compactness, is the objective.

## What replaced it: link-balanced ordering (diag bit9 RING_BALANCED)

Same ring membership as production; only the visiting order changes. Groups are processed sequentially in
two passes, each picking the cycle that minimizes the peak GLOBAL link load it contributes to (tie-break:
total hops). Route model as above, charging both dimension orders since x-first vs y-first is not observable
from hop counts. PCC + replay identical to mask 0 on all four shapes.

| shape | cfg | baseline | bit9 balanced | mask 4 ceiling |
|---|---|---|---|---|
| 256x2048x2048 | (2,2,3,4,4) | 36.85 / 36.95 | **-9.7% / -10.8%** | +12.9% |
| 256x2048x6144 | (3,2,2,2,4) | 85.02 / 85.17 | **+1.2% / +2.9%** | +3.9% |
| 512x6144x2304 | (2,6,1,2,1) | 170.23 / 170.15 | **+4.1% / +4.2%** | +30.9% |
| 512x6144x4608 | (2,6,1,4,1) | 224.72 / 223.76 | **-1.9% / -1.4%** | +18.8% |

Partially validates the model — the predicted winner (512x6144x2304, busiest link 0.75x) does improve, and
stably in both relaunch orders — but it is **not shippable**, and the failures localize the remaining error:

- **512x6144x2304 vs 512x6144x4608 have IDENTICAL in0 ring problems**: same Pk/Ns/Sm => same 96-core
  placement, same 12 rings, and shard = W*M_block*kb = 64 tiles = 128 KB in both. The balanced order is
  therefore literally the same, yet one gains 4.2% and the other loses 1.6%. The only difference is that
  4608 carries **2x the in1 read traffic** (Nt 144 vs 72). So the model's blind spot is decisive: it balances
  in0 against in0 only, and can route in0 onto links that in1 DRAM reads, the reduction chain and the output
  writes already occupy.
- **256x2048x2048 (Sm=3) loses 10%**: with Sm>1 one permutation is shared by Sm mm-rings, so `peak` is
  dominated by the group's self-overlap and the lexicographic (peak, hops) objective will trade away a lot
  of distance for a marginal peak gain. A hop budget (as the existing PARETO objective uses) is missing.

## Recommended next iteration

1. **Model all four traffic classes, not just the in0 ring**: in1 DRAM reads (bytes = valid_k*valid_n per
   core, from that core's bank endpoint), the split-K reduction edges, and the output writes. All are known
   from the plan; the same route model applies. Minimize the peak over the combined map.
2. **Add a hop budget** to the balancing objective (minimize peak subject to hops <= (1+eps) x hop-optimal),
   which is the guard the current objective lacks and the likely fix for the Sm>1 regression.

Both are host-only and evaluable offline in `ring_topology_probe.py` before any device run — the model's
predictive accuracy on (a) and (d) is what makes that worth doing.

## Reproduction

```
export TT_METAL_DEVICE_PROFILER=1 ARCH_NAME=blackhole
python3 tools/mm_sweep/picker_gen/ring_topology_probe.py 16 192 72 2 6 1     # offline topology comparison
W=tools/mm_sweep/picker_gen/ablation_matrix_worker.py
python3 $W 512 6144 2304 2 6 1 2 1 0,256,4 1     # bit8 regional A/B
python3 $W 512 6144 2304 2 6 1 2 1 0,512,4 1     # bit9 link-balanced A/B
```
