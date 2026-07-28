# Whole-op link-balanced in0 ring ordering (diag bit10) — +6% on the deep shape, neutral elsewhere

Follow-on from `IDEA1_RING_TOPOLOGY.md`. Host-only, correctness-preserving: production ring MEMBERSHIP is
kept, only the visiting order changes (the ring protocol is correct for any permutation). PCC + cached replay
identical to mask 0 on all four golden configs; **111/111 mask-0 regression**.

## The two corrections bit9's measurements demanded

**(1) Model the background traffic.** bit9 balanced the in0 ring against itself. That regressed
512x6144x4608 while winning on 512x6144x2304 even though the two shapes have *identical* ring problems —
same Pk/Ns/Sm => same 96-core placement, same 12 rings, shard = W*M_block*kb = 64 tiles = 128 KB in both, so
literally the same chosen order. Charging the fixed traffic (in1 DRAM reads, in0 own-shard read, split-K
reduction, output writes) onto the link map first explains it:

| shape | background peak | production peak | ring's share of the peak link | best balanced |
|---|---|---|---|---|
| 512x6144x2304 (Nt=72) | 6.72 MB | 8.86 MB | **24%** | 7.63 MB (0.86x) |
| 512x6144x4608 (Nt=144) | **10.52 MB** | 11.22 MB | **6%** | 10.52 MB (0.94x) |

On 4608 the busiest link is already saturated by in1 reads and the ring adds only 0.70 MB, so there was no
headroom to win — bit9 was buying +5.7% ring hops for nothing. `ring_link_model.py` computes this offline.

**(2) Budget the latency terms.** Per-step ring time ~ `max(worst_edge_hops * hop_latency, shard_bytes /
link_bw)`. bit9 minimized peak load with only a total-hops tie-break, which cost **-10% on 256x2048x2048** —
a 24 KB shard, where the ring is latency-bound and the worst directed edge sets the per-step time. bit10
therefore constrains candidates to (a) never exceed `(1+10%)` of production's total hops, and (b) never
worsen the group's worst edge **when the shard is small** (< 64 KB); at 128 KB the bandwidth term dominates
and that freedom is exactly what buys the win. Both budgets are anchored on the order production itself would
pick, so production's order is always feasible — the search cannot be worse than production on either
latency metric, only different in which links it uses. Finally the reordering is adopted only if the
predicted peak drops >= 2%, else production's exact orders are kept.

## Result (median us; three relaunches, mode order forward / reversed / bit10-first)

| shape | cfg (Ns,Pk,Sm,kb,nsb) | shard | **bit10** | bit9 (previous) |
|---|---|---|---|---|
| 256x2048x2048 | (2,2,3,4,4) | 24 KB | **+0.2 / +1.2 / +1.8%** | -11.3 / -8.3% |
| 256x2048x6144 | (3,2,2,2,4) | 32 KB | **+1.5 / -0.0 / +0.4%** | +3.3 / +0.8% |
| 512x6144x2304 | (2,6,1,2,1) | 128 KB | **+5.8 / +6.4 / +6.1%** | +3.7 / +3.9% |
| 512x6144x4608 | (2,6,1,4,1) | 128 KB | **+0.0 / -0.9 / -0.0%** | -2.1 / -1.8% |

- **+6.1% median on 512x6144x2304** (170.5 -> 160.2 us), stable across three relaunches — and *better than
  bit9's +3.9%*, i.e. modelling the background does not merely avoid bit9's regressions, it finds a better
  order on the shape where the ring does load the critical link.
- Neutral within +-1% on the other three, including both shapes bit9 regressed.
- Ring-forward cost on 512x6144x2304 is 52.7 us (30.9% of the wall), so +6.1% = ~10.4 us recovers **~20% of
  the entire ring-forward cost from ordering alone**, with no change to bytes, placement, or kernels.

## Model accuracy (the reason to keep iterating offline)

Predicted peak-load ratio vs measured outcome, over the six (strategy, shape) pairs measured so far:

| case | predicted peak | measured |
|---|---|---|
| bit8 regional, 2304 | 1.25x (worse) | -8.7% |
| bit9 in0-only, 2304 | 0.86x | +3.9% |
| bit9 in0-only, 4608 | 0.94x but background-dominated | -1.8% |
| bit9 in0-only, 2048 (Sm3) | 1.00x, worst edge inflated | -10% |
| bit10, 2304 | 0.86x, edge free (128 KB shard) | **+6.1%** |
| bit10, 4608 | 0.94x, background-dominated | -0.0% |

The model gets the sign right in every case once both the background and the latency regime are included.
It mis-sizes the *magnitude* where the peak is background-dominated (4608: predicts a 6% peak drop, delivers
0%), which is the known remaining weakness.

## Open items

- **Tighten the adopt gate**: 4608 passes the >=2% predicted-peak gate yet gains nothing, because its peak
  link is 94% background. Gating on the *ring's share* of the production peak link (>= ~15%) would make that
  shape exactly neutral by construction. Not adopted here: with four shapes it cannot be validated without
  overfitting — it needs the 60-shape corpus.
- **256x2048x6144 forgoes bit9's +3.3%**: its 32 KB shard triggers the edge cap. The 64 KB latency/bandwidth
  crossover is a first-principles estimate, not a measured threshold; a shard-size sweep would place it.
- The DRAM endpoint row is approximated by the bank-adjacent worker's row (exact for undisplaced cores).
- Not yet promoted to mask 0: that needs the 60-shape corpus (no-regression gate) per the standing adoption
  rule, at which point bits 8/9 should be dropped and bit10's logic folded into `optimize_in0_ring_order`.

## Reproduction

```
export TT_METAL_DEVICE_PROFILER=1 ARCH_NAME=blackhole
python3 tools/mm_sweep/picker_gen/ring_link_model.py 16 192 72 2 6 1 2 1      # offline: Mt Kt Nt Ns Pk Sm kb nsb
W=tools/mm_sweep/picker_gen/ablation_matrix_worker.py
python3 $W 512 6144 2304 2 6 1 2 1 0,1024,512 1                               # base / bit10 / bit9
```
