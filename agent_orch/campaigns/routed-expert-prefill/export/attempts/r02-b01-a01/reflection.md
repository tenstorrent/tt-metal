# r02-b01-a01 result: 1.4854 (ok), a regression vs the parent r01-b04-a04 (1.5011)

## What happened vs expected
All 9 shapes are valid and bit-identical to the baseline (PCC, max_abs and mean_abs equal to every
printed digit). The picker switched exactly the predicted shapes to 2 groups of 5 rows (per_core_M 1),
and the profile confirms the layout: every row of both groups ends within ~2-5% of the others. No hang.
The speed did not improve. Per shape vs the parent (noise ±1%):

| shape | parent µs | this µs | Δ | layout |
|---|---|---|---|---|
| dsv4-flash | 331.2 | 348.5 | **+5.2% (slower)** | 5x2 (was 2x5) |
| kimi-k2.7 | 764.3 | 775.7 | +1.5% (slower) | 5x2 (was 2x5) |
| dsv4-pro | 1128.0 | 1137.1 | +0.8% (noise) | 5x2 (was 2x5) |
| others | | | dsv3 +2.2%, glm +1.6%, kimi-k3 +1.0%, mm3 +0.4%, gpt-oss +0.9%, mistral -3.9% | unchanged code path |

The unchanged shapes moved by run-to-run noise (mistral's ±5% is known). The predicted 5-15% gains
on the three switched shapes did not appear; dsv4-flash lost outside noise.

## Why (best explanation, with profiler evidence)
Per-row TRISC1 kernel ends (reports/r02-b01-a01/.logs/profile_log_device.csv, /tmp/r01b01a04/rows.py),
10 measured calls:
- The tail is gone, as designed: kimi-k2.7 rows end at 741-802 µs (parent: 2-expert rows 507-660,
  3-expert rows 702-781). dsv4-pro rows end at 1101-1151 (parent: 763-966 vs 1066-1160). dsv4-flash
  rows end at 309-390 (parent: 164-320 vs 294-345).
- But the balanced 2-group run is no shorter than the parent's tail-ended run. Per group the
  per_core_M=1 expert time is much slower than a lone group's: kimi-k2.7 ~127 µs per expert (baseline
  single 8-row group: 97), dsv4-flash ~84 (65), dsv4-pro ~189 (133). Total weight throughput with
  2 groups is only ~390 GB/s on kimi-k2.7 / dsv4-pro and ~340 GB/s on dsv4-flash. In the parent the
  5-group phase moved ~440 GB/s.
- So the model's flaw is a single shared DRAM ceiling. **Achievable DRAM bandwidth depends on how
  many groups (weight streams) run concurrently**: 1 group ~250 GB/s, 2 groups ~340-390, 5 groups ~440.
  The extra streams buy back what the tail costs in the parent layout, so the old units+chunks model's
  picks were (by luck or not) the right ones. The fluid model with a fixed BW over-rates few-group
  layouts.
- dsv4-flash (K=128 tiles, the shortest weight blocks) lost the most: its 2-group run is the least
  bandwidth-efficient (~340 GB/s), consistent with per-stream latency/issue limits rather than DRAM peak.

## Classification
flawed idea (as modelled): moving tail work onto fewer, larger groups loses DRAM concurrency. The
implementation is correct (bit-identical, exact picks), but a child should revert adaptive_chunk.hpp to
the parent's units+chunks scoring, or recalibrate the model with a concurrency-dependent BW.

## What a child of this node should try next
- Start from r01-b04-a04, not this node (or revert this header change).
- The real limit on few-group layouts is per-group weight-stream throughput (~170-200 GB/s per 5-row
  group at per_core_M 1). A lever that raises it would make both the 2-group layout AND the 5-group
  tails faster: more outstanding DRAM reads per group column, e.g. split each column's gate/up/down
  weight reads over more RISCs/NoCs (receivers read a slice and the column shares by multicast), or
  issue reads to more banks at once. Measure per-group throughput first.
- If the cost model is revisited: make BW a function of the active group count (≈250/340-390/440
  GB/s for 1/2/5 groups on Blackhole) and re-fit; with that, today's 2x5 picks for kimi-k2.7 / dsv4-pro /
  dsv4-flash probably stay. A wave-wise layout switch (5 groups for full rounds, merge idle groups for the
  last round) would then be worth at most the tail difference at 390 vs 270 GB/s (~5-7% on kimi-k2.7 /
  dsv4-pro), with a large implementation cost (weight CB slot / semaphore realignment between layouts).
- The fluid simulator and calibration scripts are in /tmp/r02b01a01 (sim.py, fit3.py, intmodel.py, t.cpp).
