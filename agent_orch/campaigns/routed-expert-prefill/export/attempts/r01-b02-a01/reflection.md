# r01-b02-a01 result: 1.4017 (ok)

## What happened vs expected
Valid on all 9 shapes with bit-identical accuracy (PCC / max_abs / mean_abs equal to baseline to
every printed digit), so the reader/compute/writer row mapping agrees under every group size.
Speedups (noise ±1%): kimi-k3 1.73x (G=1, 28 experts x ~3 tiles), dsv4-flash 1.54x, kimi-k2.7
1.49x, glm 1.41x, dsv3 1.40x, mistral 1.35x, dsv4-pro 1.32x, minimax-m3 1.25x, gpt-oss 1.20x.
Roughly as predicted in direction; kimi-k3 and dsv4-pro gained less than hoped.

## Why (best explanation, with profiler evidence)
Test-log weight throughput rose from a flat ~200-265 GB/s to ~320-360 GB/s on every non-bias
shape (dsv3 330, kimi-k2.7 363, kimi-k3 342, dsv4-pro 360), consistent with the baseline being
limited by only 11 weight-issuing cores: groups give 4-8x more issuers and drop the phantom
M-rows (per_core_M 1 on 8 rows -> ceil(count/G) on G rows). The new plateau (~340 GB/s, ~2/3 of
BH DRAM peak) is again uniform across shapes, so the next limit is likely per-K-block
handshake/latency in the read->multicast->compute pipeline or DRAM-bank contention, not the
schedule.
Shape-specific shortfalls:
* dsv4-pro: G=1 with 12 experts over 8 groups -> rows 0-3 run 2 experts, rows 4-7 run 1, so
  half the grid idles for ~1/3 of the op (round-robin dealing, no balancing).
* kimi-k3: 28 over 8 groups -> 4 vs 3 experts per row, same imbalance (~14%).
* gpt-oss (bias, 9-tile experts, G=4, per_core_M 3): only 115 GB/s, so it is compute/overhead
  bound rather than weight-read bound; minimax-m3 (4 experts, G=2, 1 expert per group) has no
  overlap between experts at all.

## Classification
win

## What a child of this node should try next
1. Balance the expert -> group deal: when experts % groups != 0 the last wave runs on part of the
   grid. Options: pick G per launch to minimise ceil(E/groups) * per-expert cost (e.g. dsv4-pro
   G=2 gives 3 per group, all busy), or split the leftover experts' N (or M) across the idle
   groups instead of leaving them idle.
2. Attack the new ~340 GB/s plateau: deeper weight prefetch (3-slot cb_in1_* or issuing the next
   block's DRAM read before the current multicast/handshake), now that G=1 groups do no
   multicast at all on weights and could stream straight into the CB.
3. For per_core_M >= 2 (all G<8 shapes now) the matmuls still re-unpack every bf4 in1 tile once
   per tile-row (out_subblock_h = 1); a taller subblock (h = per_core_M, w reduced to fit DST)
   would cut unpack work — matters most for gpt-oss/minimax which look compute-bound.
