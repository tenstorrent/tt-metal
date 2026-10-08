# r01-b02-a04 result: 0.9853 (ok)

## What happened vs expected
Valid on all shapes. Accuracy is bit-identical to the parent (PCC 0.9999985, max_abs 0.022-0.024). But it is slower than
the parent r01-b02-a03 (1.1721) on every shape: h3584 15.09 µs (parent 14.56), h4096 19.07 (15.75), h6144 24.94
(19.73), h7168 28.03 (22.20). I expected the drain tail to shrink by 1.5-3 µs. Instead the drain got 1.7x longer on the
wide shapes. A uniform 50/50 split of the output writes between NoC0 (BRISC) and NoC1 (NCRISC) is a large regression.

## Why (profiler evidence)
Per-core zones, device 1, µs from the call's first kernel start (/tmp/r01b02a04_percore.py: per run host id, per core
(x,y), zone ends). W_DRAIN end includes BRISC's wait for the reader's drain_sem, so it is the later of the two
halves. R_DRAIN ends at the same time.

h3584 (call idx 12), drain end per core, row y=2:
| x | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 10 | 11 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| parent (NoC0 only) | 12.26 | 12.37 | 12.73 | 12.88 | 13.33 | 13.46 | 13.78 | 13.69 | 13.98 | 13.05 | 13.21 |
| this (even NoC0 / odd NoC1) | 15.12 | 15.18 | 14.90 | 14.68 | 14.28 | 13.88 | 13.34 | 13.13 | 13.10 | 12.95 | 12.93 |
h7168 (call idx 51), row y=2: parent 18.8 (x=1) rising to 21.9 (x=7), 19.7-21.8 (x=10-14). This node: 28.3 (x=1)
falling to 21.6 (x=7), 26.1 (x=10) falling to 22.0 (x=14). y=3 shows the same pattern.

1. **The drain-end gradient follows the NoC direction.** NoC0 only (parent): drain end grows with x, so cores
   "downstream" on the eastward links finish last. With half the tiles on NoC1 (westward), the gradient flips:
   low-x cores finish last. On NoC1 the left-half cores (x=1..7) are much worse than on NoC0 (h7168 x=1:
   28.3 vs 18.8 µs). The right-half cores (x=10..14) are about equal or slightly better (h3584 12.9-13.1 vs
   13.1-14.0).
   The total write rate fell from ~235 GB/s (parent) to ~140 GB/s at h7168. So the NoC1 write path from the left half
   toward the DRAM banks is congested worse than NoC0 ever was. Adding a second NoC did not add usable capacity. The
   left-half NoC1 traffic for the x=9 DRAM column has to wrap westward through the x=0 column and the whole right
   half. NoC1 is also dimension-order reversed (y first), so it crosses other rows too.
2. **Compute is not affected.** TRISC end moved ~+0.4 µs at h3584 and ~+1.3 µs at h7168, the same as the AG-end
   shift (W_PUSH spread was larger in this run: ~1.5-2 µs later on cores (1-3, 2)). Those are the usual call-to-call
   and chip-skew effects. The kernel end is the drain.
3. The handshake protocol works. No hang, data is correct, and the reader's R_DRAIN and the writer's pop line up.
   The drain_sem / no-pop-on-reader scheme is reusable.

## Classification
flawed idea (as executed: a position-blind 50/50 NoC0/NoC1 split of the output drain). Do not retry the uniform
split. A position-aware split is untested and is the only variant worth trying (below).

## What a child of this node should try next
1. If you retry dual-NoC at all, make the share per core and position-aware: cores at physical x >= 10 (the right
   half, whose NoC1 drain was as good or better) give some blocks to NCRISC, and cores at x <= 7 keep NoC0 only. Pass
   a per-core RT arg "reader blocks mask/stride" (0 = BRISC only) and keep this node's drain_sem protocol. Or pick the
   NoC per tile by the destination DRAM bank's column (NOC_UNICAST_ADDR_X of the accessor's NoC address), taking the
   path that does not wrap. Expected gain is small (the right-half tail only, ~0.5-1 µs), so it is low priority.
2. Better levers for this lineage: port the BRISC gamma read (r01-b04-a03, 1.2075, which is the same compute and
   reader otherwise). Then cheaper PRE (accumulate x^2 in DST per block). Then the column split (r01-b03-a02/a03).
3. For the drain itself: the NoC0 gradient (cores further east finish later) says the bottleneck is the shared
   eastward links into the DRAM column, not DRAM. Reducing bytes on those links (e.g. a bf8 output is not allowed;
   but placing workers closer to / spread across both DRAM columns, i.e. choosing worker cores so each row's traffic
   splits between x=0 and x=9 sides) may help more than adding NoC1 traffic.
