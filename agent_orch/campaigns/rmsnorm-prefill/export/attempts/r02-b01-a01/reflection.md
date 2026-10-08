# r02-b01-a01 result: 1.1058 (ok)

## What happened vs expected
The run is valid with bit-identical accuracy (PCC 0.9999985, max_abs 0.022-0.024). But it is a clear regression
from the round root r01-b04-a04 (1.2132) on every shape:

| shape | root µs | this µs | change |
|---|---|---|---|
| h3584 | 14.17 | 15.53 | +9.6% |
| h4096 | 15.43 | 16.57 | +7.4% |
| h6144 | 19.08 | 21.13 | +10.7% |
| h7168 | 20.98 | 23.31 | +11.1% |

I expected the drain tail to shrink by 1-2 µs. Instead the drain got slower, and the per-core gradient flipped.

## Why (profiler evidence)
`drain_cmp.py` (in this node dir), run on `reports/<node>/.logs/profile_log_device.csv`. Values are medians over
measured calls 3-12 on all 4 chips, in µs from kernel start. The DRAIN table is W_DRAIN end minus the last
W_AGWAIT end.

| shape | AG end root -> this | TRISC max | DRAIN max | DRAIN max - TRISC max |
|---|---|---|---|---|
| h3584 | 7.82 -> 8.35 | 11.80 -> 12.34 | 13.94 -> 15.22 | 1.88 -> 2.80 |
| h4096 | 8.82 -> 8.83 | 13.05 -> 13.07 | 15.19 -> 16.05 | 2.22 -> 3.22 |
| h6144 | 10.10 -> 10.36 | 15.39 -> 15.65 | 18.65 -> 20.50 | 3.11 -> 4.89 |
| h7168 | 10.47 -> 11.08 | 16.17 -> 16.43 | 19.67 -> 22.15 | 3.47 -> 5.50 |

Per-core drain end - AG at h7168 (workers occupy grid rows y=2,3; x is the translated/NoC0 column):
```
           x=  1    2    3    4    5    6    7   10   11   12   13   14   15
root  y=2     6.8  6.8  7.0  7.1  7.7  8.1  8.4  8.4  8.8  9.1  7.3  7.2  7.6
this  y=2    10.5 10.8 10.3  9.5  8.9  8.0  7.2  6.9  7.0  7.0  6.8  6.8  7.0
root  y=3     7.0  6.9  7.2  7.3  7.9  8.3  8.6  8.6  9.0  9.4  9.7
this  y=3    10.7 10.7 10.2  9.6  8.9  8.0  7.5  6.9  6.9  7.0  7.2
```
1. **The right-half rule worked.** Right-half cores (x>=10) send east-column tiles on NoC1, west to x=9, and
   west-column tiles on NoC0 through the wrap. Their drain improved by 1.5-2.4 µs (8.4-9.7 -> 6.8-7.2). They are
   now flat across x.
2. **The left-half rule failed badly.** Left-half cores send west-column tiles on NoC1, and x=1-3 went from the
   fastest (6.8) to the slowest (10.5). The gradient now rises toward x=1, the mirror of NoC0's rise toward x=7.
   The numbers fit one shared link: 14 left-half cores x 28 west-bank tiles x 2 KB = 0.8 MB in ~10 µs, about
   80 GB/s, which is one NoC link. All of this traffic enters the west DRAM column at rows 2-3 and then has to
   move vertically to the four bank endpoints. So the single column-0 link next to the worker rows is saturated
   (or the 1->0 row links feeding it, if NoC1 also routes x-first). My model assumed NoC1 goes y-first in the
   source column, which would have spread the traffic. The measurement says the traffic did not spread.
3. Taken together: the drain limit is **the vertical links inside the DRAM columns next to rows 2-3**, where all
   20 workers sit. It is not the eastward row links alone. Moving a stream to the other NoC helps only when
   that stream enters the DRAM column at a different place or direction than the traffic already there. The
   left-half NoC1 stream made one column-0 link carry what used to be spread over the NoC0 wrap path.
4. AG end and TRISC are within ~0.5 µs of the root (call-to-call / chip skew; W_PUSH and R_INPUT unchanged), so
   the whole regression is in the drain. Running NoC1 writes from BRISC (`noc_local_state_init(other)` + a second
   `Noc`) works: no hang, correct data. The plumbing is reusable.

## Classification
Flawed idea as executed: the destination-column rule for left-half cores is wrong on this placement. Partial
signal: the right-half half of the rule is a real per-core win of ~2 µs on the slowest cores. Since only those
cores were on the critical path at the root, that part alone should help.

## What a child of this node should try next
1. **Keep only the right-half rule** (one-line change in `want_noc1`: `!core_left_half && !dst_west`). Left-half
   cores go back to all-NoC0. That was the root's fast group (6.8-8.4); removing the right-half cores' wrapped
   east-bank traffic from the left half's row links should make it a little faster. Right-half cores stay at
   ~7.0. Expected: h7168 DRAIN max - AG ~7.5-8 vs 9.7 at the root, -1.5 to -2 µs per wide shape, so the score
   should rise above the root. Judge with `drain_cmp.py`: the per-core table should be flat at ~7.
2. **Worker placement (factory only).** All 20 workers sit in grid rows 2-3, so every write enters the DRAM
   columns at the same two rows. Spread the 20 workers over all 10 grid rows (2 per row: one in each half, or
   both near the east column), with the forwarder near the middle. Each row's traffic then enters the DRAM column
   at its own row, and the vertical column links share the load. This attacks the bottleneck found here directly,
   and it also helps the input read (same columns, the other direction).
3. Don't send left-half west-column traffic on NoC1 again while the workers are clustered in a couple of rows.
4. A finer rule (per worker x bank, using the physical NoC0/NoC1 endpoint rows from blackhole_140_arch.yaml
   dram_views) could choose by whole-path link load, not by x only. Try it only after 1 and 2.
