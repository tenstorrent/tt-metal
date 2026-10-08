# r01-b03-a04 result: 1.1653 (ok)

Per shape (parent r01-b03-a03 in brackets): h3584 14.59 µs / 1.165 (15.29 / 1.112), h4096 16.06 / 1.137
(16.39 / 1.114), h6144 20.29 / 1.157 (20.44 / 1.149), h7168 21.63 / 1.203 (22.60 / 1.151). PCC 0.9999985 and max_abs
are unchanged on every shape (same data, only the write NoC changed). Geomean +3.0% over the parent. h3584 (-0.70 µs),
h4096 (-0.33) and h7168 (-0.98) are outside the ±1% noise. h6144 (-0.15 µs, -0.7%) is inside it. This is not the
campaign best (r01-b04-a03, 1.2075). It is the best column-split node.

## What happened vs expected
I expected the far-core drain tail (AG+10 µs at h7168) to drop to ~AG+6 µs. It only dropped ~0.7 µs. The NoC split
works mechanically: the even and odd halves finish together on every core. But **the position gradient flipped instead
of flattening**.

## Why (profiler evidence)
`split.py` (in this node dir) on `reports/<node>/.logs/profile_log_device.csv`: medians over calls 3-12 and all 4 chips,
µs relative to the last W_AGWAIT end.

| shape | DRAIN max end - AG, parent -> a04 | TRISC max end - AG | BRISC max kernel end |
|---|---|---|---|
| h3584 | 6.33 -> 5.41 | 4.3 | 15.19 -> 14.16 |
| h4096 | 6.97 -> 6.34 | 4.5 | 16.46 -> 15.95 |
| h6144 | 9.26 -> 8.70 | 5.0 | 20.15 -> 19.52 |
| h7168 | 10.27 -> 9.57 | 5.2 | 21.28 -> 20.82 |

Per-core drain end at h7168 (end - AG, µs; W_DRAIN waits on drain_sem, so it ends at the same time as R_DRAIN):
```
parent (all NoC0)   x=1    4    7   11   15        a04 (even NoC0 / odd NoC1)   x=1    4    7   11   15
y=2                 3.6  4.1  5.6  6.2  7.5                                y=2   9.2  9.0  8.5  7.9  5.5
y=5                 5.5  6.1  7.9  8.7  9.7                                y=5   8.0  7.7  7.2  6.9  6.3
y=8                 6.5  7.4  8.6  9.6 10.2                                y=8   4.8  6.4  6.1  6.6  6.8
```
- With all writes on NoC0, the slow cores were bottom-right (high x, y). With half the tiles on NoC1, the slow cores are
  top-left (low x, y). Those are the NoC1 (-x/-y) mirror of the NoC0 congestion, and they bound each core because both
  halves must finish. So each NoC is directionally congested, and a fixed 50/50 split per core moves the worst case to
  the cores that are far on the other NoC.
- The worst-case drain still runs about 4.4 µs after compute on h7168 (AG+9.6 vs TRISC AG+5.2), so the drain is still
  the critical path. A few cores are always fast (e.g. (12,2), (5,5), (1,6-8)). These look like the cores closest to a
  DRAM endpoint on both NoCs.
- The aggregate write rate went up only a little (h7168: 2.24 MB/chip over ~8.5 µs of drain, ~260 GB/s). Two NoCs
  didn't double it, so part of the limit is shared beyond the per-NoC links (DRAM-side write ingress or the DRAM column
  links).

## Classification
win (small: +3.0% geomean over the parent, 3 of 4 shapes outside noise). The dual-NoC drain is real but limited by
symmetric congestion. It is a partial fix, not a flawed idea.

## What a child of this node should try next
1. **Per-core NoC split ratio instead of 50/50.** The two gradients are mirror images, so give each core a split
   weighted by position: cores near the top-left send most tiles on NoC0, cores near the bottom-right send most on NoC1.
   A simple version: share_noc1 = clamp(((x_rank + y_rank) / (max)) ...), or pick per core the NoC whose measured
   parent drain was faster (parent table above vs. the a04 R_DRAIN table) and give 75/25. Pass it as one reader+writer RT
   arg (the number of leading/trailing CB positions on NCRISC). Judge with `split.py`: the per-core table should be flat.
2. **Port r01-b04-a03's pieces onto this lineage** (x*gamma under the AG, gamma read on BRISC at kernel start, trid
   input read). TRISC end would move from AG+5.2 to ~AG+2.5 µs. That pays only to the extent that the drain can start
   earlier, but the drain is throughput-bound from the first output tile, so expect a smaller gain than b04 got.
   Careful: BRISC gamma + the drain_sem handshake both touch the writer. Keep the reader-side drain after its reads.
3. Fewer, fatter writes: the gain from more NoC paths was small, so check whether the limit is on the DRAM side. Try
   NOC write posted mode (no ack round-trip, `noc.async_write` with posted option) for the output tiles, then flush
   before the pop.
