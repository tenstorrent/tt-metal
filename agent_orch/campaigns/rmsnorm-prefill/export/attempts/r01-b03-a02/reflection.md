# r01-b03-a02 result: 1.0942 (ok)

Per shape: h3584 1.105, h4096 1.011, h6144 1.141, h7168 1.124. PCC is 0.9999985 and max_abs matches baseline on all shapes.
Parent r01-b03-a01: 0.987. Second-best node so far, behind r01-b01-a01 at 1.109.

## What happened vs expected
- **The repair worked exactly as diagnosed.** Op-to-op latency is 564-576 ns on every shape and chip. The parent had
  35-52 µs on its 2-kernel-group shapes. Kernel durations are now balanced across chips: h7168 d0..d3 is
  23.3/23.2/23.1/23.0 µs, where the parent had 31.0/28.2/20.7/22.7. The "cross-chip skew" in the parent's reflection was
  this device-side dispatch stall caused by uneven slices (2 kernel groups), not host launch order.
  Host per call grew to 14.7-15.4 µs (81 cores) vs 13.5-14.1 at baseline. The device is still slower than the host,
  so the queue backs up and the cross-chip skew does not return.
- k=4 (80 workers) beats k=3 on glm: 20.57 µs vs the parent's 21.82, both with one kernel group.
- h3584, glm and h7168 landed in the expected range (1.10-1.14). **h4096 did not (1.011).** See below.

## Why (profiler evidence, dev 0 zone medians, µs from kernel start)
| shape (tiles/core) | R_INPUT end | F_FABRIC | AG wait end | TRISC end | W_DRAIN start -> end |
|---|---|---|---|---|---|
| h3584 (7) | 1.7-3.9 | 5.6-8.5 | 8.8-9.4 | 11.5-13.7 | 9.3 -> 12.1-15.6 |
| h4096 (8) | 1.2-5.3 | 7.0-10.4 | 10.6-11.2 | 13.8-16.0 | 11.1 -> 14.0-18.6 |
| h6144 (12) | 3.2-5.4 | 7.2-11.4 | 11.7-12.2 | 13.7-17.2 | 12.2 -> 15.6-22.1 |
| h7168 (14) | 4.2-6.7 | 8.4-11.9 | 12.1-12.7 | 13.9-17.8 | 12.7 -> 16.4-23.4 |

1. **The drain is now the critical path, and its speed is set by aggregate traffic, not by per-core tiles.** The slowest
   core takes ~0.8 µs per tile in every shape (h3584 6.3 µs for 7 tiles, glm 10 µs for 12, h7168 10.7 µs for 14).
   80 cores x 2 KB / 0.8 µs is about 200 GB/s per chip, the same ~200 GB/s the baseline drain got with 20 cores.
   Adding cores did not raise write throughput. TRISC finishes 4-6 µs before the last BRISC drain on wide shapes.
2. **h4096 is the outlier, which points to DRAM bank camping.** It has 32 cols with slices starting at cols 0/8/16/24.
   Interleaved page -> bank = page % num_banks, and on this chip I assume 8 banks (not checked). Every row stride
   (32) and every slice start is then 0 mod 8, so all 80 cores read and write the same bank at the same time. Its R_INPUT
   (5.3 µs for 8 tiles) and drain (7.5 µs for 8 tiles) are both slower than h3584 with 7 tiles (3.9 / 6.3 µs), which is
   more than 8/7 would explain. The other shapes have slice starts that spread across banks: 28 -> starts 0/7/14/21,
   56 -> 0/14/28/42 = banks 0/6/4/2, 48 -> 0/12/24/36 = banks 0/4/0/4 (half). The baseline layout has the same
   problem: 32/48/56 cols are 0 mod 8, so all 20 row-workers march across the banks in phase.
3. The AG section, from F_COLLECT end through F_FABRIC to the AG wait end, is ~3-4 µs fixed. The leader-combine
   hop adds ~1-1.5 µs on leaders: followers' W_PUSH ends ~2.7 µs, leaders' ~5.7 µs at h3584.

## Classification
win (+9.4% geomean vs baseline, +10.9% vs parent). It repairs the parent's bad execution (bug: uneven slices
-> 2 kernel groups -> ~45 µs dispatch stall per op).

## What a child of this node should try next
1. **De-phase the DRAM bank access (likely the biggest lever, cheapest to try).** Give each core a start offset
   inside its slice. Reader and writer visit tiles in rotated order (start = (core_idx * something) % slice, wrapping),
   so concurrent cores hit different banks. The writer only needs the order changed in W_DRAIN: output_cb holds the
   resident row, but compute pushes blocks in order. Either rotate at block granularity, or wait for the whole row on the
   last block. Check the real bank count with `device->num_banks(BufferType::DRAM)` first. h4096 should then catch up
   with h3584 (~15 µs), and every shape's drain should shrink.
2. Split the output drain across both RISCs/NoCs. NCRISC is idle from ~4-10 µs to the end. Let it write half of each
   slice on NOC1. This attacks the x-dependent NoC0 write congestion seen in the parent: drain end grows with core x.
3. Port r01-b01-a01's gamma reorder (x*gamma under the AG wait, single post-AG pass). It is orthogonal and worth ~1-2 µs.
   With 7-14 tiles/core the post-AG compute is short, so most of the gain only appears once the drain is faster (1/2).
4. Keep: one kernel group (k | num_tile_cols) and k=4 / 80 workers. Don't go back to uneven slices without making the
   slice width a runtime arg.
