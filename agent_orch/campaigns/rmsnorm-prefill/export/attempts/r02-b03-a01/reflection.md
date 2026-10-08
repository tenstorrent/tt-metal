# r02-b03-a01 result: 1.0312 (ok)

## What happened vs expected
Valid on all shapes. Accuracy is bit-identical to the lineage (PCC 0.9999985, max_abs 0.0217-0.0243). But it is a large
regression against the round root r01-b04-a04 (1.2132) on every shape: h3584 16.61 µs (root 14.17), h4096 17.68 (15.43),
h6144 22.73 (19.08), h7168 25.10 (20.98). I expected the drain tail to shrink by 0.5-2 µs. Instead it grew by 2-4 µs.
The protocol works (no hang, correct data, R_DRAIN and W_DRAIN end together), so the loss is entirely drain throughput.

## Why (profiler evidence)
`drain.py` (in this dir) on `reports/<node>/.logs/profile_log_device.csv`: measured calls, all 4 chips, medians.
Times are µs relative to the last W_AGWAIT end on the chip (AG). Root r01-b04-a04 -> this node:

| shape | TRISC max end - AG | W_DRAIN max end - AG | W_DRAIN mean end - AG | kernel end |
|---|---|---|---|---|
| h3584 | 3.99 -> 4.00 | 5.88 -> 7.91 | 5.12 -> 5.85 | 14.15 -> 16.47 |
| h4096 | 4.23 -> 4.23 | 6.46 -> 8.58 | 5.50 -> 6.34 | 15.41 -> 17.51 |
| h6144 | 5.30 -> 5.30 | 8.44 -> 11.88 | 7.20 -> 8.32 | 18.87 -> 22.39 |
| h7168 | 5.81 -> 5.81 | 9.42 -> 13.38 | 8.02 -> 9.25 | 19.89 -> 23.79 |

Everything up to the AG and the compute is unchanged. Per-core drain end at h7168, dev 0 (end - AG, µs, writer end ==
reader end because the writer waits drain_sem):
```
            x=2   3    4    5    6    7   10   11   12   13   14
root  y=2   6.5  6.9  7.1  7.6  8.0  8.3  8.3  8.6  8.9  6.8  7.0     (NoC0 only: grows with x)
this  y=2  13.4 13.4 12.5 11.4 10.3  7.8  9.4  8.4  7.3  6.5  6.5     (gradient flipped, ~1.5x steeper)
root  y=3   6.7  7.2  7.3  7.9  8.2  8.4  8.5  8.8  9.2
this  y=3  11.4 11.5 10.0  8.9  7.3  6.7  7.5  6.7  6.6
```
All 4 chips (different column harvesting) show the same picture.
1. **The right-half cores got faster** (x=12-14: 6.5-7.3 vs 6.8-8.9). Their NoC1 tiles go to the x=9 banks, 1-5 hops west.
2. **The left-half cores got much slower, and the farthest-west core is worst** (x=2: 13.4 vs 6.5). Their NoC1 tiles go to the
   x=0 banks, which is only 2-7 hops by my model, yet these are the slowest writes on the chip. So hop count / total link
   load (noc_drain_sim.py: max link 72 -> 31) does not predict the drain time on this machine. The x=2 core is the most
   *downstream* injector on the westward NoC1 flow toward the x=0 column (x=7..3 traffic passes through its router first),
   and it starves, the mirror of the NoC0-only picture where the most-eastern cores starve. Per-router arbitration
   (through-traffic beats local injection) on the shared final segment into one DRAM column decides the tail, not path
   length.
3. This also re-explains r01-b02-a04 and r01-b03-a04: in all three dual-NoC nodes the left-half / low-x cores are slow on
   NoC1 whatever their destination (x=9 there, x=0 here). I was wrong that b02-a04 failed because of long wrap paths.
   The robust observation is: NoC1 writes from the right half are good, NoC1 writes from the left half are bad.

## Classification
flawed idea (as a hop-count model). The destination-aware split moves the starvation to the westmost cores instead of
removing it. The handshake/plumbing is fine and reusable.

## What a child of this node should try next
1. If dual-NoC is retried at all, use the only variant all three nodes support: **right-half cores only** put their x=9
   bank tiles on NoC1 (they were the fastest cores here, 6.5 µs), **left-half cores stay NoC0-only**. With this node's
   code that is a one-line change in `dit_rmsnorm_drain_noc1_bank_mask()`: return 0 when the physical x < 9. That should
   take the root's slowest cores (x=10-12, AG+8.3-9.2) down to ~AG+7 and leave the left half at the root's AG+6.5-8.3.
   Expected gain is modest (~0.5-1 µs on h6144/h7168).
2. Better: attack the starvation itself. The drain tail is set by the most-downstream injectors on a shared link. Options:
   pace the upstream cores (e.g. a per-core start delay or a cap on outstanding writes for the cores that finish early),
   or reorder each core's writes so different cores feed different DRAM columns at the same moment (half the cores start
   with the x=0 banks, half with x=9, then swap), so no single column's ingress is hammered by everyone at once.
3. Or move work away from the drain entirely: the drain is ~2.24 MB/chip at ~250 GB/s; the other branches' levers
   (column split, PRE tail) don't change that. Picking worker cores that are spread over the grid (not the first 20
   row-major cores, which are all in rows y=2/3 and share the same row links) is untried and may spread NoC0 load
   across more rows.
