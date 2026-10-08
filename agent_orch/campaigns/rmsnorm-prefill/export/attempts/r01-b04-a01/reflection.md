# r01-b04-a01 result: 1.0761 (ok)

## What happened vs expected
Valid on all shapes, accuracy unchanged (pcc 0.9999985, max_abs 0.022–0.024, same as baseline).
Per shape: h3584 1.116, h4096 1.126, glm h6144 1.052, h7168 1.016 (the last is just above the ±1% noise).
I expected ~2–3 µs off every shape. The small shapes got that (−1.8/−2.0 µs). The large shapes got
only −1.2/−0.4 µs.

## Why (profiler evidence, chip 1, zone timeline, µs from kernel start)
- AG window unchanged (~2.3–3.4 µs F_FABRIC), and the x*gamma pass hides under it as intended.
- Compute (TRISC kernel end) now finishes well before the writer (BRISC end):
  h3584 13.9 vs 15.8, h4096 13.8 vs 16.3, h6144 19.4 vs 22.2, h7168 22.5 vs 25.7.
  **The output drain (W_DRAIN) is now the critical path**: ~5.7 µs for 28 tiles, ~11 µs for 56 tiles
  (≈200 ns per 2 KB tile per core; 20 cores × 56 tiles × 2 KB ≈ 2.3 MB/chip in 11 µs ≈ 210 GB/s, well below
  BH DRAM write bandwidth). The writer issues one tile write at a time and calls
  `noc.async_writes_flushed()` + `pop_front` per 4-tile block (output_cb = 2 padded rows, so
  the flush per block isn't needed for CB space). It looks like this is latency-bound, not bandwidth-bound.
  On big shapes, compute savings mostly turn into extra writer slack, which is why the gain shrinks with width.
- Input read (R_INPUT end 4.2–8.9 µs) is unchanged and is the other large serial phase.

## Classification
win (small shapes clearly; large shapes marginal, limited by output drain)

## What a child of this node should try next
1. Make the output drain non-blocking: drop the per-block `async_writes_flushed` in
   `dit_rmsnorm_fused_worker_writer.cpp` W_DRAIN. output_cb already holds 2 padded rows, so wait for the
   whole row cumulatively (or per block without flushing), issue all writes, and do one barrier at the end.
   Alternatively, split the output writes across both RISCs/NoCs (the reader's NCRISC is idle after ~9 µs:
   let it drain half of the columns on NOC1). Expect this mostly to help h6144/h7168.
2. Spread each row over more cores (20 of ~120 cores used; column-split each tile-row across 2–3
   workers that share the stick/AG machinery) to shrink the input read, PRE and drain per core. This is
   structurally bigger, but it attacks all three serial phases at once.
3. PRE could also overlap better with input arrival, but R_INPUT itself (~4–9 µs at ~500 GB/s
   aggregate) seems close to the DRAM-read limit, so it is not the first target.
