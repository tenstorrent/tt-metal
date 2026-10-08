# r01-b03-a03: De-phase DRAM bank access: each column-split core walks its slice from a greedy per-core rotation so concurrent cores hit different interleaved banks

## Motivation
Parent r01-b03-a02 (1.0942, 80 workers, k=4 column slices of 7/8/12/14 tiles) is drain-bound: the slowest core's
W_DRAIN runs at ~0.8 us/tile on every shape and finishes 4-6 us after TRISC on wide shapes (its reflection, and
`reports/r01-b03-a02` dev 0 h7168 call 8: TRISC end 19.6-21.5 us, W_DRAIN end 19.9-27.2 us).
h4096 is the outlier: 1.011 vs 1.10-1.14 on the other shapes, and 18.05 us vs 15.38 us for h3584 with only one more
tile per core (8 vs 7). Its R_INPUT (5.3 us / 8 tiles) and drain (7.5 us / 8 tiles) are both slower than h3584's.

DRAM tensors are interleaved, page p -> bank p % num_banks (8 on BH). Every core walks its slice in column order, and
page = row*cols + col_start + p. Simulating bank collisions per slice position across the 80 cores (8 banks, ideal 10
cores per bank; /tmp/r01b03a03/sim.py):

| shape | cols | stride % 8 | slice starts % 8 | max cores on one bank per step |
|---|---|---|---|---|
| h3584 | 28 | 4 | 0,7,6,5 | 10 (balanced) |
| h4096 | 32 | 0 | 0,0,0,0 | **80** |
| h6144 | 48 | 0 | 0,4,0,4 | 40 |
| h7168 | 56 | 0 | 0,6,4,2 | 20 |

The only balanced shape (h3584) is the fastest per tile, and the worst (h4096) is the outlier. The same in-phase
pattern hits the input read and the broadcast gamma face-row reads (all 20 rows of a slice read the same gamma pages
at the same moment, the hot spot r01-b04-a02 measured).

## Mechanism
Rotate each core's walk over its column slice: CB position p holds column (p + col_rot) % slice.
- Factory (`device/dit_fused_distributed_rmsnorm_program_factory.cpp`): for the column-split-eligible plain layout
  (no RoPE, one head, broadcast affine, resident POST, equal slices), choose col_rot per worker core greedily: in
  core order, pick the rotation that minimises the summed bank load over all slice positions given the cores already
  placed (bank count from `allocator()->get_num_banks(DRAM)`). This brings all four shapes to 10 cores/bank/step.
  Pass col_rot as reader RT arg 4 and as a writer RT arg after col_start. Every other layout gets 0.
- Reader: input page and broadcast weight/bias page use the rotated column.
- Worker writer: W_DRAIN writes CB position p to the rotated output column.
- Compute is unchanged: for this config (no RoPE) it only pairs input position p with gamma position p, and the
  sum of squares is order-independent.
Still one kernel group, same CBs, no CT-arg change, one extra RT word per reader/writer.

## Why this is not a repeat
No node has changed the DRAM page visiting order. r01-b02-a02 changed read/drain depth (trids, one flush per row),
and found the drain tail is contention, not flush serialisation. r01-b01-a02/r01-b04-a02 changed gamma read
scheduling. This is the parent reflection's top recommendation, untried.

## Expected effect and risk
h4096 should improve the most (target ~15.5-16 us, close to h3584 scaled by 8/7). h6144 should gain some (~0.5-1.5 us),
h7168 a little, h3584 ~no change (already balanced; rotation choices there are only a reshuffle). Estimated geomean
+3-5% over the parent.
Risks: a rotation mismatch between reader and writer would mis-place output columns -> PCC fail (clear signal).
If the drain is NoC-link bound rather than bank bound, all shapes except h4096 stay within noise; then h4096's
change tells us whether the bank hot spot was real.
