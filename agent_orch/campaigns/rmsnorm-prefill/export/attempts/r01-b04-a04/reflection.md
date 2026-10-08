# r01-b04-a04 result: 1.2132 (ok)

## What happened vs expected
Valid on all shapes, and accuracy is unchanged: PCC 0.9999985, max_abs 0.022-0.024. So the DST accumulation is
numerically correct, which confirms that ELWMUL accumulates onto Dst on BH. Per shape vs the parent r01-b04-a03:

| shape | parent µs | this node µs | change |
|---|---|---|---|
| h3584 | 14.46 | 14.17 | -2.0% |
| h4096 | 15.27 | 15.43 | +1.0% |
| h6144 | 19.14 | 19.08 | -0.3% |
| h7168 | 21.09 | 20.98 | -0.5% |

Geomean is 1.2132 vs 1.2075 (+0.5%), inside the ±1% noise band, so this is NOT an improvement. I expected about
-1 µs per shape from a faster PRE. It didn't happen.

## Why (profiler evidence)
/tmp/r01b04a03/zones.py on device 1, call idx 10 (h3584) and 58 (h7168). Times are µs from the first marker of the
run, min-max over the 20 workers:

| zone end | parent h3584 | a04 h3584 | parent h7168 | a04 h7168 |
|---|---|---|---|---|
| R_INPUT | 3.76-4.80 | 3.77-4.45 | 14.69-15.76 | 13.79-15.30 |
| W_GAMMA | 4.29-5.29 | 4.66-5.04 | 14.46-15.36 | 14.57-15.39 |
| W_PUSH (PRE done + stick) | 5.13-6.58 | 5.37-5.86 | 15.98-17.65 | 15.17-17.15 |
| W_AGWAIT | 9.41-9.76 | 9.51-9.87 | 20.05-20.41 | 19.76-20.11 |
| TRISC end | 13.3-13.7 | 13.4-13.8 | 25.7-26.2 | 25.4-25.9 |
| W_DRAIN | 13.99-15.71 | 14.29-15.77 | 26.59-29.87 | 26.88-29.60 |

- **The PRE tail after the input did not shrink.** W_PUSH still ends ~1.4-1.8 µs after R_INPUT on both shapes, the
  same as the parent. One fp32 pack per row instead of one L1-accumulating pack per tile changed nothing measurable.
  So PRE was never pack-bound per tile. It keeps up with the input stream, and the ~1.4 µs tail is a fixed per-row
  cost after the last input block lands. That cost is the reconfig + reduce<SUM,REDUCE_ROW> (helper init, scalar
  tile) + transpose_init/transpose_tile/pack_reconfig + the stat-tile pushes and the W_PUSH NoC write + atomic
  barriers. It does not scale with width: the tail is the same at 28 and 56 tiles. That rules out the per-tile pack
  and per-tile math.
- **The stick-preempts-gamma poll never fired.** All 20 W_PUSH zones start after W_GAMMA ends (h3584: gamma ends
  4.66-5.04, push starts 4.70-5.08). PRE still finishes after BRISC's gamma issue loop. The poll is harmless and
  would matter only once PRE gets faster.
- h3584's -2% is borderline, and the kernel-start spread in the sampled call happened to be tighter. h4096's +1%
  is noise in the other direction. The AG section (F_COLLECT end -> W_AGWAIT end) is ~3.8 µs at h3584 here vs ~3.0
  in the parent. That is cross-chip arrival skew and dominates any sub-µs PRE change.
- The DST-accumulating PRE is a valid simplification: one pack per row, no L1-acc reconfig, bit-equivalent accuracy.
  It is worth keeping only as plumbing.

## Classification
neutral (within noise). The mechanism works correctly but targeted a non-bottleneck. The PRE tail is fixed per-row
overhead (reduce + transpose + handshake), not per-tile packing.

## What a child of this node should try next
1. **Attack the fixed PRE tail, not per-tile cost.** Add TRISC zones (DeviceZoneScopedN around reduce, transpose,
   x*gamma) to see which of the reduce helper, the transpose, or the reconfigs costs the ~1.4 µs. A candidate fix
   that removes two ops: produce the row sums directly in ROW 0 instead of reduce-to-col-0 then transpose.
   For example, transpose x^2 accumulation to the FPU's column reduce (reduce<SUM, REDUCE_COL> on the transposed
   input), or do the row reduce with a matmul against a ones tile whose output lands transposed. Then pack the stick
   tile once.
2. **Port this lineage onto the column split (r01-b03-a02 / a03: k=4, 80 workers, bank rotation).** It is still the
   biggest structural lever, as the parent said. With 7-14 tiles per core, a fixed per-row PRE tail matters even
   more, so (1) composes.
3. **Drain tail.** W_DRAIN ends 1-4 µs after TRISC end at h7168 (26.9-29.6 vs 25.4-25.9). Split writes across both
   NoCs: NCRISC is idle after R_INPUT, at ~15 µs here.
4. Don't spend more attempts on per-tile PRE throughput (fidelity, pack batching). The per-tile part is not on the
   critical path.
