# r02-b02-a03 result: 1.2396 (ok)

## What happened vs expected
Valid on all shapes. PCC 0.9999985 everywhere, max_abs 0.0204-0.0240 (gate 0.05), so C = ones * S^T has the right
orientation (row 0 = per-row sums) and the same precision as the reduce + transpose it replaces.
Per shape, this node / grandparent r02-b02-a01 / parent r02-b02-a02 (µs, chip mean):
h3584 13.95 / 13.87 / 13.88, h4096 15.12 / 15.20 / 15.09, h6144 18.67 / 18.73 / 19.09, h7168 20.38 / 20.42 / 20.77.
Score 1.2396 vs 1.2385 (grandparent) and 1.2292 (parent): +0.1% over the best node, inside the ±1% noise band.
I expected ~1-2% (stat ready 0.2-0.3 µs earlier). It wasn't.

## Why (profiler evidence)
Parent's `tl.py` on reports/<node> (medians over measured calls, all 4 chips, µs from first worker kernel start;
full output in `tl_out.txt`). a01 / a02 / a03:

| shape | R_INPUT end max | stat ready (W_PUSH start) min | W_PUSH dur min | W_PUSH end max | F_COLLECT end | AG end max |
|---|---|---|---|---|---|---|
| h3584 | 3.37 / 3.36 / 3.44 | 3.75 / 3.17 / 3.47 | 0.47 / 0.83 / 0.47 | 5.06 / 4.85 / 4.93 | 4.94 / 4.77 / 4.81 | 8.30 / 7.63 / 7.83 |
| h4096 | 3.96 / 3.87 / 3.94 | 3.79 / 3.20 / 3.67 | 0.48 / 0.85 / 0.48 | 5.80 / 5.58 / 5.56 | 5.66 / 5.46 / 5.43 | 8.58 / 8.45 / 8.47 |
| h6144 | 5.61 / 5.67 / 5.75 | 5.20 / 4.85 / 5.11 | 0.48 / 0.84 / 0.48 | 7.70 / 7.50 / 7.63 | 7.55 / 7.36 / 7.48 | 10.38 / 10.62 / 10.48 |
| h7168 | 6.46 / 6.45 / 6.50 | 6.18 / 5.84 / 6.22 | 0.48 / 0.84 / 0.48 | 8.18 / 8.13 / 8.15 | 8.05 / 8.02 / 8.04 | 11.56 / 12.04 / 11.59 |

1. **The writer side is repaired**: W_PUSH is back to 0.48 µs (the parent's gather cost 0.36 µs is gone).
2. **But the stat is barely earlier than the grandparent's**: -0.28/-0.12/-0.09/+0.04 µs at min, vs the parent's
   -0.35..-0.6 µs. Removing ONE of the two post-S stages (reduce or transpose) saved almost nothing. So the
   ~1 µs PRE tail is not "per stage": it is dominated by the S pack -> L1 -> unpack handoff (cross-TRISC CB
   sync + fp32 tile pack/unpack + matmul/reduce re-init), which ones*S^T still has and the parent's diagonal
   matmul (stat straight out of the accumulating DST) did not. One extra single-tile FPU stage after that handoff
   costs only ~0.1 µs.
3. F_COLLECT end is the slowest core's W_PUSH end; it moved 0.0-0.2 µs. The AG end differences are mostly the
   usual cross-chip variance (F_COLLECT -> AG end 3.0-3.5 µs). POST and the drain are unchanged.

## Classification
neutral (within noise). Correct and accurate, cheaper than reduce + transpose by ~0.1 µs, but it keeps the S
round trip that is the real cost of the PRE tail. Useful finding: the PRE tail lives in the DST -> L1 -> SrcA
handoff, not in how many single-tile ops follow it.

## What a child of this node should try next
1. **Go back to the parent's diagonal matmul (stat comes straight out of the accumulating DST, no S round trip,
   -0.35..-0.6 µs) and make the diagonal gather cheap or move it off BRISC's serial path.** Options:
   (a) gather with plain (non-volatile) fully-unrolled loads into registers after one fence, so the in-order core
   can overlap L1 load latency (parent: 64 serialized volatile accesses = 0.36 µs);
   (b) do the gather on the PACK TRISC right after pack_tile (it owns the tile and is idle until x*gamma), write
   the 32 words contiguously into a second CB page and push that; BRISC then sends one 128 B write;
   (c) SFPU on DST before the pack: move C[i][i] into row 0 (dst-only, no L1 round trip). Riskiest.
2. Don't spend another attempt on replacing reduce/transpose with other single FPU ops after the S pack: this
   node shows that part is worth ~0.1 µs.
3. Unchanged larger levers on this lineage: per-core/row-aware NoC0 share for the drain (r02-b02-a01 reflection #1),
   the AG section (~3 µs, forwarder outside allowed_paths), and the R_INPUT skew across cores (max R_INPUT end
   gates F_COLLECT as much as the PRE tail does).
