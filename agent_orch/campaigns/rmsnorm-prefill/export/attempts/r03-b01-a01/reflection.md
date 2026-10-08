# r03-b01-a01 result: 1.3129 (ok)

## What happened vs expected
Valid on every shape. PCC 0.9999985 and max_abs 0.0204-0.0240 are the parent's values (same SFPU functions, same fp32
DST, only the lanes that run changed). Per shape vs the parent r02-b02-a03 (µs, chip mean):

| shape | parent | this node | change |
|---|---|---|---|
| h3584 | 13.95 | 12.97 | -0.98 (-7.0%) |
| h4096 | 15.12 | 14.24 | -0.88 (-5.8%) |
| h6144 | 18.67 | 17.73 | -0.94 (-5.0%) |
| h7168 | 20.38 | 19.48 | -0.90 (-4.4%) |

Geomean 1.3129 vs 1.2396 (+5.9%), every shape far outside the ±1% noise band. New campaign best. It landed at the top
of the predicted -0.5..-0.9 µs per shape.

## Why (profiler evidence)
`r02-b02-a02/tl.py` on both reports (medians over measured calls, all 4 chips, µs from first worker kernel start;
this node's output is in `tl_out.txt`). Post-AG compute = TRISC end - AG wait end, min over workers:

| shape | AG end parent -> this | post-AG compute parent -> this | drain end max parent -> this |
|---|---|---|---|
| h3584 | 7.47 -> 7.59 | 4.10 -> 3.25 | 13.28 -> 12.56 |
| h4096 | 8.12 -> 8.12 | 4.31 -> 3.46 | 14.65 -> 13.85 |
| h6144 | 10.12 -> 9.85 | 5.36 -> 4.54 | 18.30 -> 17.18 |
| h7168 | 11.24 -> 11.13 | 5.91 -> 5.06 | 19.63 -> 18.76 |

- Everything before the AG end is unchanged (W_PUSH end, AG end, drain start - AG = 0.66-0.72 µs are the same within
  call-to-call variance). The whole gain is **-0.82 to -0.85 µs of post-AG compute on every shape**, a fixed cost, as
  expected for the stat combine. The drain end moved by the same amount, so the drain was following POST and the
  saving went straight to the kernel end.
- So of r02-b02-a04's 1.28 µs "unpack idle before C_POST", ~0.85 µs was the full-tile SFPU mul/add/rsqrt (32 iterations
  each; rsqrt is the expensive one). The remaining ~0.4 µs is the 2 ELWADDs, transpose_dest<fp32>, the 4 KB fp32 pack
  to reduce_result_cb, the CB handoff and the POST re-init.
- The SFPU row-mapping assumption holds: VectorMode::R with ITERATIONS=2 covers all 32 columns of row 0 (an SFPU
  iteration = 4 dest rows x even/odd columns). If it didn't, half the rows would have had un-rsqrt'ed stats and
  accuracy would have collapsed.
- Drain tail after TRISC end is unchanged (~1.0-2.0 µs at max), as is AG section length.

## Classification
win (+5.9% geomean over the parent, all shapes > noise). Fixed-cost removal on the post-AG critical path.

## What a child of this node should try next
1. **Shrink the rest of the combine (~0.4 µs).** Same trick on the remaining full-tile work:
   - Skip transpose_dest: produce col 0 with matmuls instead. Accumulate `matmul_tiles(I_scaled, G_d, transpose=1)` for
     the 4 gathered tiles (C = A*G^T puts row 0 of G into col 0; A = identity, or (1/H)*identity to fold the
     mul_unary too), then add eps + rsqrt with VectorMode::C ITERATIONS=... on col 0 (faces 0/2; col 0 sits in the even
     columns of every iteration, so all 8 iterations are needed: use the R-before-transpose path or accept 16 iterations).
     Needs an identity tile in a CB (writer can fill it next to the reduce scalars).
   - Or pack only faces 0 and 2 of reduce_result (the POST bcast_cols only reads col 0): a partial-face pack halves
     the 4 KB fp32 pack + the POST unpack of the stat.
   - Pre-issue POST's `reconfig_data_format` / `mul_bcast_cols_init` before waiting on reduce_result (unpack side).
   Add TRISC zones (C_COMB/C_POST, see r02-b02-a04's compute diff) to measure what remains before cutting it.
2. **Same SFPU restriction anywhere else full tiles are processed for a 32-value result** — e.g. in PRE, only row 0 of
   the stat matters; check for any other full-tile SFPU/pack on the stick path (pack of the stat tile is 4 KB fp32
   where 128 B matter: a partial pack would shave the PRE tail too).
3. Unchanged larger levers on this lineage: per-core/row-aware NoC0 share for the drain (r02-b02-a01 #1; drain tail
   1-2 µs after TRISC end), the AG section (~3 µs; the forwarder fork from r02-b02-a04 is reusable), and the column split.
