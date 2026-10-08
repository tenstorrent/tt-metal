# r03-b04-a01 result: 1.3131 (ok)

## What happened vs expected
All shapes pass. PCC is 0.9999985 and max_abs is 0.0204-0.0240, the same values as the root. That fits the
expectation: the same SFPU functions run on the same fp32 values, and the change only skips rows that POST never
reads. Per shape, root r02-b02-a03 -> this node (µs, chip mean):

| shape | root | this | change |
|---|---|---|---|
| h3584 | 13.95 | 12.90 | -1.06 (-7.6%) |
| h4096 | 15.12 | 14.20 | -0.92 (-6.1%) |
| h6144 | 18.67 | 17.83 | -0.84 (-4.5%) |
| h7168 | 20.38 | 19.53 | -0.85 (-4.2%) |

The geomean is 1.3131, against 1.2396 for the previous best. That is +5.9%, every shape is far outside the ±1% noise
band, and it is the new campaign best. I expected -0.6 to -0.8 µs per shape (about 1.28-1.29). The actual gain was a
bit larger.

## Why (profiler evidence)
I ran `comb.py` (r02-b02-a04's script, copied here; output in `comb_out.txt`) as
`python3 comb.py $DREAM_HOME/rmsnorm-prefill/reports/<node>`. Values are medians over the measured calls on all 4
chips, and they include the same C_COMB/C_POST zones as r02-b02-a04, so the two are directly comparable:

| TRISC_0 (unpack) idle between C_COMB end and C_POST start | r02-b02-a04 | this node |
|---|---|---|
| h3584 / h4096 / h6144 / h7168 | 1.281 / 1.273 / 1.280 / ~1.28 | 0.431 / 0.423 / 0.430 / 0.424 |

- The combine's math + pack time fell by **0.85 µs on every shape**, from 1.28 to 0.43 µs. Running mul, add and the
  fp32 rsqrt body on 4 SFPU iterations instead of 32 removed about two-thirds of the chain. The full-tile SFPU work
  (mostly rsqrt) was the cost.
- POST itself is unchanged: unpack runs 1.77-3.58 µs, the same as a04. The drain still ends 1.1-1.8 µs after pack's
  POST end. So the whole saving is the earlier POST start, and the kernel end moves by the same ~0.85 µs.
- The narrow shapes gained slightly more (-1.06 µs at h3584). The likely cause is the cross-call coupling seen in
  r02-b02-a01: a shorter call lets the next call start more evenly. It could also be AG-variance noise.
- The remaining 0.43 µs consists of ELWADD x2, the 32-bit transpose_dest, 3 small SFPU calls with their inits, the fp32
  pack, and the reduce_result CB handoff to unpack.

## Classification
win (+5.9% over the previous best, every shape -0.84 to -1.06 µs, accuracy unchanged).

## What a child of this node should try next
1. **Squeeze the remaining 0.43 µs combine gap.** Options, cheapest first:
   - Fuse the three SFPU calls into one custom 2-iteration functor (load, *1/H, +eps, rsqrt, store). That saves two
     start/done/stall rounds and one init.
   - Fold 1/H into PRE: scale the ones-row reduce scalar used by the mm_row_stat matmul by 1/H, or use the AVG scalar
     tile if it is the 1/H_full row. Watch the tf32 rounding of 1/H (a uniform scale bias that PCC ignores, but it
     costs a little max_abs).
   - Move POST's unpack-side `reconfig_data_format` + `mul_bcast_cols_init` ahead of the reduce_result wait. This is
     unsafe on BH while math runs transpose_dest (it switches ALU SrcA format), so check it first.
   - The 32-bit transpose_dest only needs row 0 -> col 0. A 16-bit transpose of the hi/lo halves is not an option. The
     alternative is to let pack write the 32 stats as a column some other way.
2. **The drain is the tail again** (1.1-1.8 µs after pack's POST end). Tune the NoC0 share per core/row
   (r02-b02-a01 reflection #1).
3. **Stick read after go** (~0.67 µs DRAM round trip, r02-b02-a04) and the AG section are the other fixed post-PRE costs.
4. Keep the C_COMB/C_POST zones. They cost nothing measurable and make `comb.py` work.
