# r03-b03-a02 result: 1.3120 (ok)

## What happened vs expected
All shapes are valid. PCC is 0.9999985 everywhere. max_abs is 0.0223-0.0245, against the parent's 0.0204-0.0240, so the
fp32 SFPU row sum is fine but no more accurate on this metric.

Per shape, parent r03-b03-a01 -> this node (µs, chip mean):

| shape | parent | this | change |
|---|---|---|---|
| h3584 | 12.86 | 12.90 | +0.3% |
| h4096 | 14.07 | 14.26 | +1.3% |
| h6144 | 17.86 | 17.83 | -0.2% |
| h7168 | 19.46 | 19.51 | +0.3% |

Score 1.3120 vs 1.3175 (-0.4%), inside the ±1% noise band. I expected the stat to be ready 0.3-0.5 µs earlier and
a +1.5-3% gain. **The stat was not ready any earlier.**

## Why (profiler evidence)
I ran r02-b02-a02's `tl.py` on both reports. Values are medians over the measured calls on all 4 chips, in µs from
the first worker kernel start. The full side-by-side output is in `tl_parent_vs_this.txt`. Parent -> this node:

| shape | R_INPUT end min | stat ready (W_PUSH start) min | W_PUSH end max | F_COLLECT end | AG end max |
|---|---|---|---|---|---|
| h3584 | 2.88 -> 2.83 | 3.57 -> 3.59 | 5.03 -> 4.95 | 4.91 -> 4.86 | 7.63 -> 7.87 |
| h4096 | 3.20 -> 2.83 | 3.97 -> 3.71 | 5.45 -> 5.63 | 5.33 -> 5.49 | 8.62 -> 8.66 |
| h6144 | 4.36 -> 4.39 | 5.11 -> 5.11 | 7.61 -> 7.61 | 7.47 -> 7.46 | 10.33 -> 10.32 |
| h7168 | 5.27 -> 5.25 | 6.07 -> 6.05 | 8.16 -> 8.17 | 8.03 -> 8.04 | 11.51 -> 11.49 |

- On the wide shapes, stat-ready, W_PUSH end, F_COLLECT and AG end all match the parent to within 0.03 µs. h3584
  and h4096 move by ±0.2 µs in both directions, which is AG/cross-chip variance: h3584's F_COLLECT is 0.05 µs earlier
  but its F_FABRIC ends 0.24 µs later.
- The gap from input-landed to stat-ready (W_PUSH start - R_INPUT end, at min) is ~0.7-0.8 µs on every shape, both
  before and after this change. Removing the S pack -> L1 -> unpack -> ones*S^T matmul -> pack hop did not shorten it.
- Two explanations fit, and this run can't separate them (there is no MATH-thread zone on PRE):
  1. **The hop was cheap, and the PRE tail is the last input block's math.** The ELWMUL dest-MAC at HiFi4 runs about
     125 ns/tile (r01-b02-a02). The compute waits per 4-tile block, so the last block alone is ~0.5 µs of math after
     its data lands, plus the pack and CB push. That gives a ~0.7 µs tail that doesn't depend on width, which matches
     what we see. This also fits r02-b02-a03, where swapping the reduce+transpose for one matmul saved only ~0.1 µs.
  2. **The hop's cost (~0.2-0.3 µs) was replaced by an equal cost:** transpose_dest<fp32>, which is the hi/lo 16-bit
     MOV sequence plus cfg RMWs, plus the 4-iteration SFPU column reduce and its init.
  Either way, r02-b02-a02's "stat 0.35-0.6 µs earlier" is not reproduced by removing the S round trip on its own. That
  number was measured against reduce + transpose, i.e. two hops, and in different runs.
- POST, the combine and the drain are unchanged. TRISC end - AG end and drain end - AG end are the same as the parent.

## Classification
Neutral (within noise). The mechanism is correct: the row-0 stat comes out of DST with an SFPU column sum and fp32
accuracy. It is not a speedup, because the S round trip was not what gated the stick push. That makes this a flawed
premise rather than an execution bug. Don't retry "keep the stat in DST" variants of PRE.

## What a child of this node should try next
1. **Measure the PRE tail before cutting it again.** Add a MATH-thread (TRISC_1) zone around the last input block's
   mul_tiles plus the stat finish, and a TRISC_0 zone around the last block's unpack. If the last block's math is the
   ~0.5 µs, then:
   - **Lower the fidelity of the PRE x*x only.** Call `mul_init` / `mul_tiles` with an explicit MathFidelity::HiFi3
     template, and leave POST at HiFi4. With bf16 inputs, HiFi3 drops only the srcA-low x srcB-low partial product
     (~2^-12 relative), so it is close to exact. HiFi2 halves the math but truncates srcB's last mantissa bit, a ~2^-8
     downward scale bias on sum(x^2). PCC ignores that, but max_abs (0.024 now, gate 0.05) would rise.
   - **Or shrink the last wait unit.** Wait per tile (not per 4-tile block) for the last block, so the math overlaps
     the last block's arrival.
2. Otherwise, go back to the post-AG levers: the remaining 0.38 µs combine gap (sibling branches), the drain tail
   1.0-1.9 µs after POST (r02-b02-a01 per-core NoC0 share), and the stick DRAM read after go (~0.67 µs).
3. This node's code is a valid drop-in. A child can keep it (fp32 stat, one less CB hop, unpack thread freed
   earlier for x*gamma) or revert it to the parent's matmul path. The measured difference is nil.
