# r02-b02-a02 result: 1.2292 (ok)

## What happened vs expected
The run is valid. PCC is 0.9999985 on every shape. max_abs is 0.0239-0.0251, against the parent's 0.0217-0.0243; it is a
different (fp32-DST matmul) summation order and stays well under the 0.05 gate. The x·x^T diagonal gives the right
per-row sums of squares, so the matmul's B-transpose orientation is right.

Per shape (parent r02-b02-a01 in brackets): h3584 13.88 µs (13.87), h4096 15.09 (15.20), h6144 19.09 (18.73),
h7168 20.77 (20.42). Geomean is 1.2292 vs 1.2385 (-0.75%), which is inside the ±1% noise band. I expected
-0.3 to -0.6 µs per shape. The result is neutral: the half of the mechanism that was planned worked, and the new BRISC
gather cost almost the same amount of time.

## Why (profiler evidence)
I ran `tl.py` (in this node dir; `python3 tl.py $DREAM_HOME/rmsnorm-prefill/reports/<node>`) on `reports/<node>`. Values are medians over the measured calls on all
4 chips, in µs from the first worker kernel start, min-max over the 20 workers. Parent -> this node:

| | h3584 | h4096 | h6144 | h7168 |
|---|---|---|---|---|
| R_INPUT end (max) | 3.37 -> 3.36 | 3.96 -> 3.87 | 5.61 -> 5.67 | 6.46 -> 6.45 |
| stat ready = W_PUSH start (min) | 3.75 -> 3.17 | 3.79 -> 3.20 | 5.20 -> 4.85 | 6.18 -> 5.84 |
| W_PUSH duration (min) | 0.47 -> 0.83 | 0.48 -> 0.85 | 0.48 -> 0.84 | 0.48 -> 0.84 |
| W_PUSH end (max) | 5.06 -> 4.85 | 5.80 -> 5.58 | 7.70 -> 7.50 | 8.18 -> 8.13 |
| F_COLLECT end | 4.94 -> 4.77 | 5.66 -> 5.46 | 7.55 -> 7.36 | 8.05 -> 8.02 |
| AG wait end (max) | 8.30 -> 7.63 | 8.58 -> 8.45 | 10.38 -> 10.62 | 11.56 -> 12.04 |
| TRISC end - AG end | ~4.1 -> ~4.1 | ~4.3 -> ~4.4 | ~5.5 -> ~5.7 | ~6.1 -> ~6.0 |

1. **Removing the reduce + transpose works.** The stat is in BRISC's CB 0.35-0.6 µs earlier relative to the input,
   on every shape. The matmul x·x^T keeps up with the input stream, and R_INPUT is unchanged. So the per-tile
   32x32x32 HiFi4 matmul is not slower than the eltwise square in any way that matters here. This confirms
   r01-b04-a04's diagnosis that the ~1 µs PRE tail is the fixed reduce/transpose/handoff chain.
2. **The BRISC diagonal gather gave the saving back.** W_PUSH, measured from the moment the stat is available (it is
   entered from the gamma loop's poll, so no CB wait is included), went from 0.48 to 0.84 µs. The +0.36 µs (~490
   cycles) is the 32 strided volatile L1 loads plus 32 stores plus two fences, done while BRISC's ~2x56 gamma reads are
   still in flight. So W_PUSH end and F_COLLECT end only moved 0.03-0.2 µs earlier.
3. The rest is AG / cross-chip variance. F_COLLECT -> AG end was 2.5-3.65 µs here and 2.5-3.15 µs in the parent, with
   no code change on that path. That is why the narrow shapes broke even and h6144/h7168 lost ~0.35 µs. POST and the
   drain are unchanged.

## Classification
neutral (within noise). The diagnosis is right and the compute half works (stat ready ~0.5 µs earlier). This is a
repairable execution problem. The bug: moving the "transpose" onto BRISC as a strided L1 gather costs ~0.36 µs on the
same critical path.

## What a child of this node should try next
1. **Get a row-0 stat out of the FPU with ONE op, so no gather is needed.** Keep the parent's DST-accumulated
   S = sum_k x_k∘x_k (eltwise, packed once). Then replace reduce + transpose with a single
   `matmul_tiles(ones_cb, pre_intermediate_cb, 0, 0, 0)` with `matmul_init(ones, S, transpose=1)`. That computes
   C = 1·S^T, so C[r][i] = sum_j S[i][j], and row 0 already holds the 32 row sums in the layout the old transposed
   stick used. The writer's original two 64 B face-row writes then work unchanged, so drop stick_from_diag. This
   needs a tile of ones (all 32x32, or at least row 0 of faces 0/1) in a CB. The writer could fill it next to the
   reduce scalars, or a block of reduce_scalar_sum_cb could be reused if it is all-ones. S goes to tf32 in srcA just
   as it does for the current reduce, so accuracy is the same as the parent. Expected: the ~0.5 µs saving here
   without the 0.36 µs BRISC cost.
2. If the gather is kept instead, make it cheaper: plain (non-volatile) loads after a single fence, fully unrolled,
   or move the gamma issue loop after the stick push so the gather doesn't compete with in-flight gamma reads (the NoC
   can't gather strided words itself). W_GAMMA end also slipped +0.4-0.9 µs because the push runs inside its loop.
3. Unchanged levers on this lineage: per-core NoC0 share for the drain (parent's #1), and the AG fabric section
   (~2.5-3.6 µs, the largest fixed cost, but the forwarder kernel is outside allowed_paths).
