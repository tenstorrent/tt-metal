# r04-b04-a03 result: 1.3949 (ok)

## What happened vs expected
All shapes are valid. PCC is 0.9999985 and max_abs is 0.0204-0.0240, the parent's values: same bytes in the same CB slots.
Score is 1.3949 vs the parent r04-b04-a02's 1.3997 (-0.3%), inside the ±1% band. Chip mean in µs, parent -> this node:

| shape | parent | this | change |
|---|---|---|---|
| h3584 | 11.96 | 12.07 | +0.9% |
| h4096 | 13.62 | 13.46 | -1.2% |
| h6144 | 16.91 | 17.06 | +0.9% |
| h7168 | 17.93 | 18.08 | +0.8% |

I expected F_COLLECT end -0.15..-0.5 µs on every shape and a score of ~1.41. That didn't happen. The per-shape moves are
explained by a run-dependent launch-skew state, not by the change (see below).

## Why (profiler evidence)
Scripts are in `analysis/`; run each as `python3 X.py $DREAM_HOME/rmsnorm-prefill/reports/<node>`. Outputs: `*_parent_out.txt`
for r04-b04-a02 and `*_out.txt` for this node.

**1. The deeper read depth didn't make the left cores' reads faster.** From `lr.py`, the median per-core R_INPUT
duration, left (x<9, lookahead 6) / right (lookahead 4):

| shape | parent L / R (dev 0..3) | this L / R (dev 0..3) |
|---|---|---|
| h3584 | 2.91/2.86, 3.22/3.03, 3.12/2.97, 3.09/2.94 | 2.94/2.91, 3.16/3.06, 3.09/2.93, 3.09/2.95 |
| h7168 | 5.60/5.43, 6.23/5.79, 5.90/5.42, 5.83/5.35 | 5.58/5.45, 5.93/5.97, 5.51/5.41, 5.47/5.34 |

- At h3584 nothing moved.
- At h7168 the left read is 0-0.4 µs shorter, but the right read got longer on dev 1, so the total did not move:
  R_INPUT end max is 6.56 -> 6.81 (chip timeline `tl.py`).
- With 50% more in flight, the left cores did not read 50% faster. So the left side's lag is not set by outstanding
  reads (Little's law) but by where it sits (path/arbitration), and the read is aggregate-bound. That matches r04-b03-a03,
  where bank phase was not the limit either.
- The left group is still the push gate on every shape and chip. From `lr.py`, push_e max L vs R at h7168 is
  8.19/7.87, 8.54/8.16, 8.21/7.76, 8.26/7.64.

**2. A bistable cross-call launch-skew state drives the per-shape chip-mean moves.**
"worker BRISC start max" (the latest worker kernel start after the chip's first, `tl.py`):

| shape | parent | this |
|---|---|---|
| h3584 | 0.29 | 0.31 |
| h4096 | 1.43 | **0.28** |
| h6144 | 1.46 | **0.29** |
| h7168 | 0.34 | **1.45** |

- When a shape is in the skewed state, the left-half cores start ~1-1.4 µs late. They drained last in the previous
  call, and the next call launches them last. Their stick push is then ~0.5 µs later, so the AG starts later.
- In the parent run, h4096/h6144 were skewed and h7168 was not. In this run it is the other way around. The chip-mean
  changes follow that flip exactly: h4096 -0.48 µs F_COLLECT, h6144 -0.25, h7168 +0.49.
- The reader change only affects the read *after* a core starts. It can't set the launch time, so this flip is
  run-to-run state, not this node's effect. It is also probably why sibling nodes see unexplained ±0.5 µs per-shape
  swings: r04-b03-a03 had h7168 +0.5 µs on every chip, and r04-b04-a02 had h4096 "not gated by local pushes".

## Classification
neutral (within noise). The premise was wrong. The left side's slower read is a position effect under an aggregate-bound
read, not a lack of reads in flight, so more read depth for the slow side doesn't rebalance it. The change is harmless,
but don't keep it. A child should start from r04-b01-a03 (1.4475), not from here.

## What a child of this node should try next
1. **Break the cross-call launch-skew loop. It is worth ~0.3-0.5 µs on whichever shapes are skewed in a run, and it is a
   major noise source for every comparison.**
   - The loop: the cores that end last (the left/middle-x drains, which take 5.8-6.0 µs vs 5.2 on x=13/14 at h7168)
     launch last next call, push last, and gate the AG. The AG couples everything, so they end last again.
   - Ideas:
     - (a) Release go in reverse drain-lateness order. Today go is slot order: row 2 then row 3, x ascending, spread
       0.35 µs. Give the historically slow-draining left/middle cores go first.
     - (b) Give the slow-draining cores more NoC0 drain share (r02-b02-a01 #1), so all drains end together.
     - (c) Order tiles so slow cores drain their far-bank tiles first.
   - Measure "worker BRISC start max" per shape. Success means it is ~0.3 µs on every shape in the run.
2. **Don't retry read depth or read bank order for the 20-core read** (this node, r04-b03-a03). The read is
   aggregate-bound at ~440-480 GB/s marginal with a ~0.6 µs fixed start. Per-core position sets each core's share.
3. When comparing nodes, check `tl.py`'s "worker BRISC start max" per shape before attributing a per-shape move to a
   mechanism. A 1.4 vs 0.3 µs start skew is worth ~0.5 µs of kernel time on that shape.
