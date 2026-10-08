# r05-b02-a01 result: 1.4078 (ok)

## What happened vs expected
Valid on every shape on the first device run (no compile iteration needed). PCC 0.9999985; max_abs 0.0239-0.0251
(parent 0.0204-0.0240). That matches r02-b02-a02's x·xᵀ matmul (0.0239-0.0251): the summation order differs, the gate
is 0.05. So the matmul orientation, the identity mask (writer-built in the trans_mat CB, copied to DST[1]) and the
SFPU column sum into row 0 are all correct.

µs, chip mean, parent r04-b04-a02 → this node:

| shape | parent | this | change |
|---|---|---|---|
| h3584 | 11.96 | 11.91 | -0.4% |
| h4096 | 13.62 | 13.38 | -1.8% |
| h6144 | 16.91 | 16.99 | +0.5% |
| h7168 | 17.93 | 17.81 | -0.7% |

Score 1.4078 vs 1.3997 (+0.6%), inside the ±1% noise band, so this is **not a measured win**. I expected +1-2.5%
from a 0.15-0.35 µs earlier stat at the slowest core. The stat did come earlier, but by less than that.

## Why (profiler evidence)
`analysis/pre.py` (r04-b01-a03's script) gives `pre_out.txt` for this node and `pre_parent_out.txt` for r04-b04-a02.
`analysis/tl.py` (r04-b04-a03's script) gives `tl_out.txt` and `tl_parent_out.txt`. Values are medians over the
measured calls × 4 chips, µs from the call's first worker start. Each cell is parent → this:

| shape | R_INPUT end max | stat ready − input end, med | … max | push end max | F_COLLECT end | drain end max |
|---|---|---|---|---|---|---|
| h3584 | 3.39 → 3.35 | 0.67 → 0.66 | 0.85 → 0.82 | 4.42 → 4.35 | 4.52 → 4.46 | 11.56 → 11.49 |
| h4096 | 3.99 → 3.95 | 0.79 → 0.76 | 1.07 → **0.88** | 5.27 → 5.08 | 5.35 → 5.12 | 13.25 → 13.08 |
| h6144 | 5.73 → 5.59 | 0.83 → 0.78 | 1.15 → **1.03** | 7.10 → 6.87 | 7.21 → 6.97 | 16.60 → 16.75 |
| h7168 | 6.33 → 6.38 | 0.82 → 0.78 | 1.13 → **0.95** | 7.75 → 7.56 | 7.84 → 7.65 | 17.58 → 17.31 |

- **The PRE tail barely moved at the median: -0.01..-0.05 µs.** The slowest core's tail improved by 0.03-0.19 µs, and
  that core gates F_COLLECT. Push end max and F_COLLECT end moved 0.06-0.24 µs earlier on every shape.
- So, at HiFi4, `matmul x·xᵀ (64 MVMULs/tile) + SFPU mask (full-tile fp32 multiply) + SFPU column sum` costs about the
  same as `ELWMUL x·x (32 ELWMULs/tile) + S pack → unpack → ones·Sᵀ matmul → pack`. The ~0.4 µs advantage I derived
  from r02-b02-a02 vs r02-b02-a03 did not reproduce. Either it was cross-run AG/launch variance, or the in-DST
  extraction (~0.2-0.3 µs) costs about what the S round trip did. This node can't separate those without a TRISC_1
  zone. Either way, the median tail is still ~0.66-0.78 µs, far from HiFi2's 0.45-0.50. The per-tile HiFi4 math is
  not cheaper on the matmul engine: 64 MVMULs ≈ 32 ELWMULs per tile.
- **The chip means are dominated by the bistable launch-skew state** (r04-b04-a03). "Worker BRISC start max" was:
  - parent: 0.29 / 1.43 / 1.46 / 0.34 (h4096 and h6144 skewed);
  - this run: 0.28 / 1.25 / 0.29 / 1.46 (h4096 and h7168 skewed).

  h7168 ran in the skewed state (worth ~+0.3-0.5 µs) and was still 0.12 µs faster. Its drain end max is 0.27 µs earlier.
  h6144 ran un-skewed but F_FABRIC end was +0.16 µs later (cross-chip). Even so, F_COLLECT was 0.24 µs earlier, so
  its +0.5% is AG-side noise, not the change. Net: a small real AG-start gain (~0.1-0.2 µs) that the per-shape noise
  hides.

## Classification
neutral (within noise, +0.6%). The mechanism is correct and slightly helps the slowest core (F_COLLECT -0.06..-0.24 µs
on every shape), but the premise was wrong: the HiFi4 matmul is not faster per tile than the HiFi4 ELWMUL, and the
in-DST diagonal extraction costs about what the S round trip did. It is a valid drop-in. Pros: an fp32 stat with no
tf32 truncation of S, and a ~0.1-0.2 µs earlier max-core push. Con: max_abs +0.002. A child may keep or revert it.

## What a child of this node should try next
1. **Don't spend more attempts on reformulating the PRE stat at HiFi4.** All variants are now within noise:
   - reduce + transpose (r01/r02);
   - ones·Sᵀ (r02-b02-a03);
   - transpose_dest + SFPU column sum (r03-b03-a02);
   - x·xᵀ + mask + column sum (this node).

   The tail is the per-tile HiFi4 math lag plus ~0.45 µs of fixed handoff. Without lower fidelity, only a smaller last
   wait unit can help: wait per tile for the last block (≤0.1 µs).
2. **The launch-skew loop is the biggest noise source and probably the cheapest real gain left.** "Worker BRISC start
   max" flips between ~0.3 and ~1.4 µs per shape from run to run, and the skewed state costs ~0.3-0.5 µs. Attack the
   cross-call loop: make the late-draining left/middle cores finish their drain no later than the others (per-core
   NoC0 share, or release go to them first). Success means "worker BRISC start max" ≤0.35 µs on every shape.
   Always report `tl.py`'s start-max next to chip means.
3. If this node's PRE is kept: DST[1] holds the identity, and its SFPU mask is a full 32-iteration multiply. A custom
   SFPU pass that masks and column-sums only faces 0 and 3 would cut ~0.1 µs. That is low value given #1.
