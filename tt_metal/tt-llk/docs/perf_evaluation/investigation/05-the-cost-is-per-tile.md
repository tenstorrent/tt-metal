# 05. The slow state costs a few cycles per tile

This is the finding that reframes the whole investigation.

## Question

Is the extra time a **one-off event** — a stall at kernel start, a hiccup
somewhere in the loop — or a **rate difference** sustained across every
iteration?

Nobody had checked, and the two answers lead to completely different places.

## Method

Divide each flagged measurement's jump by the work it did:

```
per_loop      = jump / loop_factor
per_tile_loop = jump / (loop_factor x tile_cnt)
```

The two matmul tests make this a genuine test rather than a curve fit, because
they differ enormously in scale: `perf_matmul` uses a loop factor of 64 and its
measurements are 10k–28k cycles; `perf_math_matmul` uses 1024 and its
measurements are 118k–218k. If the jump is a fixed event, these normalisations
will disagree wildly between the two tests. If it is a rate, they will agree.

## Result

| test | loop_factor | measurements | median `per_tile_loop` | range |
|---|--:|--:|--:|---|
| `perf_math_matmul` | 1024 | 30 | **2.20** | 0.77 – 3.85 |
| `perf_matmul` | 64 | 12 | **1.71** | 1.01 – 4.34 |

**They agree.** A 16-fold difference in loop factor and a 20-fold difference in
total cycles, and both land on one to four extra cycles per tile.

For comparison, `per_loop` — dividing by iterations but not by tiles — is
noticeably less consistent: medians of 5.72 and 6.55 with ranges of 2.6–13.6 and
3.8–17.4. Normalising by tiles as well as iterations is what collapses the two
tests together, which says the **tile** is the unit the cost attaches to.

## Conclusion

> **The slow state is not an event. It is a slightly slower steady rhythm —
> roughly two extra cycles for every tile processed — sustained for the whole
> loop.**

A 13,965-cycle jump is not a 13,965-cycle stall. It is about 2.7 cycles x 5 tiles
x 1024 iterations.

## Why this explains everything else

| earlier observation | explained by a rate difference |
|---|---|
| No thread's zone stands out (03) | All three threads run at the slower rhythm together |
| The counter build loses the effect (02) | A different instruction stream lands the pipeline in the other rhythm |
| The outcome is discrete, not a spread (§8.4) | Two rhythms, not a distribution |
| The alternate value is shared across configurations (§8.4) | The *rate* difference is a property of the pipeline, so the absolute jump just scales with the work |
| Repetition does not average it out (04) | There is nothing transient to average; the whole run is in one rhythm or the other |

It is also the shape an extra stall cycle in the inner loop produces: one
arbitration resolving differently, repeated once per tile.

## What it unlocks

RTL simulation was previously impossible — a 200,000-cycle kernel at 10–1000
simulated cycles per second is hours to days, and RTL is deterministic so it
would not reproduce a probabilistic effect by chance.

**If the cost is per tile, the reproducer shrinks.** At `loop_factor=16` and
`tile_cnt=1`, the expected jump is about 32 cycles on a roughly 1,900-cycle
kernel — still over 1.5%, still detectable, and small enough for RTL to run in
about a minute. At that size you are not hoping to catch a race; you are reading
the handshake cycle by cycle.

## The experiment that tests it

`.claude/scripts/perf_loop_factor_sweep.sh` runs `perf_math_matmul` at loop
factors 1024, 256, 64 and 16 with `run_count=20`, and reports how many
measurements still flip and the per-tile cost at each.

- **The jump scales linearly and still flips at 16** — the per-tile reading is
  confirmed and you have a minimal reproducer.
- **It stops flipping below some loop factor** — the two rhythms need time to
  establish, which is a strong clue in itself, and that threshold is the smallest
  case that reproduces.
