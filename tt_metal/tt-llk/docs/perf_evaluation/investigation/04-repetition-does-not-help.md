# 04. Repeated runs cannot average the instability away

## Question

A gate need not compare single runs. If it ran the sweep N times per side and
compared a summary statistic, would the bistable matmul measurements stop firing?

## Method

Ten values per measurement are available from the ten-run baseline. Draw two
disjoint subsets of size k, apply a statistic to each, and apply the gate rule to
the pair. Repeat over 200 random splits per measurement.

The number that matters is not the per-measurement rate but the **expected false
failures per gate run across the whole sweep** — one firing measurement blocks
the PR, and there are 67,314 of them.

Wormhole `L1_TO_L1`, `TILE_LOOP` and `KERNEL` markers, rule `>2% AND >30 cycles`.

## Result

| statistic | measurements that can fire | expected false failures per gate run |
|---|--:|--:|
| median-of-1 | 83 | **22.6** |
| median-of-3 | 28 | 7.5 |
| median-of-5 | 16 | **5.2** |
| min-of-1 | 83 | 22.0 |
| min-of-3 | 42 | 17.2 |
| min-of-5 | 30 | **22.1** |

## Conclusion

**No.** Median-of-5 costs five times the runtime — a 45-minute gate on Wormhole
instead of 9 — and still fails every PR five times over.

## Why the median stops helping

The measurements that survive at k=5 are the strongly mixed ones: four-to-six or
five-to-five splits across ten runs. For a measurement that lands in two states
with near-equal probability, **the median is itself a coin flip**. There is no
majority to converge on, so more repeats do not help.

## Why the minimum is worse, not better

The minimum looked promising because it converges monotonically, and there is a
principled argument for it — the fast state is the execution without the stall,
which is the number a regression test wants.

It fails because it is driven by the extreme value. When one side of the
comparison happens to capture a rare fast run and the other does not, the two
minima differ by the *full* jump. For a measurement whose fast state appears once
in ten runs, roughly half of all comparisons will have it on exactly one side.
The minimum is the most sensitive statistic to this failure mode, not the least.

## Consequence

Repetition is dead as a strategy. The options that remain are to gate matmul at
around 9%, which catches only large regressions there; to leave matmul out until
the behaviour is fixed; or to fix it.
