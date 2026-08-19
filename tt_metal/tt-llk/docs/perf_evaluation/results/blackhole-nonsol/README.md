# Blackhole perf gate baseline — non speed-of-light

One `bh_p150b` card. `main` plus the `--perf-run-types` harness commit.
Test selection `perf and not accuracy`, compile `-n 10`, measure `-n 15`.
Speed of light is off throughout. CI's perf job uses it; this gate is being
designed without it.

Two questions were asked.

## 1. How long does a gate run take?

Cold build tree each time, because a gate on a fresh runner pays the compile.

| config | run types | compile | measure | total |
|---|---|--:|--:|--:|
| full | all declared | 15:33 | 4:08 | **19:41** |
| isolates | UNPACK_ISOLATE, MATH_ISOLATE, PACK_ISOLATE | 9:55 | 2:45 | **12:40** |
| l1 | L1_TO_L1 | 4:21 | 1:56 | **6:17** |

Raw seconds in `timings/`.

Three things follow:

- The whole suite, on one card, is under 20 minutes. A non-SoL gate is affordable.
- Compile is 74-79% of every configuration. Build caching on the gate runner
  would save more than any run-type choice.
- Cost tracks the number of (module x run-type) pairs almost linearly. Isolates
  cover 61% of the 70 pairs and cost 64% of the time. L1_TO_L1 covers 26% and
  costs 32%. The gap is fixed per-variant overhead, which run-type selection
  cannot remove — that is the case for reducing test variants.

## 2. How much do results move on their own?

We ran the same code five times on the same card. Nothing changed between runs,
so every difference is measurement noise. 108,377 numbers were measured.

`move` = (largest of the 5 runs - smallest) / median.

| marker | numbers measured | moved >0.5% | >1% | >2% | >5% | worst move | worst move in cycles |
|---|--:|--:|--:|--:|--:|--:|--:|
| TILE_LOOP | 35,576 | 26 | 7 | **0** | 0 | 1.88% | 5110 |
| KERNEL | 35,576 | 28 | 7 | **0** | 0 | 1.87% | 5108 |
| INIT | 35,576 | 2,184 | 1,390 | 464 | 14 | 7.91% | **25** |
| UNINIT | 1,649 | 91 | 35 | 10 | 0 | 3.57% | **9** |
| all | 108,377 | 2,329 | 1,439 | 474 | 14 | 7.91% | 5110 |

Full detail, including the per-point data, is in `noise_l1_to_l1.md` and
`how_much_results_move.md`.

### What this says

TILE_LOOP is the steady-state per-tile cost — the number anyone optimizing LLK
cares about. **Not one of its 35,576 measurements moved by more than 2%.** KERNEL
behaves the same. Most measurements did not move at all: identical code returns
identical cycle counts.

All the noise is in INIT and UNINIT. Those are small numbers, around 350 cycles,
and they wobble by up to 25 cycles. That is a large percentage of a small number,
which is why they look noisy and TILE_LOOP does not.

## The threshold

> **Flag a regression when a number is more than 2% slower AND more than 30
> cycles slower.**

On five runs of unchanged code, this rule fires **zero times** out of 108,377
measurements.

Each clause handles one failure mode:

- The percentage catches large numbers drifting. On TILE_LOOP nothing reached 2%.
- The cycle count removes the small INIT numbers. Their worst move was 25 cycles,
  below the 30-cycle floor, so none of the 464 that exceeded 2% survive the rule.

The absolute clause never binds on TILE_LOOP: 2% of a large number is thousands
of cycles, far above 30.

**The trade-off:** the rule cannot detect an INIT regression smaller than 30
cycles, about 8% of a typical INIT. That is a fair price. INIT is one-time setup
cost, not the steady-state number. For TILE_LOOP there is no compromise — 2% is
real detection.

**Repeating runs does not help.** We checked whether averaging two runs per side
gives a quieter gate. It does not improve the typical case at all; only the
extreme tail. Use one run per side and do not pay for a second.

## What this does not cover

- **Blackhole only.** Wormhole needs its own five runs. Thresholds do not
  transfer between architectures.
- **L1_TO_L1 only.** The isolates baseline is pending. L1_CONGESTION has not been
  measured and, being a contention metric, is likely the noisiest.
- **One card, one session.** The build tree was cached after run 1. A real gate
  compares runs from different runners on different days, so this is a lower
  bound on the noise a gate will see.
- **How big real regressions are.** Everything here constrains the threshold from
  below: it must exceed the noise. Nothing yet constrains it from above: it must
  be small enough to catch a real regression. That answer is in the commit
  history, not on the card.

## Reproduce

    .claude/scripts/perf_gate_budget.sh --arch blackhole --configs full,isolates,l1 \
      --build-root $HOME/llk-build --out $HOME/perf_gate_budget

    SKIP_MAIN_CHECK=1 PERF_RUN_TYPES=L1_TO_L1 OUT_DIR=$HOME/perf_noise_baseline \
      .claude/scripts/run_perf_noise_baseline.sh blackhole 5
