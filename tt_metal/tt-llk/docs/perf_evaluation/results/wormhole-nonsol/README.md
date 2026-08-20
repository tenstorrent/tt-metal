# Wormhole perf gate baseline — non speed-of-light

`wh-lb-35`, one card. `main` plus the `--perf-run-types` harness commit.
Test selection `perf and not accuracy`, compile `-n 10`, measure `-n 15`.
Speed of light off throughout.

Same two questions as Blackhole: how long a gate run takes, and how much the
numbers move on their own.

## 1. How long does a gate run take?

Cold build tree each time.

| config | run types | compile | measure | total | vs Blackhole |
|---|---|--:|--:|--:|--:|
| full | all declared | 21:57 | 8:17 | **30:14** | 1.54x |
| isolates | UNPACK_ISOLATE, MATH_ISOLATE, PACK_ISOLATE | 13:07 | 5:38 | **18:45** | 1.48x |
| l1 | L1_TO_L1 | 6:03 | 3:16 | **9:19** | 1.48x |

Wormhole costs a consistent 1.5x more than Blackhole, across every
configuration — a per-machine factor, not something specific to a run-type
selection.

The proportions are architecture-independent: isolates cost 62% of full here and
64% on Blackhole; L1_TO_L1 costs 31% and 32%. Cost tracks the number of
(module x run-type) pairs, not the silicon, so any future configuration's cost
is predictable from the pair count.

Compile is 65-73% of the total, slightly less dominant than Blackhole's 74-79%.

The two architectures run on separate CI runners in parallel, so gate latency is
the Wormhole number, not the sum: about 9 minutes for L1_TO_L1, 19 for isolates.

## 2. How much do results move on their own?

Five runs of the same code on the same card.
`move` = (largest of the 5 runs - smallest) / median.

### L1_TO_L1 — 100,971 numbers

| marker | measured | >0.5% | >1% | >2% | >5% | worst move | worst cycles |
|---|--:|--:|--:|--:|--:|--:|--:|
| TILE_LOOP | 33,657 | 186 | 64 | 27 | 0 | 4.62% | 9165 |
| KERNEL | 33,657 | 176 | 61 | 26 | 0 | 4.60% | 9165 |
| INIT | 33,657 | 8,776 | 3,882 | 742 | 2 | 5.65% | **27** |
| all | 100,971 | 9,138 | 4,007 | 795 | 2 | 5.65% | 9165 |

### Isolates — 290,952 numbers

| marker | measured | >0.5% | >1% | >2% | >5% | worst move | worst cycles |
|---|--:|--:|--:|--:|--:|--:|--:|
| TILE_LOOP | 96,984 | 1,274 | 892 | 732 | 500 | 24.27% | 41091 |
| KERNEL | 96,984 | 1,213 | 887 | 725 | 469 | 22.49% | 41091 |
| INIT | 96,984 | 15,356 | 7,988 | 1,853 | 8 | 5.91% | **26** |
| all | 290,952 | 17,843 | 9,767 | 3,310 | 977 | 24.27% | 41091 |

### Isolates, split per run type — where the problem is

| run type | marker | measured | >0.5% | >1% | >2% | >5% | worst move | worst cycles |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| MATH_ISOLATE | TILE_LOOP | 31,564 | 16 | 9 | **0** | 0 | 1.76% | 4087 |
| MATH_ISOLATE | KERNEL | 31,564 | 17 | 10 | **0** | 0 | 1.74% | 4096 |
| MATH_ISOLATE | INIT | 31,564 | 4,328 | 3,125 | 511 | 8 | 5.91% | 20 |
| UNPACK_ISOLATE | TILE_LOOP | 32,282 | 97 | 10 | **0** | 0 | 1.63% | 2109 |
| UNPACK_ISOLATE | KERNEL | 32,282 | 84 | 9 | **0** | 0 | 1.61% | 2106 |
| UNPACK_ISOLATE | INIT | 32,282 | 6,272 | 1,808 | 323 | 0 | 3.66% | 15 |
| PACK_ISOLATE | TILE_LOOP | 33,138 | 1,161 | 873 | **732** | 500 | 24.27% | 41091 |
| PACK_ISOLATE | KERNEL | 33,138 | 1,112 | 868 | **725** | 469 | 22.49% | 41091 |
| PACK_ISOLATE | INIT | 33,138 | 4,756 | 3,055 | 1,019 | 0 | 4.53% | 26 |

## The threshold

The same rule as Blackhole:

> **Flag a regression when a number is more than 2% slower AND more than 30
> cycles slower.**

| configuration | measurements | rule fires on unchanged code |
|---|--:|--:|
| L1_TO_L1 | 100,971 | 53 |
| MATH_ISOLATE | 94,692 | **0** |
| UNPACK_ISOLATE | 96,846 | **0** |
| PACK_ISOLATE | 99,414 | 1,457 |

INIT behaves exactly as designed on every configuration: 2,595 INIT points moved
more than 2%, but never by more than 27 cycles, so the cycle clause blocked all
of them. Without that clause the gate would fail on 20-cycle changes to
350-cycle numbers.

## The finding: the packer is unstable

Every failure is the packer.

**All 1,457 isolate failures are PACK_ISOLATE.** Math and unpack produced zero.
Within PACK_ISOLATE, 1,246 come from `perf_matmul`, 98 from
`perf_unpack_tilize`, 87 from `perf_math_matmul`, and 26 from six other tests.

**All 53 L1_TO_L1 failures are matmul** — 45 `perf_math_matmul`, 8 `perf_matmul`.

Full characterisation in `outliers_pack_isolate.md` and `outliers_l1_to_l1.md`.
Note that KERNEL and TILE_LOOP are paired views of the same measurement, so
1,457 rows are roughly 730 independent measurements.

### It is not a per-run state

Which of the five runs disagrees with the others, across flagged PACK_ISOLATE
points:

| run_1 | run_2 | run_3 | run_4 | run_5 |
|--:|--:|--:|--:|--:|
| 24.4% | 22.9% | 20.3% | 15.0% | 17.4% |

Independence predicts 20% each. A state established once per run and held for
its duration predicts nearly 100% on one run. This is independence: **each
measurement is affected on its own.**

### It is not a one-directional penalty

The odd run against the median of the other four:

| min | 25% | median | 75% | max | mean |
|--:|--:|--:|--:|--:|--:|
| -19.8% | -7.6% | +2.3% | +7.9% | +24.3% | **+0.6%** |

Symmetric about zero. The odd run is as often much faster as much slower, so
nothing is being *added* to some runs. The measurement lands in one of two
places.

### It is mostly two-state

Distinct values taken by the five runs:

| 2 values | 3 | 4 | 5 |
|--:|--:|--:|--:|
| 76.0% | 15.8% | 3.2% | 4.9% |

Three quarters are exactly two-valued, which is what a discrete state change
looks like. The remaining quarter is not, so this is a majority behaviour and
not a universal one.

### Output format separates the affected configurations

Within `perf_matmul`, 29,376 points, flag rate by output format:

| Float32 | Bfp8_b | Float16 | Float16_b |
|--:|--:|--:|--:|
| 0% | 0% | 3% | **13%** |

Correlated parameters point the same way: `formats.register_*` and
`formats.sfpu_math` at Tf32 = 10% against Bfp8_b = 2%, and `dest_acc = Yes` at
9% against `No` at 2%.

Association, not cause — these sweep parameters co-vary. But zero against
thirteen percent is where an investigation starts.

It also argues against cross-core contention as the explanation: contention from
neighbouring tests would not care what output format the kernel writes.

### L1_TO_L1 may be a different phenomenon

Its flagged points do not share the PACK_ISOLATE signature. Only 21% are
two-valued against 76%, and deviations span ±4.6% against ±24%. Its apparent
concentration on run_2 is 18 counts out of 53 rows, roughly 9 independent
measurements — too few to claim anything. Do not assume one cause for both.

**This is a hardware or test question, not a threshold question.** No threshold
low enough to be useful can accommodate a 24% swing.

## What this does not cover

- L1_CONGESTION not measured. Deliberately excluded; the gate is not expected to
  use it.
- Cost per individual isolate thread not measured. A gate selects the isolates as
  a set, so 18:45 is the number that matters.
- One card, one session, build tree cached after run 1. A real gate compares runs
  from different runners on different days, so this is a lower bound on the noise
  a gate will see.
- How big real regressions are. Everything here constrains the threshold from
  below. Nothing yet constrains it from above.

## Reproduce

    .claude/scripts/perf_gate_budget.sh --arch wormhole --configs full,isolates,l1 \
      --build-root $HOME/llk-wh-build --out $HOME/wh_gate_budget

    SKIP_MAIN_CHECK=1 PERF_RUN_TYPES=L1_TO_L1 OUT_DIR=$HOME/wh_noise_l1 \
      .claude/scripts/run_perf_noise_baseline.sh wormhole 5

    SKIP_MAIN_CHECK=1 PERF_RUN_TYPES=UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE \
      OUT_DIR=$HOME/wh_noise_isolates \
      .claude/scripts/run_perf_noise_baseline.sh wormhole 5

Then per configuration:

    python ~/gate_report.py <out>/noise_report.points.csv "<title>" move_<name>.md
