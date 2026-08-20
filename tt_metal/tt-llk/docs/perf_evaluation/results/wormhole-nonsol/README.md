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

**All 53 L1_TO_L1 failures are matmul** — 44 `perf_math_matmul`, 9 `perf_matmul`.
L1_TO_L1 runs unpack, math and pack in sequence, so the packer instability
leaks into the end-to-end measurement.

It is not a first-run artefact. On the 53 L1_TO_L1 failures we checked which of
the five runs disagreed with the others:

    run_2  18    run_3  14    run_5  9    run_1  8    run_4  4

Spread across all five. On Blackhole the single outlier was always run 1, the
cold-build run. Here any run can be the odd one, so this is genuine run-to-run
instability in the packer path, not a property of the measurement method.

Magnitude: up to 24% and 41,091 cycles. That is far outside anything the rest of
the suite produces.

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
