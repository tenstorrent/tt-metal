# Blackhole perf gate baseline — non speed-of-light

One `bh_p150b` card. `main` plus the `--perf-run-types` harness commit.
Test selection `perf and not accuracy`, compile `-n 10`, measure `-n 15`.
Speed of light is off throughout. CI's perf job uses it; this gate is being
designed without it.

Two questions were asked: how long does a gate run take, and how much do the
numbers move on their own.

## 1. How long does a gate run take?

Cold build tree each time, because a gate on a fresh runner pays the compile.

| config | run types | compile | measure | total |
|---|---|--:|--:|--:|
| full | all declared | 15:33 | 4:08 | **19:41** |
| isolates | UNPACK_ISOLATE, MATH_ISOLATE, PACK_ISOLATE | 9:55 | 2:45 | **12:40** |
| l1 | L1_TO_L1 | 4:21 | 1:56 | **6:17** |

Raw seconds in `timings/`. Detail in `cost_summary.md`.

Three things follow:

- The whole suite, on one card, is under 20 minutes. A non-SoL gate is affordable.
- Compile is 74-79% of every configuration. Build caching on the gate runner
  would save more than any run-type choice.
- Cost tracks the number of (module x run-type) pairs almost linearly. Isolates
  cover 61% of the 70 pairs and cost 64% of the time; L1_TO_L1 covers 26% and
  costs 32%. The gap is fixed per-variant overhead, which run-type selection
  cannot remove — that is the case for reducing test variants.

## 2. How much do results move on their own?

The same code was run five times on the same card. Nothing changed between runs,
so every difference is measurement noise.

`move` = (largest of the 5 runs - smallest) / median.

### L1_TO_L1 — 108,377 numbers measured

| marker | numbers measured | moved >0.5% | >1% | >2% | >5% | worst move | worst move in cycles |
|---|--:|--:|--:|--:|--:|--:|--:|
| TILE_LOOP | 35,576 | 26 | 7 | **0** | 0 | 1.88% | 5110 |
| KERNEL | 35,576 | 28 | 7 | **0** | 0 | 1.87% | 5108 |
| INIT | 35,576 | 2,184 | 1,390 | 464 | 14 | 7.91% | **25** |
| UNINIT | 1,649 | 91 | 35 | 10 | 0 | 3.57% | **9** |
| all | 108,377 | 2,329 | 1,439 | 474 | 14 | 7.91% | 5110 |

### Isolates — 311,352 numbers measured

| marker | numbers measured | moved >0.5% | >1% | >2% | >5% | worst move | worst move in cycles |
|---|--:|--:|--:|--:|--:|--:|--:|
| TILE_LOOP | 102,675 | 21 | 4 | 1 | 1 | 19.75% | 353 |
| KERNEL | 102,675 | 26 | 5 | 1 | 1 | 14.12% | 349 |
| INIT | 102,675 | 4,038 | 2,364 | 861 | 1 | 5.15% | **21** |
| UNINIT | 3,327 | 222 | 222 | 222 | 2 | 14.29% | **6** |
| all | 311,352 | 4,307 | 2,595 | 1,085 | 5 | 19.75% | 353 |

Per-configuration tables in `move_l1_to_l1.md` and `move_isolates.md`. Full
reports, including the least stable points, in `noise_l1_to_l1.md` and
`noise_isolates.md`.

### Isolates, split per run type

Each run type is a separate ELF and a separate device run, so the three isolate
modes were already measured independently. Full table in
`move_isolates_by_run_type.md`.

| run type | marker | measured | >0.5% | >1% | >2% | worst move | worst cycles |
|---|---|--:|--:|--:|--:|--:|--:|
| MATH_ISOLATE | TILE_LOOP | 33,177 | 14 | 3 | **0** | 1.47% | 12 |
| MATH_ISOLATE | KERNEL | 33,177 | 7 | 0 | **0** | 0.93% | 17 |
| MATH_ISOLATE | INIT | 33,177 | 1,656 | 979 | 555 | 5.15% | 11 |
| MATH_ISOLATE | UNINIT | 1,649 | 114 | 114 | 114 | 14.29% | 6 |
| PACK_ISOLATE | TILE_LOOP | 35,573 | 2 | 1 | 1 | 19.75% | 353 |
| PACK_ISOLATE | KERNEL | 35,573 | 12 | 3 | 1 | 14.12% | 349 |
| PACK_ISOLATE | INIT | 35,573 | 1,873 | 1,216 | 287 | 4.73% | 21 |
| PACK_ISOLATE | UNINIT | 29 | 0 | 0 | **0** | 0.00% | 0 |
| UNPACK_ISOLATE | TILE_LOOP | 33,925 | 5 | 0 | **0** | 0.88% | 98 |
| UNPACK_ISOLATE | KERNEL | 33,925 | 7 | 2 | **0** | 1.21% | 98 |
| UNPACK_ISOLATE | INIT | 33,925 | 509 | 169 | 19 | 2.61% | 8 |
| UNPACK_ISOLATE | UNINIT | 1,649 | 108 | 108 | 108 | 3.85% | 1 |

Two things this shows that the merged table hides.

**PACK_ISOLATE is not a noisy thread. It has one bad configuration.** Only 2 of
its 35,573 TILE_LOOP measurements moved more than 0.5% — the fewest of the three
threads. One of those two is the outlier documented below. Remove it and pack is
the cleanest of the three.

**The cycle clause is essential for UNINIT.** For MATH and UNPACK the counts at
>0.5%, >1% and >2% are identical (114 and 108). Those points move by 6 cycles and
1 cycle. A 1-cycle move exceeding 2% means the value is under 50 cycles. Without
the absolute clause the gate would fail on single-cycle changes.

### What the tables say

TILE_LOOP is the steady-state per-tile cost — the number anyone optimizing LLK
cares about. Across both configurations, 138,251 TILE_LOOP measurements produced
exactly **one** that moved more than 2%, and that one is explained below. KERNEL
behaves the same. Most measurements did not move at all: identical code returns
identical cycle counts.

The noise concentrates in INIT and UNINIT. Those are small numbers, a few
hundred cycles, that wobble by up to 25 cycles. A fixed 20-cycle wobble is a
large percentage of a small number, which is why they look noisy and TILE_LOOP
does not.

## The threshold

> **Flag a regression when a number is more than 2% slower AND more than 30
> cycles slower.**

| configuration | numbers measured | points the rule fires on, with no code change |
|---|--:|--:|
| L1_TO_L1 | 108,377 | **0** |
| MATH_ISOLATE | 101,180 | **0** |
| UNPACK_ISOLATE | 103,424 | **0** |
| PACK_ISOLATE | 106,748 | **2** (one test config, see below) |
| isolates, all three | 311,352 | **2** |

Four of the five measured configurations produce no false positives at all.

Each clause handles one failure mode:

- The percentage catches large numbers drifting.
- The cycle count removes the small INIT and UNINIT numbers. Their worst moves
  were 25 and 9 cycles, below the 30-cycle floor, so all 1,557 of them that
  exceeded 2% are correctly ignored.

The absolute clause never binds on TILE_LOOP: 2% of a large number is thousands
of cycles, far above 30.

**The trade-off:** the rule cannot detect an INIT regression smaller than 30
cycles, about 8% of a typical INIT. That is a fair price. INIT is one-time setup
cost, not the steady-state number. For TILE_LOOP there is no compromise — 2% is
real detection.

**Repeating runs does not help.** We checked whether averaging two runs per side
gives a quieter gate. It does not improve the typical case at all; only the
extreme tail. Use one run per side and do not pay for a second.

## The one exception, documented

Two of the 311,352 isolates measurements break the rule. They are the same
measurement point seen through two markers:

`perf_pack_dest_bank`, PACK_ISOLATE, Float16_b in and out, `L1Accumulation.Yes`,
`DestAccumulation.No`, loop_factor 8, tile_cnt 8, 2 blocks of 4 tiles,
`Tilize.No`.

| marker | run 1 | run 2 | run 3 | run 4 | run 5 |
|---|--:|--:|--:|--:|--:|
| TILE_LOOP | **1434** | 1787 | 1787 | 1787 | 1787 |
| KERNEL | **2123** | 2472 | 2472 | 2472 | 2472 |

This is not random jitter. Runs 2 to 5 are bit-identical, and run 1 is faster by
exactly 353 and 349 cycles. The behaviour is bimodal: one value on the first run,
another on every run after.

In this method, run 1 is the run that compiles the ELFs cold; runs 2 to 5 reuse
them, so a first-run effect was the obvious reading.

**That reading does not survive scrutiny.** This is one measurement. With a
single point, run 1 carries a one-in-five chance of being the odd one by
coincidence, and the Wormhole data — 1,457 flagged points, characterised in
`../wormhole-nonsol/outliers_pack_isolate.md` — shows the odd run spread evenly
across all five, with no per-run concentration at all. So there is no evidence
here for a first-run cause, only for a two-valued measurement.

What this point *does* share with Wormhole is its format: Float16_b in and out.
On Wormhole, Float16_b output carries a 13% flag rate against 0% for Float32 and
Bfp8_b. One point cannot confirm that, but it is consistent with it.

Either way it is one configuration out of 311,352 measurements, and it should
not set the threshold. Raising the rule to 20% to accommodate it would make the
gate useless everywhere else.

It also flags a limitation of the method itself: **run 1 compiles cold and runs
2 to 5 do not**, so the five runs are not strictly interchangeable. No point in
either architecture shows a run-1 effect once measured properly, but the
asymmetry is real and worth remembering.

## What this does not cover

- **Blackhole only.** Wormhole needs its own five runs. Thresholds do not
  transfer between architectures.
- **L1_CONGESTION not measured.** Deliberately excluded: the gate is not expected
  to use it. It is a contention metric and would likely need a looser rule.
- **One card, one session.** A real gate compares runs from different runners on
  different days. This is a lower bound on the noise a gate will see.
- **How big real regressions are.** Everything here constrains the threshold from
  below: it must exceed the noise. Nothing yet constrains it from above: it must
  be small enough to catch a real regression. That answer is in the commit
  history, not on the card.

## Reproduce

    .claude/scripts/perf_gate_budget.sh --arch blackhole --configs full,isolates,l1 \
      --build-root $HOME/llk-build --out $HOME/perf_gate_budget

    SKIP_MAIN_CHECK=1 PERF_RUN_TYPES=L1_TO_L1 OUT_DIR=$HOME/perf_noise_l1 \
      .claude/scripts/run_perf_noise_baseline.sh blackhole 5

    SKIP_MAIN_CHECK=1 PERF_RUN_TYPES=UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE \
      OUT_DIR=$HOME/perf_noise_isolates \
      .claude/scripts/run_perf_noise_baseline.sh blackhole 5

Then, per configuration:

    python3 ~/move_table.py <out>/noise_report.points.csv "<title>" move_<name>.md
