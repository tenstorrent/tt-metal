# Blackhole perf gate baseline — non speed-of-light

Measured on one `bh_p150b` card, `main` + the `--perf-run-types` harness commit,
markers `perf and not accuracy`, compile `-n 10`, measure `-n 15`.
Speed of light is off throughout: CI's perf job uses it, this gate is being
designed without it.

## 1. Cost — how long a gate run takes

Cold build tree per configuration, because a gate on a fresh runner pays the compile.

| config | run types | compile | measure | total | vs full |
|---|---|--:|--:|--:|--:|
| full | all declared | 15:33 | 4:08 | 19:41 | 100% |
| isolates | UNPACK_ISOLATE, MATH_ISOLATE, PACK_ISOLATE | 9:55 | 2:45 | 12:40 | 64% |
| l1 | L1_TO_L1 | 4:21 | 1:56 | 6:17 | 32% |

Raw seconds in `timings/`. Detail in `cost_summary.md`.

Findings:

- Cost tracks (module x run-type) pairs almost linearly. Isolates cover 61% of
  the 70 pairs and cost 64% of the time; L1_TO_L1 covers 26% and costs 32%. The
  gap is fixed per-variant overhead that run-type selection cannot remove — that
  is the argument for reducing test variants.
- **Compile is 74-79% of every configuration.** A warm build cache on the gate
  runner would save more than any run-type choice.
- The whole suite, unsharded, on one card, is under 20 minutes. A non-SoL gate
  is affordable.

## 2. Noise — how large a difference must be to mean anything

Five runs of the SAME commit on the SAME card. Every difference is noise.
`noise_l1_to_l1.md` is the full report; 108,377 points, 2.17M simulated
comparisons.

Median-of-1, which is what a "latest main vs this run" gate does:

| marker | p95 | p99 | max |
|---|--:|--:|--:|
| TILE_LOOP | 0.00% | 0.01% | 1.91% |
| KERNEL | 0.00% | 0.01% | 1.90% |
| UNINIT | 0.00% | 0.93% | 3.57% |
| INIT | 0.00% | 1.36% | 7.91% |

Findings:

- **TILE_LOOP is essentially deterministic.** p99 of 0.01%. Half of all
  comparisons are exactly zero — identical code returns identical cycle counts.
- **All the noise is in INIT and UNINIT.** Every one of the 25 least stable
  points is INIT, median 300-450 cycles, absolute spread 14-25 cycles. It is a
  small fixed jitter that looks large only because the denominator is small.
- **A single global relative threshold is the wrong design.** One threshold
  clearing every sample must be 8%, set entirely by INIT. TILE_LOOP alone could
  use 2% — four times tighter, on the marker that carries the detection value.
- **Repeating runs does not help.** median-of-2 has a slightly WORSE p99 (0.51%
  vs 0.41%); only the extreme tail improves. Do not pay for a second run.

Recommended:

| scope | threshold | basis |
|---|---|---|
| TILE_LOOP, KERNEL | 2% | clears all 1.4M observed comparisons |
| INIT, UNINIT | absolute cycles, not % | observed jitter is 14-25 cycles |

## 3. Caveats

- **Lower bound.** All five runs used one card, one session, and a build tree
  cached after run 1. A real gate compares runs from different runners on
  different days. Machine and day drift are not in these numbers.
- Blackhole only. Wormhole needs its own baseline; thresholds do not transfer.
- L1_TO_L1 only. The isolates noise baseline is pending. L1_CONGESTION is
  untested and, being a contention measurement, is likely the noisiest.
- `cost_summary.md` omits the yield columns: the budget script counts rows in
  the wrong directory (`tests/python_tests/perf_data` instead of
  `tt-llk/perf_data`). Timings are unaffected.

## 4. Reproduce

    .claude/scripts/perf_gate_budget.sh --arch blackhole --configs full,isolates,l1 \
      --build-root $HOME/llk-build --out $HOME/perf_gate_budget

    SKIP_MAIN_CHECK=1 PERF_RUN_TYPES=L1_TO_L1 OUT_DIR=$HOME/perf_noise_baseline \
      .claude/scripts/run_perf_noise_baseline.sh blackhole 5
