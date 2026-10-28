# Tafsut G1 floor proposal (round 0)

Proposal only: `doc/optimize/gates_g1_proposed.json`. `gates.json` stays the provisional set until a decision adopts it
(D0 Q3, D1 ruling 5, PLAN v1.9, v1.16, v1.17, v1.20).

## Evidence
- Round-0 records: `iter_00/bench.json` (job 942, `39ed4c32618`) and `iter_00/suite.json` (job 947, `bbc4715df7c`),
  all clean stamps. `check_gates.py <bench+suite> --baseline '' --gates gates_g1_proposed.json`: **0 failures**
  (`check_gates_g1.txt`).
- Broken outputs (`tests/perf/score_broken.py`, `broken_g1.txt`): collapsed, inactive_collapsed, active1pct_0.3x,
  all_0.88x, all_1.12x and sign_flip, on all 5 bench and 6 suite cases, both precisions: **every one fails at least
  one floor**. inactive_collapsed is skipped on cases with no inactive rows (single, single_long, ragged4, horizon4),
  where it equals the golden.
- Unit checks (v1.17): `tests/perf/test_check_gates.py::test_g1_proposal_passes_round0` and
  `::test_g1_proposal_fails_broken`.

## What each floor catches
| floor | accuracy | performance | catches |
|---|---|---|---|
| pcc_normalized / pcc_final | 0.999 / 0.99 | 0.995 / 0.98 | collapse, sign flip (v1.20 absolute floors) |
| pcc_final_spread | 0.999 | 0.995 | collapse in data space |
| pcc_row_min (all rows) | 0.98 | 0.70 | inactive collapse (row PCC 0.10-0.23). Measured min: 0.992 acc; 0.830 perf (long_context, a near-flat inactive row) |
| pcc_row_min_active | 0.998 | 0.97 | one bad active row, active1pct_0.3x |
| spread_ratio_p50 | [0.98, 1.02] | [0.95, 1.05] | uniform 0.88x / 1.12x (cases with >= 8 active rows) |
| spread_ratio_p01 / p99 | [0.95, 1.05] | [0.85, 1.15] | tail narrowing (v1.9 LoFi) |
| spread_ratio_min / max | 0.92 / 1.08 | 0.92 / 1.08; paper and long_horizon 0.6 / 1.35 | 0.3x row; 0.88x / 1.12x on 1-7 active-row cases |
| inactive_max_err | 0.05 | 0.30 | gross inactive-row error |

The performance paper/long_horizon min/max overrides are wide because round 0 measured 0.868/1.237 and 0.742/1.105
there (bf8 + LoFi spread on single rows); their p01/p99/p50 floors carry the band check. abs_err_max is a drift at
1.5x the fixed record (v1.16), proposed for the target and screening shapes in both precisions once G1 fixes the record.

## single_long (case-specific floor, D1 ruling 5)
Measured (job 947): pcc_normalized 0.99782 acc, 0.99433 perf. Proposed: 0.997 acc, 0.99 perf on pcc_normalized,
pcc_final, pcc_final_spread and pcc_row_min_active.

Row error in real units (`/home/andy/bringup/scratch/tafsut/single_long_err.py`, H = 64, series scale 148.8):

| | accuracy | performance |
|---|---|---|
| max abs err (data units) | 1.175 (h 9, q0.9) | 0.939 (h 61, q0.9) |
| mean abs err | 0.050 | 0.090 |
| median max abs err | 0.121 | 0.199 |
| / recent context std (120.3) | 1.0 % | 0.8 % |
| / mean band half-width q0.1-q0.9 (1.199) | 0.98 | 0.78 |
| / mean adjacent-quantile gap (0.300) | 3.9 (mean err 0.17) | 3.1 (mean err 0.30) |
| forecast median std over H | 0.163 | |

Why: the forecast band is 0.8 % of the series scale, so a normalized-space error of about 0.008 (bf16 level on an
O(1) activation; abs_err_max 0.0078) is the size of the band. The worst element sits where the golden's own
quantiles cross (min adjacent gap -0.91), at q0.9. The mean error is a sixth of an adjacent-quantile gap. Collapsed
and sign-flip outputs still fail the case-specific floor (row PCC far below 0.99).

## D2 changes (2026-10-28)

**Fixed round-0 record.** `iter_00/bench.json` (job 1009) and `iter_00/suite.json` (job 1008), both stamped
`4d6cbdb6183`, clean tree, run after the commit. They replace jobs 942/947 (kept as `bench_job942.json`,
`suite_job947.json`): the gate metrics are bit-identical, device time moved by at most 0.08 % (paper 1697.8 -> 1697.5
ms), and the new records carry the two metrics below, which the drift checks need.

**Drift** (`gates_g1_proposed.json` `drift` / `drift_by`), against the fixed records, same case and precision, every
bench and suite case: PCCs (normalized, spread, final, final spread) -0.0005; `pcc_row_p01_active`,
`pcc_row_min_active` -0.01; `spread_ratio_p01/p99` 0.02; `spread_ratio_p50` two-sided +-0.005 accuracy / +-0.01
performance (v1.11); `flat_max_err` +0.01; `median_err` x1.25; `inactive_max_err` x1.5; **`abs_err_max` x1.5, a gross
single-element check, not a band check** (v1.16); `bias_row_p50` +0.005; `pcc_step` -0.005.

**Two new metrics** (`tests/pcc/metrics.py`). In the performance preset, elements err by up to 0.38 normalized units on
paper, so neither `abs_err_max` nor `median_err` can see a +0.05 shift. Element PCCs can't see a one-step lag either.
- `bias_row_p50`: median over rows of |mean(h - h_ref)|. Round 0 max is 0.0034 accuracy, 0.0100 performance;
  level_shift is exactly 0.05. Floors: 0.01 / 0.02.
- `pcc_step`: PCC of the step-to-step differences. Round 0 min is 0.9930 accuracy (single_long), 0.9907 performance
  (ragged4). lag1 is 0.908 at most, and -0.22 on single_long. Floors: 0.99 / 0.97.

**Floors set by D2 item 6.** Performance `pcc_row_min` 0.78; paper performance `spread_ratio` 0.80 / 1.30;
long_horizon performance 0.68 / 1.17.

**Broken outputs** (`broken_g1.txt`, CPU job 1024). Each one fails a floor or a drift check on every case, in both
precisions, including `level_shift` (+0.05 normalized) and `lag1`: 0 pass. level_shift in real units, as the median
over cases of the row mean |shift| / row std: 0.02 to 7.2, depending on the case's band width. The round-0 records
pass every floor and their own drift (`check_gates_g1.txt`, full output, jobs 1025/1026: 0 failures).

## single_long: where the error first appears (D2 item 5, job 1008)

`test_block_outputs` now checks the context rows, the REG row and the future rows of every block, the last block
(read-out cut, future rows only) and the head. On single_long (accuracy), every block's future-row PCC is >= 0.99992
(block 13 read-out 0.999994) and every context-row PCC is >= 0.99993. The PCC first drops at the **head** (final
RMSNorm + the 2-layer residual MLP), to 0.99856. Max error there is 0.011 normalized, against a band whose q0.1-q0.9
half-width is 0.8 % of the series scale. The head already runs HiFi4 with fp32 output, so the likely source is the
bf16 storage of the final-norm output, the hidden activation and the head weights. That is inferred, not isolated op
by op. No precision is changed in this gate (D2).
