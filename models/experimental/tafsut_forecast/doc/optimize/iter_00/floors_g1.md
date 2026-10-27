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
