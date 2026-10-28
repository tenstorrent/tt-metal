# Tafsut G1 floor proposal (round 0)

Proposal only: `doc/optimize/gates_g1_proposed.json`. `gates.json` stays the provisional set until a decision adopts it
(D0 Q3, D1 ruling 5, PLAN v1.9, v1.16, v1.17, v1.20).

## Evidence (refreshed after D3)
- Fixed round-0 records: `iter_00/bench.json` (job 1061) and `iter_00/suite.json` (job 1062), stamped `a1ee197c94f`,
  clean tree, started after the commit (18:47:05Z and 18:50:45Z vs commit 18:46:57Z). Every gate metric of jobs
  1009/1008 is bit-identical (kept as `bench_job1009.json`, `suite_job1008.json`); the new records add
  `bias_row_max`, `bias_row_p95`, horizon4's teacher-forced forward 2 (`f2_*`) and `started_at`. Paper 1697.51 ms.
- `check_gates.py bench.json suite.json --gates gates_g1_proposed.json` (CPU job 1085 at `51f15e2ac53`, stamp in the
  first line): **0 failures**, full output in `check_gates_g1.txt`.
- Broken outputs (`score_broken.py`, CPU job 1084 at `51f15e2ac53`, `broken_g1.txt`): 226 scorings (11 cases, both
  forwards where there are two, both precisions, 9 broken kinds incl. `shift10pct`). **One passes:**
  horizon4 / forward 1 / performance / `shift10pct` (see "Row-bias tail").
- Unit checks: `tests/perf/test_check_gates.py` (stamp, started-before-commit, f2 family, coverage, bias_row_max).

## What each floor catches
| floor | accuracy | performance | catches |
|---|---|---|---|
| pcc_normalized / pcc_final | 0.999 / 0.99 | 0.995 / 0.98 | collapse, sign flip (v1.20 absolute floors) |
| pcc_final_spread | 0.999 | 0.995 | collapse in data space |
| pcc_row_min (all rows) | 0.98 | 0.78 (D2) | inactive collapse (row PCC 0.10-0.23) |
| pcc_row_min_active | 0.998 | 0.97 | one bad active row, active1pct_0.3x |
| spread_ratio_p50 | [0.98, 1.02] | [0.95, 1.05] | uniform 0.88x / 1.12x (cases with >= 8 active rows) |
| spread_ratio_p01 / p99 | [0.95, 1.05] | [0.85, 1.15] | tail narrowing (v1.9 LoFi) |
| spread_ratio_min / max | 0.92 / 1.08 | 0.92 / 1.08; paper 0.80 / 1.30, long_horizon 0.68 / 1.17 (D2) | 0.3x row; 0.88x / 1.12x on 1-7 active-row cases |
| inactive_max_err | 0.05 | 0.30 | gross inactive-row error |
| bias_row_p50 | 0.01 | 0.02 | level_shift |
| bias_row_max | 0.03; long_horizon 0.12 | 0.04; paper 0.12, long_horizon 0.35, horizon4 0.08 | shift on a few rows |
| bias_row_p95 (>= 100 rows) | 0.025 | 0.04; long_horizon 0.07 | shift on >= 5 % of rows |
| pcc_step | 0.99; single_long 0.98 (D3) | 0.97 | lag1 |
| f2_* | every forward-1 floor except pcc_final/pcc_final_spread; long_horizon case floors below | | the same, on forward 2 |

Drift (all against the fixed records, same case and precision; f2_* has the same tolerances): see "D2 changes"; new:
`bias_row_max` x1.5, `bias_row_p95` +0.005. `abs_err_max` is a drift at 1.5x the fixed record in every case and
precision (v1.16), a gross single-element check, not a band check.

## single_long (case-specific floor, D1 ruling 5)
Measured (job 947, historical: the D1-era record; the fixed record is now jobs 1061/1062 and the D4 re-record): pcc_normalized 0.99782 acc, 0.99433 perf. Proposed: 0.997 acc, 0.99 perf on pcc_normalized,
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

## Forward 2 (D3 item 1)

Forward 2 is teacher-forced: context 3072 = the case's 2048 plus the reference's forward-1 median (1024 steps), S = 129
padded to 256 (chunk 128). It is gated as `f2_*` on long_horizon (bench) and horizon4 (suite, new). The checker gates a
family where the record or its fixed record has it, and fails a family with no floors or no drift, a dropped family,
and a candidate set missing a fixed-record case (`--partial` for screens only).

Why long_horizon's forward 2 has the wider band spread (`/home/andy/bringup/scratch/tafsut/f2rows.py` on the job
1040 dumps; the gate metrics are bit-identical in 1061): the last 1024 context steps are a smooth median, so the
model forecasts **narrower bands**. Median q0.1-q0.9 half-width over active rows is 0.276 normalized in forward 2 vs
0.709 in forward 1 (p10 0.063 vs 0.094), and 29 active rows have a half-width under 0.05 (4 in forward 1). The same
absolute error is then a larger share of the band, and the absolute errors are also larger than forward 1's.

| precision | worst row (metric) | row std | half-width | mean abs err | band err / half-width | mean err / row std |
|---|---|---|---|---|---|---|
| accuracy | 393 (spread_ratio_min 0.762) | 0.221 | 0.174 | 0.016 (max 0.416) | 0.12 | 0.074 |
| accuracy | 16 (spread_ratio_max 1.132) | 0.342 | 0.441 | 0.031 | 0.07 | 0.090 |
| accuracy | 228 (pcc_row_min_active 0.9955) | 0.235 | 0.258 | 0.022 | 0.06 | 0.095 |
| performance | 16 (spread_ratio_max 1.493, pcc_row_min_active 0.9695) | 0.342 | 0.441 | 0.114 (max 0.780) | 0.25 | 0.334 |
| performance | 250 (spread_ratio_min 0.822) | 0.158 | 0.184 | 0.016 | 0.08 | 0.102 |

(Forward 1's worst rows at accuracy: band error 0.01-0.02 x half-width.) Row 16 at performance is the outlier: its band
is 1.49x wide, a quarter of the half-width off. That is the bf8/LoFi preset on a long, smooth context.

Proposed case floors (long_horizon; measured in brackets):
- accuracy: `f2_spread_ratio_min` 0.70 (0.762), `_max` 1.20 (1.132), `_p01` [0.87, 1.05] (0.901), `_p99` [0.95, 1.08]
  (1.054), `f2_pcc_row_min_active` 0.993 (0.9955), `f2_bias_row_max` 0.045 (0.0286).
- performance: `f2_spread_ratio_min` 0.75 (0.822), `_max` 1.6 (1.493), `_p99` [0.85, 1.35] (1.305),
  `f2_pcc_row_min_active` 0.96 (0.9695), `f2_bias_row_max` 0.17 (0.112), `f2_bias_row_p95` 0.05 (0.034).
- Every other f2 floor is the precision's forward-1 floor. horizon4's forward 2 passes those unchanged.

**Forward-2 misses of the existing floors (v1.10)**, all on long_horizon: accuracy `spread_ratio_min` 0.762 < 0.92,
`spread_ratio_p01` 0.901 < 0.95, `spread_ratio_p99` 1.054 > 1.05, `spread_ratio_max` 1.132 > 1.08,
`pcc_row_min_active` 0.9955 < 0.998; performance `spread_ratio_p99` 1.305 > 1.15, `spread_ratio_max` 1.493 > 1.08,
`pcc_row_min_active` 0.9695 < 0.97, `spread_ratio_min` 0.822 < 0.92. horizon4: none.

Every broken output also scores on forward 2 (36 scorings: long_horizon and horizon4, both precisions): all fail.

## Row-bias tail (D3 item 3)

`bias_row_max` (any row) and `bias_row_p95` (>= 100 rows) join `bias_row_p50`. `shift10pct` adds +0.05 to the first
ceil(10 %) of the active rows. It fails on every case and forward in both precisions, **except horizon4 / forward 1 /
performance** (1 of 4 rows). There, the preset's own worst row bias is 0.056, above the 0.05 shift. A broken output
made from the golden can only lower every row statistic below the measured record, so no floor on row bias magnitude
separates the two without failing round 0.

Near-miss: long_horizon / forward 1 / performance is caught only by the `spread_ratio_p50` two-sided drift. The
golden-made output has ratio 1.0 against the record's 1.03, so that catch is an artifact. Measured there: the
row-bias p95 is 0.048 and the max 0.248 (row 283, std 0.97, error 0.26 x row std). The shift sits inside the
preset's own tail.

Per-row bias at round 0 (all rows; scratch `biasdist.py`): accuracy max 0.013 paper, 0.079 long_horizon f1, 0.029
f2, <= 0.015 elsewhere; performance max 0.082 paper, 0.248 long_horizon f1, 0.112 f2, 0.056 horizon4, <= 0.021
elsewhere. Bias relative to the row std does not separate either (performance up to 0.43 x row std on paper).

## D4 (2026-10-28): row-indexed drift, the named rows, the new fixed record
Commits: `e36066bff49` (checker, metrics, probes, row-stage test; gates.json adopted), `98e205a7d21` (fixed record),
`dc6d14d9e48` (unit-test fix: the golden control is excluded from the must-fail set and must itself be clean).

**Fixed record:** bench job 1108 (10 passed) and full `tests/pcc/` job 1109 (**159 passed**: 147 + 12 row-stage),
both at `e36066bff49`, clean, started 11 s after the commit. 1040 gated scalars compared with 1061/1062: **0
differ** (bit-identical). The new records add `row_active_idx`, `bias_row_vec` and `spread_ratio_row_vec` (f1 and f2).
Paper/performance device time 1653.7 ms (1697.5 in 1061; metrics identical, a timing difference only).

**Row-indexed drift (items 1-2).** `bias_row_drift_max` floor 0.03 at both presets, f1 and f2, with no relative term.
`spread_ratio_row_drift_max` floor 0.15 at both presets. The measurement behind 0.15: a whole preset change
(accuracy -> performance, which is much larger than any one candidate) moves a row's ratio by p99 0.02-0.09 on every
forward-1 case (max 0.29, long_horizon), and by p99 0.27 on long_horizon f2. The accuracy preset's own distance from
fp32 reaches 0.24 on f2 (row 393). A tighter bound would trap accuracy candidates that move such rows back toward 1.
A 1.3x band moves a row by 0.3 x r0 (>= 0.2 for r0 >= 0.67). Reruns are bit-identical: the drift against itself is
0 on every case.

**Probes (score_broken at `dc6d14d9e48`, CPU job 1129, `broken_g1.txt`).** 642 broken scorings and 22 golden controls.
- **The golden itself fails 0 checks** in every case, forward and preset (item 5's re-centred p50 drift).
- **0 broken outputs pass.**
- The 442 `level_shift_rows#0-7`, `band1.3x_row#0-7` and `shift10pct` scorings are each caught by their own
  row-indexed drift, on every case, forward and preset. No catch is vacuous.
- How the row vectors are made: the probe's own row change is composed onto the fixed record's device rows (e0 + e,
  r0 x r). This is exact for shifts and band scalings.

**Named rows (item 3; re-run in round 1 as job 1145, see `iter_01/row_stages_1145.txt`, which also covers the band
floors' rows 121, 394, 250, 973 and the tail isolation of rows 283 and 536; `test_row_stages.py`, job 1109 `row_stages_1109.txt`; conditioning in the same job).** The device
ran each case's forward-k input (forward 2 teacher-forced) on the named rows plus 5 control rows. Every sub-batch row
reproduces the full-batch metrics (row 16 perf 1.4925 / 0.9695, row 283 +0.2483, row 536 -0.0820, row 1 -0.0561).
Relative L2 error of the future rows, embedding -> block 13 (the last; NL = 14) -> head (D5 R1-0c: the old
text said "block 11", a label error of the test's printout, fixed in c2cb103069a):
- **row 16** (lh f2): acc 0.0016 -> 0.018 -> 0.036; perf 0.0016 -> 0.090 -> 0.136. Perf controls reach 0.03-0.08
  at block 13.
- **row 311** (lh f2, perf): 0.0016 -> 0.115 -> 0.257.
- **row 283** (lh f1): acc 0.034 -> 0.086 at the head; perf 0.106 -> 0.273. Control row 51 also reaches 0.088 at
  block 13, but only 0.045 at the head.
- **row 536** (paper, perf): 0.045 -> 0.189.
- **row 1** (horizon4, perf): 0.050 -> 0.062.

In every row the error grows smoothly through the stack, at the same per-block rate as the controls; the performance
preset's bf8/LoFi step shows from block 0 in every row. No row has a jump at one block. What differs is the head's
gain on the last block's error. **Verdict: precision, not a defect.** These rows are ill-conditioned: under bf16-sized
weight noise (2^-9, fp32 reference, 3 seeds), their bias moves more than that of 94-100 % of 48 control rows:

| row | case | d bias, rank | d ratio, rank |
|---|---|---|---|
| 16 | lh f2 | 7.0e-3, 94 % | 2.5e-2, 92 % |
| 311 | lh f2 | 9.1e-3, 96 % | 2.1e-3, 4 % |
| 393 | lh f2 | 6.2e-3, 94 % | 1.0e-1, 98 % |
| 283 | lh f1 | 1.1e-2, 100 % | 8.4e-4, 6 % |
| 536 | paper | 5.5e-3, 100 % | 1.7e-3, 65 % |
| 1 | horizon4 | 4.2e-3, 67 % of 3 | 1.9e-3, 33 % |

- Row 16 is sensitive in both bias and band. Its |spread| mean is 0.213, so it is not narrow; it is a conditioning
  row, not a narrow-band row.
- Rows 311, 283 and 536 are sensitive in bias only, so their band floors are not needed for them.
- **horizon4 row 1 is not distinguishable from its controls** (4 rows only). Its case floor `bias_row_max` 0.08 is
  therefore not justified by conditioning: D4 says it reverts to the `*/performance` floor (0.04). Its round-0
  value of 0.056 then fails that floor, so this is a finding to fix or to take to an X gate. It is listed as an
  open item in GATE.md and has not been changed here.
