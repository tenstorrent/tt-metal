# Row bias of the fixed record (D4 items 1-2)

From `models/experimental/kairos_forecast/doc/optimize/iter_00/suite.jsonl`.

## Outlier rows (head-space bias_row_max > 0.05)

One line per row; the bias columns give both precisions (D5 R1-0d: the ragged row was listed twice with the same numbers).

| record | precision | row | series | half | fwd | n_valid | ctx std | golden row std | band half-width | bias perf | bias acc | perf bias / half-width |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long_horizon | performance | 48 | 48 | x | 0 | 2048 | 9293 | 0.2683 | 0.3292 | +0.0692 | +0.0080 | 0.21 |
| long_horizon_tf | performance | 448 | 0 | -x | 7 | 2176 | 3284 | 0.5688 | 0.5316 | +0.1264 | +0.0192 | 0.24 |
| paper | performance | 15 | 15 | x | 0 | 2048 | 1090 | 0.6601 | 0.8295 | -0.0927 | -0.0205 | 0.11 |
| ragged | both (over 0.05 in each) | 238 | 110 | -x | 1 | 1422 | 1745 | 0.4524 | 0.6294 | +0.2873 | +0.0545 | 0.46 |

## Row drift between the precisions

max_i |b_perf_i - b_acc_i|: the row-indexed move of a whole precision change (bound 0.02 per candidate).

| record | head | final |
|---|---|---|
| long_horizon | 0.0613 | 0.0448 |
| long_horizon_tf | 0.1072 | 0.0758 |
| paper | 0.0722 | 0.0610 |
| ragged | 0.2328 | 0.1851 |
| single | 0.0221 | 0.0199 |
| single_long_horizon | 0.0089 | 0.0095 |
| single_long_horizon_tf | 0.0169 | 0.0184 |
| single_plain | 0.0177 | 0.0177 |
| single_short | 0.0107 | 0.0093 |

## Reading (D4 item 2)

- **No row is short.** n_valid is 1422-2176, so a short-n_valid explanation is ruled out. Routing is exact on every row (route_mismatch_rows 0).
- **The pattern is the same in both precisions.** On each outlier series the two flip halves carry head-space biases of
  **opposite sign** (ragged s110: x -0.039 / -x +0.054 accuracy, x -0.176 / -x +0.287 performance; paper s15: x -0.020 /
  -x +0.010, x -0.093 / -x +0.048). These do not cancel in (a - flip(b)) / 2: they add, so the final row is
  -0.047 (accuracy) and -0.232 (performance) of the scale on ragged s110. Performance is about 5x accuracy on the same row,
  in the same direction: the bfp8 / LoFi error grows the bias the accuracy path already has. It does not create a new one.
- **The rows are noisy, sign-changing series** with an asymmetric golden band (ragged s110: 626 leading NaN, context
  std 1745 around a mean of 461, golden head mean x +0.30 / -x -0.54; paper s15 the same shape). Their per-row bias is
  0.1-0.46 of the band half-width.
- **Status: open finding, not explained.** Nothing here shows a defect: the shape matches between the precisions, the
  routing is exact, and the module PCCs pass. But a 0.29-normalized row bias is not proven benign either. Proving it
  needs a per-layer dump of that one row (host fp32 vs device), which is a stage-E / round-1 item. Until then the
  row-indexed drift (bound 0.02) pins the row: a candidate cannot make it worse.
