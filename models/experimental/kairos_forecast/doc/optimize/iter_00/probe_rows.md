# level_shift_rows probes (D4 item 1), broken_outputs.json, git_head bd966218ee3 dirty ''

Caught-by per record (bench set; suite set identical unless noted). n scored / uncaught at the end.

| case/precision | probe | record | caught by |
|---|---|---|---|
| paper/accuracy | level_shift_rows_h0 | paper | drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, floor:bias_row_max |
| paper/accuracy | level_shift_rows_h1 | paper | drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, floor:bias_row_max |
| paper/accuracy | level_shift_rows | paper | drift:bias_row_drift_max, drift:bias_row_max, floor:bias_row_max |
| paper/accuracy | level_shift_rows_final | paper | drift:bias_row_drift_max_final, drift:bias_row_max_final, drift:median_err, floor:bias_row_max_final |
| paper/performance | level_shift_rows_h0 | paper | drift:bias_row_drift_max, drift:bias_row_drift_max_final |
| paper/performance | level_shift_rows_h1 | paper | drift:bias_row_drift_max, drift:bias_row_drift_max_final |
| paper/performance | level_shift_rows | paper | drift:bias_row_drift_max |
| paper/performance | level_shift_rows_final | paper | drift:bias_row_drift_max_final |
| long_horizon/accuracy | level_shift_rows_h0 | long_horizon | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, floor:bias_row_max |
| long_horizon/accuracy | level_shift_rows_h0 | long_horizon_tf | drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:inactive_max_err, floor:bias_row_max, floor:inactive_max_err |
| long_horizon/accuracy | level_shift_rows_h1 | long_horizon | drift:bias_row_drift_max_final, drift:bias_row_max_final |
| long_horizon/accuracy | level_shift_rows_h1 | long_horizon_tf | drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:inactive_max_err, floor:bias_row_max, floor:inactive_max_err |
| long_horizon/accuracy | level_shift_rows | long_horizon | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_max, floor:bias_row_max |
| long_horizon/accuracy | level_shift_rows | long_horizon_tf | drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:inactive_max_err, floor:bias_row_max, floor:inactive_max_err |
| long_horizon/accuracy | level_shift_rows_final | long_horizon | drift:bias_row_drift_max_final, drift:bias_row_max_final, floor:bias_row_max_final |
| long_horizon/accuracy | level_shift_rows_final | long_horizon_tf | not applicable (identical to the golden) |
| long_horizon/performance | level_shift_rows_h0 | long_horizon | drift:bias_row_drift_max, drift:bias_row_drift_max_final |
| long_horizon/performance | level_shift_rows_h0 | long_horizon_tf | drift:bias_row_drift_max, drift:bias_row_drift_max_final |
| long_horizon/performance | level_shift_rows_h1 | long_horizon | drift:bias_row_drift_max_final |
| long_horizon/performance | level_shift_rows_h1 | long_horizon_tf | drift:bias_row_drift_max, drift:bias_row_drift_max_final |
| long_horizon/performance | level_shift_rows | long_horizon | drift:bias_row_drift_max |
| long_horizon/performance | level_shift_rows | long_horizon_tf | drift:bias_row_drift_max, drift:bias_row_drift_max_final |
| long_horizon/performance | level_shift_rows_final | long_horizon | drift:bias_row_drift_max_final |
| long_horizon/performance | level_shift_rows_final | long_horizon_tf | not applicable (identical to the golden) |
| ragged/accuracy | level_shift_rows_h0 | ragged | drift:bias_row_drift_max, drift:bias_row_drift_max_final |
| ragged/accuracy | level_shift_rows_h1 | ragged | drift:bias_row_drift_max, drift:bias_row_drift_max_final |
| ragged/accuracy | level_shift_rows | ragged | drift:bias_row_drift_max |
| ragged/accuracy | level_shift_rows_final | ragged | drift:bias_row_drift_max_final |
| ragged/performance | level_shift_rows_h0 | ragged | drift:bias_row_drift_max, drift:bias_row_drift_max_final |
| ragged/performance | level_shift_rows_h1 | ragged | drift:bias_row_drift_max, drift:bias_row_drift_max_final |
| ragged/performance | level_shift_rows | ragged | drift:bias_row_drift_max |
| ragged/performance | level_shift_rows_final | ragged | drift:bias_row_drift_max_final |
| single/accuracy | level_shift_rows_h0 | single | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:bias_row_p50, drift:bias_row_p50_final, drift:median_err, floor:bias_row_max, floor:bias_row_max_final, floor:bias_row_p50, floor:bias_row_p50_final |
| single/accuracy | level_shift_rows_h1 | single | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:bias_row_p50, drift:bias_row_p50_final, drift:median_err, floor:bias_row_max, floor:bias_row_max_final, floor:bias_row_p50, floor:bias_row_p50_final |
| single/accuracy | level_shift_rows | single | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_max, drift:bias_row_p50, floor:bias_row_max, floor:bias_row_p50 |
| single/accuracy | level_shift_rows_final | single | drift:bias_row_drift_max_final, drift:bias_row_max_final, drift:bias_row_p50_final, drift:median_err, floor:bias_row_max_final, floor:bias_row_p50_final |
| single/performance | level_shift_rows_h0 | single | drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_p50, floor:bias_row_max, floor:bias_row_p50 |
| single/performance | level_shift_rows_h1 | single | drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_p50, floor:bias_row_max, floor:bias_row_p50 |
| single/performance | level_shift_rows | single | drift:bias_row_drift_max, drift:bias_row_max, drift:bias_row_p50, floor:bias_row_max, floor:bias_row_p50 |
| single/performance | level_shift_rows_final | single | drift:bias_row_drift_max_final, drift:bias_row_max_final, drift:bias_row_p50_final, drift:median_err, floor:bias_row_max_final, floor:bias_row_p50_final |
| single_plain/accuracy | level_shift_rows | single_plain | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:bias_row_p50, drift:bias_row_p50_final, drift:median_err, floor:bias_row_max, floor:bias_row_max_final, floor:bias_row_p50, floor:bias_row_p50_final |
| single_plain/accuracy | level_shift_rows_final | single_plain | drift:bias_row_drift_max_final, drift:bias_row_max_final, drift:bias_row_p50_final, drift:median_err, floor:bias_row_max_final, floor:bias_row_p50_final |
| single_plain/performance | level_shift_rows | single_plain | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:bias_row_p50, drift:bias_row_p50_final, drift:median_err, floor:bias_row_max, floor:bias_row_max_final, floor:bias_row_p50, floor:bias_row_p50_final |
| single_plain/performance | level_shift_rows_final | single_plain | drift:bias_row_drift_max_final, drift:bias_row_max_final, drift:bias_row_p50_final, drift:median_err, floor:bias_row_max_final, floor:bias_row_p50_final |
| single_short/accuracy | level_shift_rows_h0 | single_short | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:bias_row_p50, drift:bias_row_p50_final, drift:median_err, drift:pcc_normalized, floor:bias_row_max, floor:bias_row_max_final, floor:bias_row_p50, floor:bias_row_p50_final |
| single_short/accuracy | level_shift_rows_h1 | single_short | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:bias_row_p50, drift:bias_row_p50_final, drift:median_err, drift:pcc_normalized, floor:bias_row_max, floor:bias_row_max_final, floor:bias_row_p50, floor:bias_row_p50_final |
| single_short/accuracy | level_shift_rows | single_short | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_max, drift:bias_row_p50, floor:bias_row_max, floor:bias_row_p50 |
| single_short/accuracy | level_shift_rows_final | single_short | drift:bias_row_drift_max_final, drift:bias_row_max_final, drift:bias_row_p50_final, drift:median_err, floor:bias_row_max_final, floor:bias_row_p50_final |
| single_short/performance | level_shift_rows_h0 | single_short | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:bias_row_p50, drift:bias_row_p50_final, drift:median_err, floor:bias_row_max, floor:bias_row_max_final, floor:bias_row_p50, floor:bias_row_p50_final |
| single_short/performance | level_shift_rows_h1 | single_short | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:bias_row_p50, drift:bias_row_p50_final, drift:median_err, floor:bias_row_max, floor:bias_row_max_final, floor:bias_row_p50, floor:bias_row_p50_final |
| single_short/performance | level_shift_rows | single_short | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_max, drift:bias_row_p50, floor:bias_row_max, floor:bias_row_p50 |
| single_short/performance | level_shift_rows_final | single_short | drift:bias_row_drift_max_final, drift:bias_row_max_final, drift:bias_row_p50_final, drift:median_err, floor:bias_row_max_final, floor:bias_row_p50_final |
| single_long_horizon/accuracy | level_shift_rows_h0 | single_long_horizon | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:bias_row_p50, drift:bias_row_p50_final, floor:bias_row_max, floor:bias_row_max_final, floor:bias_row_p50, floor:bias_row_p50_final |
| single_long_horizon/accuracy | level_shift_rows_h0 | single_long_horizon_tf | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:bias_row_p50, drift:inactive_max_err, drift:median_err, floor:bias_row_max, floor:bias_row_max_final, floor:bias_row_p50, floor:inactive_max_err |
| single_long_horizon/accuracy | level_shift_rows_h1 | single_long_horizon | drift:bias_row_drift_max_final, drift:bias_row_max_final, drift:bias_row_p50_final, floor:bias_row_max_final, floor:bias_row_p50_final |
| single_long_horizon/accuracy | level_shift_rows_h1 | single_long_horizon_tf | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:bias_row_p50, drift:inactive_max_err, drift:median_err, floor:bias_row_max, floor:bias_row_max_final, floor:bias_row_p50, floor:inactive_max_err |
| single_long_horizon/accuracy | level_shift_rows | single_long_horizon | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_max, drift:bias_row_p50, floor:bias_row_max, floor:bias_row_p50 |
| single_long_horizon/accuracy | level_shift_rows | single_long_horizon_tf | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:bias_row_p50, drift:bias_row_p50_final, drift:inactive_max_err, drift:median_err, floor:bias_row_max, floor:bias_row_max_final, floor:bias_row_p50, floor:bias_row_p50_final, floor:inactive_max_err |
| single_long_horizon/accuracy | level_shift_rows_final | single_long_horizon | drift:bias_row_drift_max_final, drift:bias_row_max_final, drift:bias_row_p50_final, drift:median_err, floor:bias_row_max_final, floor:bias_row_p50_final |
| single_long_horizon/accuracy | level_shift_rows_final | single_long_horizon_tf | not applicable (identical to the golden) |
| single_long_horizon/performance | level_shift_rows_h0 | single_long_horizon | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:bias_row_p50, drift:bias_row_p50_final, floor:bias_row_max, floor:bias_row_max_final, floor:bias_row_p50, floor:bias_row_p50_final |
| single_long_horizon/performance | level_shift_rows_h0 | single_long_horizon_tf | drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:bias_row_p50, drift:inactive_max_err, drift:median_err, floor:bias_row_max, floor:bias_row_max_final, floor:bias_row_p50 |
| single_long_horizon/performance | level_shift_rows_h1 | single_long_horizon | drift:bias_row_drift_max_final, drift:bias_row_max_final, drift:bias_row_p50_final, floor:bias_row_max_final, floor:bias_row_p50_final |
| single_long_horizon/performance | level_shift_rows_h1 | single_long_horizon_tf | drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:bias_row_p50, drift:inactive_max_err, drift:median_err, floor:bias_row_max, floor:bias_row_max_final, floor:bias_row_p50 |
| single_long_horizon/performance | level_shift_rows | single_long_horizon | drift:abs_err_max, drift:bias_row_drift_max, drift:bias_row_max, drift:bias_row_p50, floor:bias_row_max, floor:bias_row_p50 |
| single_long_horizon/performance | level_shift_rows | single_long_horizon_tf | drift:bias_row_drift_max, drift:bias_row_drift_max_final, drift:bias_row_max, drift:bias_row_max_final, drift:bias_row_p50, drift:bias_row_p50_final, drift:inactive_max_err, drift:median_err, floor:bias_row_max, floor:bias_row_max_final, floor:bias_row_p50, floor:bias_row_p50_final |
| single_long_horizon/performance | level_shift_rows_final | single_long_horizon | drift:bias_row_drift_max_final, drift:bias_row_max_final, drift:bias_row_p50_final, floor:bias_row_max_final, floor:bias_row_p50_final |
| single_long_horizon/performance | level_shift_rows_final | single_long_horizon_tf | not applicable (identical to the golden) |

All broken outputs: 576 scored, 0 uncaught; problems: 0
