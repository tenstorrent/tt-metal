# Round 0 summary (fixed record, D3 item 12)

- Commit 40e7529f66f (clean). Suite job 1122 (pcc, started 12:39:36, exit 0): 107 passed, 2 deselected (the slow CPU
  scan-emulation tests). That covers 9 cases x 2 precisions, the module tests, the B0/S5 fused-path tests, the trace
  test in both precisions plus its warning control, the dataflow tests and the checker unit tests. Bench job 1123
  (bench, started 12:48:18, exit 0): 6 shapes x 2 precisions.
- Fixed record: `fixed_record.json` = bench.json (1123) + suite.json (1122), 30 records. Every gated key is
  bit-identical to round 0 (bench 1088, suite 1094).
- check_gates (job 1133 @ 8137acd46a9): 0 failures (`check_gates_g1.txt`). score_broken (job 1136 @ ba9e377028f): all 246
  scorings are caught, perturbing the device outputs (bench dumps 1123, suite dumps 1135; 1135 is bit-identical to 1122)
  (`broken_outputs.txt` / `.json`).
- **No attention:** FlowState is an S5 SSM. F1/F9 and the SDPA tables of v1.20, v1.21 and v1.23 do not apply
  (CONTRACT:403).

| shape | accuracy device ms | performance device ms | accuracy steady ms | pcc_normalized acc / perf |
|---|---|---|---|---|
| paper (target) | 375.3 | 376.6 | 436.8 | 0.999996 / 0.999968 |
| long_context | 768.6 | 741.6 | 885.5 | 0.999997 / 0.999956 |
| long_horizon | 387.4 | 387.2 | 465.1 | 0.999989 / 0.999838 |
| scale_q | 291.7 | 294.4 | 312.2 | 0.999988 / 0.999903 |
| single | 100.0 | 100.1 | 105.9 | 0.999902 / 0.999790 |
| single_long | 768.7 | 742.0 | 875.0 | 0.999973 / 0.999723 |

Module tests, one module on its golden input (asserted): the lowest is 0.99997 in accuracy and 0.99963 in
performance (single layer 3). The stacked per-layer PCC of the full forward, report-only, was performance layer 4 =
0.99892. It carries the error of layers 0-3; layer 4 alone scores >= 0.9996.

Named rows (v1.25), long_horizon / performance:
- abs_err_max 1.472 is row 7, quantile 0.9, step 688. That is 2.0x the row's std (0.740) and 38 % of the golden band
  half-width there (3.84). At accuracy the same element errs by 0.457, and row 7's mean error is -0.011. This is
  precision (bfp8/LoFi MLP) on the widest-band tail of a 720-step horizon, not a defect. The gate is the 1.5x
  abs_err_max drift.
- bias_row_max 0.107 is row 2: 0.14x its std (0.764) and 0.15x its band RMS (0.731). At accuracy the same row is
  +0.021, with the same sign. This is precision, amplified 5x at performance (F4). The gate is row 2's own drift
  (0.03, `bias_row_drift_max`). 0.15 is an outer bound only.
