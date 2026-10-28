# FlowState r1.1 candidates (chip 0)

## Contract summary (CONTRACT.md v1.1, D0, D3)

- **Target:** `forward` at `paper` / accuracy: 64 series x 4038 -> 64, s = 1 (L = 4096), one encoder pass plus the
  CPM read-out; traced device time (median of 5 replays after a 0.3 s idle).
- **Guards (none may regress by more than 3 %):** `long_context` (16 x 16360 -> 48, s = 0.25, L = 16384),
  `long_horizon` (64 x 3382 -> 720), `scale_q` (256 x 675 -> 8, s = 6, L = 682), all accuracy; `paper` /
  performance (ships only if >= 5 % faster than accuracy and passing every floor, v1.12). Single-series records:
  `single` (1 x 2048 -> 64), `single_long` (1 x 16360 -> 48).
- **Floors:** `doc/optimize/gates.json` (D3): pcc_normalized >= 0.999 accuracy (a minimum) / 0.995 performance,
  pcc_final >= 0.99 / 0.98, band, row, bias, step and inactive-row floors, on every case in both precisions; the
  SSM convolution, Bu, state and Lbar stay fp32 in every class (D0 amendment 1).
- **Gate per candidate:** `check_gates.py` 0 failures in both precisions on every shape against
  `iter_00/fixed_record.json` (drift, row-indexed `bias_row_drift_max` <= 0.03 and `spread_ratio_row_drift_max`
  <= 0.15), the session stamp clean and started after its commit; accept at >= 2 % target device time.
- **Exit:** a round under 5 % in total (U4), two consecutive rounds under 3 %, matmuls > 65 % of FLOPs with
  non-matmul < 15 % of device time, or 6 rounds.

**No attention.** FlowState is an S5 state-space model: F1, F3, F9, the masked-SDPA tests and the SDPA tables of
PLAN v1.20, v1.21 and v1.23 do not apply (CONTRACT:403). What applies: F2, F4, F5, F6, F7, v1.9, v1.14.

## Rows

| id | candidate | commit | jobs | paper / accuracy device ms | vs best | check_gates | kept |
|---|---|---|---|---|---|---|---|
| R0 | round 0: route-K scan (fused layout), default matmul configs | 40e7529f66f | suite 1122, bench 1123, check_gates 1133, score_broken 1136 | 375.3 | - | 0 failures (fixed record) | baseline |
