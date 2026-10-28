# Kairos-50M optimization candidates

## Contract summary (CONTRACT v1.1 + D0)

- **Target:** `paper/accuracy` device time: one traced forward of 512 rows (256 series as x and -x, GIFT settings,
  H = 64), N padded to 64 (L = 65 + pad -> 96 rows per series). Napkins: 9.75-16.2 ms and 11.6-17.1 ms.
- **Guards** (none may regress by more than 3 %): `long_horizon` (64 x 2048 -> 720, 6 + 6 sequential rollout
  forwards, accuracy), `ragged` (128 series, lengths 64..2048, accuracy), `paper/performance` (ships only if >= 5 %
  faster than accuracy and passes every floor). Single-series cases in the suite and bench (possibly bimodal, v1.18).
- **Floors:** `pcc_normalized` >= 0.999 (accuracy) / 0.995 (performance) and `pcc_final` >= 0.99 / 0.98 on every case
  (v1.20); routing exact (`route_mismatch_rows` == 0, `bias_max_abs_diff` <= 1e-6) on every case, report-only for
  free-running rollout steps >= 2; band / spread / row floors in `gates.json` (proposed at G1).
- **Gate:** `check_gates.py` against the fixed round-0 records (`iter_00/bench.json`, `iter_00/suite.json`), both
  precisions, every shape; drift budget from the fixed record (v1.17).
- **long_horizon records (D5 R1-0d):** the free-running `long_horizon` record scores only forward 0's x half in head
  space (its final space covers the whole rollout). The teacher-forced `long_horizon_tf` record, which scores every
  step's rows in both halves, is the guard for the -x half and for the later steps.
- **Accept:** >= 2 % of target device time, no guard worse than 3 %, check_gates passes; chip 1 drifts up to 5 %, so
  every decision re-runs the baseline back to back.
- **Exit:** a round whose accepted changes total < 5 % of target device time, or CONTRACT §4's criteria.

## Rounds

| round | id | change | target device ms | delta | guards | check_gates | verdict | commit | jobs |
|---|---|---|---|---|---|---|---|---|---|
| 0 | R0 | bring-up baseline: matmul attention (fp32 scores), host RoPE tables, packed embedding, traced | 691.29 (min 690.77) | - | long_horizon 1029.99, ragged 339.83, paper/perf 689.46 | 0 failures (284 checks) | baseline | b5da2729a8b | suite 1018, bench 1019, profile 1020 |
