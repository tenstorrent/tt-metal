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
| 1 | R1-0 | checker items R1-0a..d (spread row drift 0.15, NaN-safe row drift, s110 dump, provenance); fixed records re-recorded, bit-identical | - | - | - | 670 ok / 0 FAIL suite and bench; broken 832 / 0 uncaught | done | 880bc6f440f..f21ad87605f | suite 1137, bench 1138, dump 1140, broken 1141 |
| 1 | R1-1+R1-3 | fold rotate-half R into the q/k projection columns (from the stored weights, bitwise up to -0), cross-attn heads through split_query_key_value_and_split_heads | 2+2 screen: 143.18 vs 235.21 | **-39.1 %** (2+2) | 2+2: long_horizon -40.3 %, ragged -40.5 %, paper/perf -39.2 % | 670 ok / 0 FAIL (suite 1155); performance records bit-identical, accuracy rows move <= 2.9e-3 bias / 4.5e-3 spread (wider projection N changes the default matmul blocking: rounding order only, disclosed) | kept | 1224bb3ebcc, 78a7e595938 | suite 1142 (bfp8 bitwise check failed on -0, fixed), 1155, fold 1163, screen **1177** (screen 1143 void: its candidate ran on a dirty tree with R1-2b) |
| 1 | R1-2b | batched attention matmuls (scores, P@V) on the full grid: MatmulMultiCoreReuseProgramConfig, whole K per core, transpose_b instead of a transpose op | 2+2 screen: 68.22 vs 143.28 | **-52.4 %** (2+2) | 2+2: long_horizon -53.6 %, ragged -54.6 %, paper/perf -52.5 % | 670 ok / 0 FAIL (suite 1164, 72 passed); not bitwise vs R1-1 (accumulation blocking: rounding order, disclosed) | kept | e814cea6399 | suite 1164, screen 1165 |
