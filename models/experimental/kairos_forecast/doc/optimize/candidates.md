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
| 1 | R1-5a | attention mask add fused into the softmax (`scale_mask_softmax`, bf16 0/NEG key masks as one tile row; causal decoder self-attention kept unfused: the fused op takes only a row-broadcast mask) | not timed | - | - | **2 FAIL**: single/accuracy bias_row_p50_final / bias_row_max_final 0.00192 vs limit 0.00141 (baseline 0.00094), suite 1226 | **rejected** (check_gates); reverted in 3ee133f0653 | 376b8a6c047, ee5ac897762, bf079d4bad4, 71adadd9f6a | suites 1191, 1212 (mask shape / batch errors), 1226; quick 1222 |
| 1 | R1-4 | linears on 2D multicast program configs (sweep job 1188: kw8, the largest dst subblock, out block <= 84 tiles), relu fused in the epilogue (precision change, D3, disclosed); M < 80 tiles keeps the default | 2+2 screen: 50.49 vs 68.03 | **-25.8 %** (2+2) | 2+2: long_horizon -10.2 %, ragged -19.4 %, paper/perf -28.2 %; first pair 1235 had single_long_horizon/perf bimodal (13.8 vs 12.8 ms, code path unchanged at that M), repeated | 670 ok / 0 FAIL (suite 1234, 72 passed); row moves vs fixed: acc bias 0.0107 (ragged), spread 0.038 (long_horizon_tf final); perf bias 0.0032 | kept | 3ee133f0653 (re-applied over the R1-5a revert; first 8b81760c1dc / 9cea7b1b635) | sweep 1188, suite 1234, screens 1235, **1243** |
| 2 | R2-1 | linear_config pads a prime per-core M (embed pm 103 -> 104, out_block_h 1 -> 26); K blocking unchanged (order-only) | 2+2 screen: 48.33 vs 50.49 | **-4.3 %** (2+2) | 2+2: long_horizon +0.0 %, ragged -1.3 %, paper/perf -3.3 % | 0 failures (suite 1259, run with R2-2 on top; 72 passed) | kept | b279f7fcf1b | suite 1259, screen **1260** |
| 2 | R2-2 | causal mask pre-expanded to [B, 8, 32, 32] per batch (same values) | 2+2 screen: 47.88 vs 48.33 | -0.95 % (2+2; decoder-only op, about -1 % at full depth) | single_plain/accuracy guard +3.2 % (small case, noise level) | 0 failures (suite 1259) | **rejected** (< 2 %); reverted in 16ddbb97bc0 | 6095c15e9d4 | suite 1259, screen 1260 |
| 2 | R2-6 | fp32 softmax probabilities straight into P@V (no bf16 typecast) | not timed | - | - | suite 1284: 8 failed (TT_THROW: static CBs 2.09 MB > 1.57 MB L1; the R1-2b whole-K bmm_config doubles in0 in fp32) | **rejected** (infeasible with the R1-2b config); reverted in 76cc8f4ce4b | abb75a8b617 | suite 1284, screen 1285 (failed with it) |
| 2 | R2-8 | RoPE as multiply + addcmul (x cos, then + xR sin fused; precision change) | 2+2 screen: 46.54 vs 48.36 | -3.75 % (2+2) | ok | **3 FAIL** (suite 1290, 72 passed): long_horizon_tf/accuracy bias_row_drift_max 0.0228 > 0.02 (row 640, D5 X-gate line); single/accuracy bias_row_p50/max_final 0.00206 > 0.00141 | **rejected** (check_gates); reverted | a631fdcb03d | suite 1290, screen 1291 |
