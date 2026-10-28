# Round 1: R1-1 (fused head split + RoPE) and R1-2/R1-3 (2D matmul configs, relu fused)

All device jobs on chip 3, after the commit they cite. Fixed record: `iter_00/bench.json` / `suite.json` (1108/1109).

## Screens (BENCH_LAYERS=4, paper + long_context, device ms)
| job | change | paper acc | paper perf | long_context acc | long_context perf |
|---|---|---|---|---|---|
| 1186 | round 0 (TAFSUT_FUSED_ROPE=0) | 515.9 | 501.6 | 189.1 | 186.5 |
| 1187 | R1-1 fused rope | 301.5 | 287.4 | 189.1 | 186.5 |
| 1210 | R1-1 + R1-2/3 (dirty, pre-commit; see below) | 133.1 | 120.3 | 180.5 | 176.5 |

long_context (ctx 32768, s_pad 1088) keeps the matmul RoPE: its cos/sin table does not fit the op's L1 budget.
Job 1204 (R1-2's first try) overflowed L1 at accuracy: the CB budget did not count the fp32 intermediate; fixed.

## Full records
| case | round 0 (1108) | R1-1 bf16 dest (1189, 3c178de7da5) | R1-1 + R1-2/3 (1220, 20fbce21f89) |
|---|---|---|---|
| paper acc (target) | 1699.2 | 949.6 | **408.6 (-76.0 %)** |
| paper perf | 1653.7 | 904.1 | 367.2 (-77.8 %) |
| long_context acc / perf | 604.6 / 595.5 | 604.3 / 595.5 | 576.7 / 564.0 (-4.6 / -5.3 %) |
| long_horizon acc / perf | 2351.2 / 2272.2 | 1525.1 / 1447.2 | 713.8 / 645.8 (-69.6 / -71.6 %) |
| gift_ragged acc / perf | 604.4 / 595.4 | | 576.6 / 564.1 |
| gift_ragged_odd acc / perf | 75.0 / 71.4 | | 49.2 / 47.1 |

Suite 1190 (R1-1) 166 passed; suite 1221 (20fbce21f89) **166 passed**. PCC test_model 15/15 (1185, 1211: dirty, pre-commit).

## check_gates
- R1-1 with a bf16 DST (1216): 3 failures: paper/acc `median_err` drift 0.083 (bound 0.0509), `abs_err_max` drift
  0.061 (0.0474), horizon4/perf `bias_row_p50` drift 0.0077 (0.0063). Fixed by accumulating in fp32 DST (234e58c7463).
- R1-1 + R1-2/3 at 20fbce21f89 (CPU job 1236, `check_gates_1236.txt`), **4 failures**:
  - long_horizon/accuracy `f2_inactive_max_err` 0.0557 vs floor 0.05, and its drift (bound 0.0296), and the same
    value as `f2_flat_max_err` drift (bound 0.0298). Round 0 0.0198.
  - horizon4/performance `bias_row_p50` drift 0.0094 vs 0.0063 (round 0 0.0013). With 4 rows, p50 is the 2nd
    smallest |bias|: row 3 went 0.0010 -> 0.0094 (its row drift 0.0084 passes the 0.03 bound); row 1 -0.0561 ->
    -0.0619 (binding 0.08 holds).

## Where the failures come from (jobs 1237, 1240: same code, numerics knobs)
| variant | lh acc f2_inactive_max_err | paper acc median_err | paper acc abs_err_max | paper acc ms |
|---|---|---|---|---|
| round 0 (1108) | 0.0198 | 0.0407 | 0.0316 | 1699.2 |
| R1-1 bf16 DST (1189) | 0.0219 | **0.0830** | **0.0614** | 949.6 |
| R1-1 fp32 DST + R1-2 (1220) | **0.0557** | 0.0384 | 0.0384 | 408.6 |
| same, auto matmul configs (1237, MM_2D=0) | **0.0557** | 0.0384 | | 949.5 |
| same, embed + head on auto configs (1237) | **0.0557** | 0.0384 | | 416.9 |
| packer_l1_acc on (1237) | 0.0250 | **0.0851** | | 400.2 |
| fp32 cos/sin tables (1240; f2 at s_pad 256 falls back to matmul rope) | 0.0224 | 0.0445 | 0.0439 | 408.5 |
| fp32 tables, bf16 DST (1240) | 0.0224 | 0.0510 | 0.0626 | 408.5 |
| fp32 tables, unfused matmul rope (1240) | 0.0224 | **0.0795** | 0.0623 | 1183.4 |

- R1-2 does not move the long_horizon number (MM_2D=0 gives the same 0.0557). The fused RoPE with a fp32 DST does.
- The fused op is **not** defective: `test_qkv_heads_rope` (1251, a1555955fba) compares it with torch on the same
  bf16 inputs at s_pad 128 and 256. V is an exact split. Q/K max abs error is 1.47e-2 on values up to 4.5 (one bf16
  output rounding), and every sequence tile is alike, so no cos/sin row is misread. A bf16 DST gives 2.5x more.
  The fp32-DST op is more accurate per element than the round-0 matmul RoPE, which rounds three times.
- Every variant that is closer to the fp32 RoPE fails one single-element maximum or another (`median_err` and
  `abs_err_max` on paper, `f2_inactive_max_err` on long_horizon). Those maxima flip between 0.02 and 0.08 with the
  rounding order, so this is precision, not a defect. Population metrics (pcc_normalized 0.999996 paper acc,
  bias_row_max 0.0135 vs 0.0132) do not move.

## Job provenance (D7 condition c)
Corrections: the screen cited as 1205 is **1210**, and the test_model run cited as 1214 is **1211** (1205 and 1214 are
kairos jobs). Times are ttq start times on 2026-10-28.

| job | kind | started | ran on | status |
|---|---|---|---|---|
| 1210 | bench screen (R1-1 + R1-2/3) | 13:55:55 | before 20fbce21f89 (13:59:55) | **dev, dirty** (tt/common.py, tt/model.py) |
| 1211 | pcc test_model | 13:56:46 | before 20fbce21f89 | **dev, dirty** |
| 1217 | bench | 13:59:00 | before 20fbce21f89 | **dev, dirty** |
| 1237 | bench, numerics knobs (`/tmp/taf_lh_sweep.sh`) | 14:17:48 | 20fbce21f89 + uncommitted knobs | **dirty**; superseded by the clean control below |
| 1240 | bench, rope table knobs (`/tmp/taf_rope_sweep.sh`) | 14:22:05 | 20fbce21f89 + uncommitted knobs | **dirty** |
| 1249 | pcc test_qkv_heads_rope (failed) | 14:28:12 | before a1555955fba (14:28:54) | **dev, pre-commit** |
| 1250 | pcc test_qkv_heads_rope | 14:28:35 | before a1555955fba | **dev, pre-commit** |
| 1251 | pcc test_qkv_heads_rope | 14:28:55 | a1555955fba | clean |

Clean evidence for the gate: bench 1220 and suite 1221 (20fbce21f89), check_gates 1236. The knobs used by 1237 and
1240 are committed in 432192c8f70 (default off: `TAFSUT_PACKER_L1`, `TAFSUT_MM_2D_SKIP_HIFI4`,
`TAFSUT_ROPE_FP32_TABLE`, `TAFSUT_ROPE_FP32_DEST`), and the MM_2D=0 attribution control reruns as a stamped job
(D7 condition b, below).

## The long_horizon f2 element (D7 condition a; bench 1256, e1e649b014b, clean)
1256 reproduces 1220's gated metrics bit for bit (only timings differ). Records now carry `<key>_at` for every
`*_max_err`, `abs_err_max` and `median_err` (row, read-out token, step in its patch, horizon step, quantile, golden,
device value, golden row std).

| preset | metric | value | row | token / step | f2 horizon step | quantile | golden -> device | row std |
|---|---|---|---|---|---|---|---|---|
| accuracy | f2_inactive_max_err = f2_flat_max_err | 0.05575 | 60 | 23 / 29 | 765 (series step 1789) | 8 (top) | 0.0909 -> 0.0351 | 0.0267 |
| performance | f2_inactive_max_err = f2_flat_max_err | 0.1164 | 60 | 17 / 29 | 573 (series step 1597) | 8 (top) | 0.0531 -> -0.0633 | 0.0267 |

- Forward 2 runs at context 3072: P = 96 patches, s_pad 256 (8 sequence tiles), read-out tokens at positions 96-127 =
  sequence tile 3. Token 23 is position 119 (row 23 of tile 3), token 17 is position 113 (row 17): neither sits on a
  tile edge (rows 0 / 31), and both are far from the teacher-forced boundary (f2 step 0) and the forward's end (1023).
- Same row (60, flat and inactive, std 0.027) and quantile in both presets, different token: an ill-conditioned row,
  not a fixed position. `test_qkv_heads_rope` runs ctx 3072 / s_pad 256 and checks every sequence tile, tile 3
  included. No X gate.

## Clean attribution control (D7 condition b)
Job 1257 at 904a7b07c5d (clean, knobs committed in 432192c8f70): `TAFSUT_MM_2D=0`, long_horizon/accuracy gives
`f2_inactive_max_err` = 0.05574963241815567, bit for bit the value of 1220 and 1256 (same element: row 60, token 23,
step 29, quantile 8). R1-2 does not move it; the attribution to the fp32-DST fused RoPE stands.
`test_qkv_heads_rope` (1258, ddc557fdb8f): 7 passed, B = 3 and B = 1024 at s_pad 128, B = 3 at s_pad 256, per-tile
bound; the misread-tile control fails on tile 1 only (err/max 1.73 vs <= 0.0035 elsewhere).

## R1-2/3 profile and isolation (D7 condition d)
Tracy profiles of the reduced forward (jobs 1261-1264 at 904a7b07c5d, `profile_<preset>_mm2d{0,1}/`); both arms carry
R1-1 (fused RoPE, fp32 DST), so the difference is R1-2/3 alone. Per ordinary layer (x 13), device us:

| op | acc auto (MM_2D=0) | acc 2D | delta | perf auto | perf 2D | delta |
|---|---|---|---|---|---|---|
| QKV matmul | 14050 | 5012 | -9037 | 12552 | 3272 | -9280 |
| wo matmul | 4439 | 1498 | -2940 | 4256 | 1307 | -2949 |
| ff-up matmul | 17381 | 5160 | -12221 | 16640 | 4395 | -12245 |
| relu (R1-3: fused into ff-up) | 4055 | 0 | -4055 | 4068 | 0 | -4068 |
| ff-down matmul | 17146 | 4877 | -12269 | 16009 | 4377 | -11632 |
| **per layer** | | | **-40.5 ms** | | | **-40.2 ms** |

x 13 layers = -527 ms (acc) / -522 ms (perf); embed -6.1 / -5.9, last layer -10.5 / -11.3, head -2.1 / -2.0.
Profile estimate 954.6 -> 408.8 ms (-545.8) at accuracy and 904.3 -> 365.2 (-539.1) at performance, matching the
benches. The auto-config matmuls ran at 6-23 % of FLOPs on 108 cores; the 2D configs at 23-75 %. The -541 ms is the
matmul configs plus the removed relu pass, nothing else.

Variance-table row (paper, full bench, isolation of R1-2/3 on top of R1-1; 1265 MM_2D=0 vs 1256, both clean):

| preset | MM_2D=0 (1265) | MM_2D=1 (1256) | delta | gated metrics |
|---|---|---|---|---|
| accuracy | 949.0 ms | 408.5 ms | -57.0 % | identical (median_err 0.03838, abs_err_max 0.0384, bias_row_p50 0.00091) |
| performance | 904.3 ms | 367.3 ms | -59.4 % | identical (median_err 0.39463, abs_err_max 0.38025, bias_row_p50 0.00457) |

## Checkers at the new HEAD (D7 condition f)
- Suite 1266 (904a7b07c5d, clean): **173 passed** (166 + the 7 `test_qkv_heads_rope` cases of D7 e). `suite_job1266.json`.
- check_gates 1286 (CPU) on bench 1256 + suite 1266, with the D7 ceilings: **0 failures** (`check_gates_1286.txt`).
  long_horizon/accuracy `f2_inactive_max_err` / `f2_flat_max_err` 0.0557496 <= 0.06; horizon4/performance
  `bias_row_p50` 0.0094002 <= 0.015. `test_check_gates.py::test_d7_ceilings` checks each ceiling fails just above itself.
- score_broken 1267 (CPU): **0 broken outputs pass**, the golden fails 0 cases (`broken_job1267.txt`).
  long_horizon/accuracy f2 `inactive_collapsed` (84 inactive rows) fails `f2_flat_max_err` and `f2_inactive_max_err`
  under the 0.06 ceiling (and `f2_pcc_row_min`).
