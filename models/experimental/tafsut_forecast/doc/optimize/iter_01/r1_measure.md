# Round 1: R1-1 (fused head split + RoPE) and R1-2/R1-3 (2D matmul configs, relu fused)

All device jobs on chip 3, after the commit they cite. Fixed record: `iter_00/bench.json` / `suite.json` (1108/1109).

## Screens (BENCH_LAYERS=4, paper + long_context, device ms)
| job | change | paper acc | paper perf | long_context acc | long_context perf |
|---|---|---|---|---|---|
| 1186 | round 0 (TAFSUT_FUSED_ROPE=0) | 515.9 | 501.6 | 189.1 | 186.5 |
| 1187 | R1-1 fused rope | 301.5 | 287.4 | 189.1 | 186.5 |
| 1205 | R1-1 + R1-2/3 | 133.1 | 120.3 | 180.5 | 176.5 |

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

Suite 1190 (R1-1) 166 passed; suite 1221 (20fbce21f89) **166 passed**. PCC test_model 15/15 (1185, 1214).

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
