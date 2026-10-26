# TiRex-2 optimization candidates

Contract: `/home/andy/avan/fleet/tirex2/CONTRACT.md` (version 1). Method: `model_bringup_optim` (the Timestral method).

- **Target:** `forward`, one pass (`tta_sign_flip=False`, `pad_context=True`), shape `paper` (1024 univariate series x
  2048 -> 64; 285 tokens per row, 192 all-NaN left pads), precision `accuracy` (bf16 weights and activations,
  HiFi2+ matmuls; recurrent states and exponential gating fp32). Device time is primary: median of 5
  `forward_device` calls (traced replay), each after a 0.3 s idle. Steady wall time (median of 5 end-to-end calls)
  is secondary.
- **Guards** (none may regress by more than 3 %): `long_context-accuracy` (32 x 8192 -> 928),
  `multivariate-accuracy` (128 groups x 8 rows x 2048 -> 64), `paper-performance` (bfp8 weights, LoFi).
- **Correctness floor:** the full PCC suite passes in one `pcc` job; target `pcc_normalized` >= 0.995 and
  `pcc_final` >= 0.99 (plus the median check and the PLAN section 9 metrics `pcc_spread`, `pcc_row_*`: floors set at G1); `performance` guard >= 0.99 / 0.98; any candidate's target
  `pcc_normalized` no more than 0.0005 below round 0 without an X gate.
- **Gate:** accept at >= 2 % target device time (wall for host-only changes), no guard regressing by > 3 %, every
  floor holding; the baseline is re-run back to back with the candidate on chip 1 (`compare_bench.py <cand>
  <baseline_rerun> --target paper:accuracy --pcc-key pcc_normalized --pcc-floor 0.99`).
- **Exit:** two consecutive rounds each < 3 %, or matmuls > 65 % of FLOPs and non-matmul < 15 % of device time, or
  6 rounds.

| round | candidate | target wall ms | target device ms | vs best | guards (device ms) | PCC (norm / final) | decision | evidence |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline: stock ttnn ops, op-level sLSTM loop (285 steps x ~27 ops), chunkwise mLSTM (5 x 64), traced forward, default dense program configs | 4518 | 3857 | — | long_context 371, multivariate 10990, paper-performance 3792 | norm 0.99991 / spread 0.99987 / row_min_var 0.959 / final 0.99997 | baseline | iter_00/summary.md, iter_00/bench.json (realistic mix, job 499), profile job 455; smooth-input run iter_00/bench_smooth.json (job 440: 3855 ms) |
| 1 | R1-0: performance policy w_in HiFi4 bf16 + HiFi2 embed/mlstm/head (accuracy unchanged) | — | perf 3806.7 vs 3792.2 | +0.4 % (perf) | — | perf row_min_act 0.846 -> 0.965, sr p01 0.867 -> 0.938 | proposed at G1b | iter_01/r1_0/precision_sweep.md (job 593), ae5d9811e3b |
| 1 | R1-1: constant left-pad skip (cut 192, cached fp32 pad states, chunk 64, T' = 128) | — | 1968.7 (base 3871.0 b2b) | -49.2 % | perf 1913.8 (r1-1 only) | norm 0.99995 / row_min_act 0.9935 / final, all floors + drift ok | accepted | iter_01/r1_1/b2b_{base,r11}.json (job 626), c2437b5172e |
| 1 | R1-2a+b: flatten B into M for every dense linear + fused SwiGLU (minimal_matmul) | — | 1144.4 (vs 1968.7 b2b) | -41.9 % | long_context 325 (372), multivariate 6060 (11051), perf 1138 | row_min_act 0.9945, all floors + drift ok | accepted | iter_01/r1_1/b2b_r12ab.json (job 626), bench_cand.json (job 618), c52e07a3f30 |
| 1 | R1-2d: fp32 `W_in` gate bias inside the matmul (drops one fp32 add per sLSTM block) | — | 1103.4 (vs 1144.3 b2b) | -3.6 % | — | all floors + drift ok | accepted | iter_01/r1_23/b2b_{base,r12d}.json (job 668), PCC job 667, 7c92331e8f2 |
| 1 | R1-3: read-out cut (block 11 FFN + head on the 64-token window) | — | 1090.7 (vs 1103.4 b2b) | -1.2 % | long_context 321.6 (325.9), multivariate 6023.7 (6061.7) | PCC job 667 passed | rejected (< 2 %); code reverted | iter_01/r1_23/b2b_r13.json (job 668) |
| 1 | note (D4 d): performance vs accuracy after R1-2a/b: paper 1137.4 vs 1144.4 (-0.6 %), long_context 324.1 vs 325.9 (-0.6 %), multivariate 6062.7 vs 6060.2 (+0.04 %) (job 618) | | | | | | non-ship class (D4) | iter_01/r1_1/bench_cand.json |
| 1 | R1-2f: minimal_matmul blocks per role (sweep job 670) | — | 1028.0 (vs 1103.2 b2b) | -6.8 % | long_context 320.7 (323.0), multivariate 4065.8 (6346.6) | all floors ok; drift vs fixed round-0 ref (r1_1/b2b_base.json): accuracy ok (long_context median_err 0.0531 <= 0.0580); performance paper sr p99 1.199 > 1.186 (floor 1.25), provisional per D5 | accepted (D5, rounding/epilogue-precision); multivariate gain to be re-stated vs ~6060 ms per D5(d) | iter_01/r1_2f/b2b_{base,cand}.json (job 697), matmul_sweep.md, mm_error.py |

Drift reference (D5): every drift check is against the fixed round-0 record `iter_01/r1_1/b2b_base.json` (check_gates.py default); a b2b base is for timing only.

D5 (b) broken outputs on the suite cases under the D5 row-count rule (CPU, `tests/perf/broken_suite.py` @ e895f50194c, `iter_01/r1_2f/broken_suite.json`), both precisions: 22 misses. 10 are vacuous (`inactive_collapsed` on cases with 0 inactive rows: model16, single, multivar4, long_context[0], long_context[7]; the output equals the golden). 12 are the uniform 0.88x / 1.12x band scales on the 1-active-row cases (single, long_context[0], long_context[7]), where only min/max (0.70 / 1.43, perf 0.60 / 1.67) and PCC apply, as D5 decided. Every other case/broken pair fails at least one floor; the bench gates all of them at 1024 rows.
| 1 | R1-2f on commit d64a038ae27 (D5 b): suite 45 passed (job 735); bench both precisions (job 736) | — | 1027.9 (perf 1007.2) | -6.8 % vs 1103.2 | long_context 320.6, multivariate 4065.1 (perf 319.5 / 3925.8) | check_gates vs fixed ref: all accuracy ok; paper/perf sr p99 1.199 > 1.186 (provisional, D5 item 4 -> X2) | committed | iter_01/r1_2f/commit/ |
| 1 | R1-2f variance table (D5 c, job 737 on dd754877cbf): base, R1-2f, one role group reverted at a time | | see table | | multivariate: variate 1352, w_in 431, mlstm 360, embed+head 139 ms of the 2282 ms gain | single-role p99 moves on paper/perf <= 0.012 (no role >= 0.03) | X2 raised | iter_01/r1_2f/variance/table.md |
| 1 | R1-2f multivariate explanation (D5 d): profiles 748 (base) / 749 (R1-2f): dense ttnn.linear 1280 -> 237 ms, minimal_matmul +212 ms (reduced layers). The ~6340 ms base reproduces (697, 737 x2 precisions), against 6060 in 618/668: vs 6060 the gain is -33 %, vs 6340 -36 % | | 4065 (vs ~6060-6340) | -33 % (vs 6060) | | | recorded vs 6060 | iter_01/r1_2f/mv_profile/compare.md |
| 1 | mm_error (D5 e): minimal_matmul vs ttnn.linear per role incl. variate w_wi (SiLU, no bias), both precisions, max abs delta, positive control | | | | | minimal rms <= linear on every role (embed / variate SiLU lower); every control fails (bfp4 control on LoFi roles after 738 showed LoFi + acc16 alone cannot fail there) | passed (job 785, 6af7772ff84) | iter_01/r1_2f/mm_error_{738,785}.log |

Note: FFN `w_down` (1408 x 512) is not in `MM_BLOCKS`, so R1-2f never touched it (rev_ffn_down equals R1-2f exactly); the (2048, 512) entry is variate `w_wo`.
