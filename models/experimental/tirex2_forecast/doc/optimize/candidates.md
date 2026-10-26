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
| 1 | R1-2f: minimal_matmul blocks per role (sweep job 670) | — | 1028.0 (vs 1103.2 b2b) | -6.8 % | long_context 320.7 (323.0), multivariate 4065.8 (6346.6) | all floors ok; drift FAIL long_context median_err 0.0531 vs 0.0510 | held at X1 | iter_01/r1_2f/b2b_{base,cand}.json (job 697), matmul_sweep.md, mm_error.py |
