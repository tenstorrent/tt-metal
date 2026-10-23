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
