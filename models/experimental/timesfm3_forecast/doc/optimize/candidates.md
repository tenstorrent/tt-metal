# TimesFM-3 optimization candidates

Contract (`/home/andy/avan/fleet/timesfm3/CONTRACT.md`, v1):
- **Target:** `forward` (one single-pass CPM decode), `paper` = 1024 univariate series x 2048 -> 64 (66 tokens per
  row, read-out token 63), `accuracy` precision (bf16 weights and activations, HiFi2 matmuls, HiFi4 + fp32
  accumulation for norms and RoPE). Device time (median of 5 `forward_device` calls, each after a 0.3 s idle) is
  primary; steady wall time (median of 5 end-to-end calls) is secondary.
- **Guards** (none may regress > 3 % on device time): `long_context` 32 x 15360 -> 1024 (accuracy),
  `multivariate` 128 items x 8 variates x 2048 -> 64 (accuracy), `paper` at `performance` (bfloat8_b weights, LoFi).
- **Floor:** the full PCC suite passes in one `pcc` job; target `pcc_normalized` >= 0.999 and `pcc_final` >= 0.99;
  performance guard 0.995 / 0.98; no candidate more than 0.0005 below round 0's target `pcc_normalized`.
- **Gate:** accept when the target device time improves >= 2 % (wall for host-only changes), no guard regresses
  > 3 %, every floor holds (`compare_bench.py <cand> <best> --target paper:accuracy --pcc-key pcc_normalized
  --pcc-floor 0.995`, then the 0.999 floor by hand).
- **Exit:** two consecutive rounds each < 3 %, or matmuls > 65 % of FLOPs and non-matmul < 15 % of device time,
  or 6 rounds.

Headline: `forward`, paper, accuracy; median of 5 steady runs; device time after a 0.3 s idle.

| round | candidate | target wall ms | target device ms | vs best | guards (device ms) | PCC (norm / final) | decision | evidence |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline: stock ops, default (1D) matmul configs, S = 96 (66 tokens), head_dim 96; realistic bench goldens (PLAN 9) | 1988.8 | 1370.5 | — | long_context 180.8, multivariate 4998.1, paper/performance 1369.7 | norm 0.99987, spread 0.99963, row p01/min 0.698/0.329 (active rows 0.993/0.949), final 0.99999 | baseline | iter_00/summary.md, iter_00/bench.json, iter_00/proposals.md |
