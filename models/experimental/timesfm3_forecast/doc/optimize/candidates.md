# TimesFM-3 optimization candidates

Contract (`/home/andy/avan/fleet/timesfm3/CONTRACT.md`, v1.4; gates and drift of D3 in `tests/golden_io.py` `GATES`):
- **Target:** `forward` (one single-pass CPM decode), `paper` = 1024 univariate series x 2048 -> 64 (66 tokens per
  row, read-out token 63), `accuracy` precision (bf16 weights and activations, HiFi2 matmuls, HiFi4 + fp32
  accumulation for norms and RoPE; since the G1 re-raise also fp32 accumulation on every matmul and SDPA). Device time (median of 5 `forward_device` calls, each after a 0.3 s idle) is
  primary; steady wall time (median of 5 end-to-end calls) is secondary.
- **Guards** (none may regress > 3 % on device time): `long_context` 32 x 15360 -> 1024 (accuracy),
  `multivariate` 128 items x 8 variates x 2048 -> 64 (accuracy), `paper` at `performance` (R1-0: bfloat8_b QKV weights, HiFi2, fp32 acc).
- **Floor:** the full PCC suite passes in one `pcc` job; every gate of `GATES` (pcc_normalized, pcc_spread,
  per-row PCC on all and active rows, per-row spread ratio p01 / p99 / min / max, pcc_final, pcc_final_spread) per
  precision, as set at G1; drift from round 0 per `DRIFT` (`tests/perf/check_gates.py`).
- **Gate:** accept when the target device time improves >= 2 % (wall for host-only changes), no guard regresses
  > 3 %, every floor holds (`compare_bench.py <cand> <best> --target paper:accuracy --pcc-key pcc_normalized
  --pcc-floor 0.995`, then the 0.999 floor by hand).
- **Exit:** two consecutive rounds each < 3 %, or matmuls > 65 % of FLOPs and non-matmul < 15 % of device time,
  or 6 rounds.

Headline: `forward`, paper, accuracy; median of 5 steady runs; device time after a 0.3 s idle.

| round | candidate | target wall ms | target device ms | vs best | guards (device ms) | PCC (norm / final) | decision | evidence |
|---|---|---|---|---|---|---|---|---|
| 0 (superseded) | baseline: stock ops, default (1D) matmul configs, S = 96 (66 tokens), head_dim 96; HiFi2 without fp32 accumulation; realistic bench goldens (PLAN 9) | 1988.8 | 1370.5 | — | long_context 180.8, multivariate 4998.1, paper/performance (bf8 + LoFi) 1369.7 | norm 0.99987, spread 0.99963, active-row min 0.949, spread ratio min 0.397 | superseded at the G1 re-raise (D1: band shrink on narrow rows) | iter_00/accumulation_sweep.md (A0), job 497 |
| 0 | baseline as above with the G1 policy: fp32 dest accumulation on every matmul and SDPA (A1); performance = bf8 except FFN + LoFi + fp32 acc (P2a) | 1971.5 | 1372.0 | — | long_context 232.0, multivariate 5110.0, paper/performance 1114.7 | norm 0.99993, spread 0.99988, active-row min 0.993, spread ratio p01 0.930 / min 0.689, final_spread 0.99993 | baseline | iter_00/summary.md, iter_00/bench.json (job 536, HEAD 81ad9734042) |
| R1-0 | performance policy (D3 cond. 1): P6b, bf8 on QKV weights only, HiFi2 + fp32 acc everywhere; 8 variants swept (P2a, P5a-d, P6a-d); every other bf8 mix or LoFi fails the paper row/spread floors | — | 1371.3 | — | paper/performance re-baselined 1132.7 (was 1114.7 under P2a, which fails the D3 floors); long_context 231.2, multivariate 5110.8 | perf paper: row p01 0.777, row min 0.514, active min 0.992, spread min 0.694 | kept (policy, not a speed candidate) | iter_00/r1_0/accumulation_sweep.md (jobs 566, 568), iter_00/r1_0/bench.json (job 578), perf suite job 577, test_host job 567 |
