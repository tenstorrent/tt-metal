# Round 3 proposals (G2-r3: automatic per skill section 7; no exception)

Baseline 23a61275446, paper/accuracy 501.6 ms (iter_02/bench.json, job 746). Drift reference stays r1_1 (D4).

| id | change | expected (paper / multivariate) | risk | measurement |
|---|---|---|---|---|
| R3-1 | packed variate attention for per-variate padding too: fold var_mask into the (tiles, 1, 32, 32) block-diagonal mask | 0 / up to -200 ms on padded inputs | medium (mask semantics, padded variates) | module tests mv (V = 8 padded), positive control, bench (+ a padded multivariate bench record) |
| R3-2 | model-local generic_op: RoPE + QK RMSNorm over q and k in one pass (replaces 2 RoPE + 2 norms, 4.98 ms/layer) | ~-60 / -80 ms | medium-low | micro-benchmark first (<= 2.5 ms/layer, PCC >= 0.99999 vs chain), op test, suite, bench |
| R3-3 | minimal_matmul block re-tune for wo, w_ov, ff0, ff1 at 65536 / 16384 rows (one sweep job) | ~-35 / -60 ms | low | sweep with pcc_vs_auto, bench |
| R3-4 | profile harness: the cut last layer as its own section (estimate within 3 %) | — | low | estimate vs measured |
| R3-5 | (stretch) fused QKV + head split + RoPE + QK-norm (after R3-2) | further ~-50 ms | high | micro-benchmark first |

Not retried: pre-norm drop, Q-window, trace (R2-5), sharded norms (R2-6).
