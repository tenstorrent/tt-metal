# Round 4 proposals (G2-r4: automatic per skill section 7; no exception)

Baseline 8da76a2d9cb, paper/accuracy 423.6 ms (iter_03/bench.json, job 771). Drift reference stays r1_1 (D4).

| id | change | expected (paper / multivariate) | risk | measurement |
|---|---|---|---|---|
| R4-1 | rope_qk_norm_qkv bandwidth: v straight reader -> writer without a per-unit flush, batched head-row reads, deeper CBs (293 -> ~410 GB/s) | ~-24 / -24 ms | medium | micro (R = 1024, S = 64; now 4.12 ms, target ~2.9), bitwise vs current, suite, bench |
| R4-2 | variate-attention QK norm from QKV (the R3-5 kernel without RoPE) replacing NlpCreateHeads + 2 norms | 0 / ~-45 ms | low-medium | op test vs chain + control, multivariate bench |
| R4-3 | generic_op row permute for the variate (S, R) <-> (R, S) transposes | 0 / ~-170 ms | medium (sub-tile rows) | micro vs ttnn.permute (bitwise), multivariate bench |
| R4-4 | cut layer / head data movement: slot queries written by the rope kernel, tile-row gather for take_slots (bundle with R4-1) | ~-4 / -4 ms | low | cut vs uncut head PCC 1.000000, cut section time |
| R4-5 | matmul output block / subblock and 2D-multicast sweep for QKV, wo, 1280x1280 (new angle vs rejected R3-3 K-only) | ~-10..19 / x2 ms (guess) | medium (pcc_row_min on inactive rows) | one sweep job, bench incl. pcc_row_min |
| R4-6 | profile accounting: count the layer-0 pre-norm once (estimate fix) | — | low | estimate vs measured |

Not retried: pre-norm drop, Q-window, trace, sharded norms, K-only retune (R3-3).
