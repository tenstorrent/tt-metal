# Round 2 proposals (provisional: no ca10312 profile yet, D7)

The round-2 profile (D6 condition 2) is blocked on disk (28 GB free, watermark 30). Under D7 ruling 3 these estimates
use the round-0 profile (job 632, `iter_00/`) for ops round 1 did not touch, and the round-1 bench (job 797,
`iter_01/bench_r1.json`) as the baseline. No candidate is accepted (no A entry) until the profile has explained R1-4/R1-6.

## Round-1 end table (provisional: op-by-op attribution pending profile)

| | paper accuracy ms | paper performance ms | vs round 0 (accuracy) |
|---|---|---|---|
| round 0 (job 684) | 8204 | 8108 | - |
| R1-3 fused SwiGLU (job 744) | 6212.9 | 6099.7 | -24.3 % |
| R1-4 matmul configs + R1-6 SDPA concat heads (job 797, ca10312c5a2) | 4466.5 | 4157.7 | -45.6 % |

Round 1 gained well over 5 %, so the loop continues (skill section 5 item 6).

## Baseline for round 2: 4466.5 ms (paper, accuracy, 2.5B-FT)

Untouched ops in round 1, full-model ms from job 632 (sections x layer_counts): RoPE q + k 406.6, create heads 270.4,
residual add 251, residual mul beta 209, RMSNorm 179.3, SDPA 204.0. Together 1520 ms, about 34 % of the round-2 baseline.

| # | candidate | est. saving (accuracy) | risk | screen |
|---|---|---|---|---|
| R2-1 | exact epsilon rescale: carry y = x / c, eps / c^2 per norm, fold 1/c into W_o / b_o / W_down (float64): removes the beta multiplies | -209 ms (-4.7 %) | low; exact in real arithmetic, a bf16 rounding change on y | folded dataflow within 1e-5 of the reference (CPU), then 313m BENCH_LAYERS=4 |
| R2-2 | residual add in the out_proj / down epilogue (`dit_minimal_matmul_addcmul_fused`, both already on `minimal_matmul`); needs R2-1 | -251 + ~28 = -223 ms (-5.0 %) | medium: epilogue precision; performance keeps bf16 wo / down (skill section 2) | bundled with R2-1 on 313m, 4 layers |
| R2-3 | QKV split + head split + RoPE fused (`minimal_matmul_split`, Timestral R3-1) | create heads + RoPE 677 -> ~340: -0.34 s (-7.6 %) | medium: off-label DiT op, separate q / k xPos tables | micro-benchmark at 2.5B layer shape, then 313m 4 layers |
| R2-4 | last-layer read-out cut before out_proj (R1-7) | -17 ms (-0.4 %) | low | bundle with R2-3 or R2-5 |
| R2-5 | split context / decoder streams (R1-1): removes the 31 % row padding and dead token 65 | about -0.9 to -1.3 s (-20 to -29 %) on the round-1 layer | high: micro-benchmark of the decoder attention first (<= 5 ms per layer, error <= the single-stream path) | micro-benchmark job, then path tests |
| R2-6 | host scaler per-patch sums (R1-8) | wall only, about -60 ms | low | wall in the same job (N2) |

Order: R2-1 + R2-2 (one bundle, one screening job), R2-3 micro-benchmark and R2-5 decoder-attention micro-benchmark
in parallel (ttq allows two jobs), then whichever passes. Each kept candidate: full suite + bench on its commit, then
`check_gates.py` against iter_00 (accuracy) and iter_01/bench_d4 (performance); the D6 named 313m cases have 0.004
budget left, so a trip there is an X with a variance table.

G2-r2: pre-approved categories (fusions, layout, host); the A entry waits for the profile (D7 ruling 3).
