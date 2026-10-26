# Round 2 proposals (G2-r2: automatic per skill section 7; no exception)

Baseline: 3804036ae03, paper/accuracy 606.9 ms (iter_01/bench.json, job 716). Drift reference stays r1_1 (D4).
Order of work: cheap and measured first, then the kernel.

| id | change | expected gain (paper / multivariate) | risk | measurement |
|---|---|---|---|---|
| R2-1 | RoPE on the (R x 8, 2, S, 96) leading-dim view (R1-6 micro: 2.56 vs 3.78 ms/call, bitwise identical), bundled with R1-5 (`cut_last_layer = True`, -1.5 % measured) | -30..-48 ms + -9 ms / -100+ ms | low | full bench, suite; >= 2 % as a declared bundle |
| R2-2 | R1-8: variate attention packs 4 items x 8 variates per 32-row tile with a block-diagonal 32x32 mask (merged with var_mask); permute xn (1280) before wqkv and o (1280) after wo instead of 4608 / 1536-wide tensors | 0 / -1.2..-1.6 s | medium (mask, SDPA mask broadcast) | D2's guard-only rule: multivariate >= 10 % better, target within +-1 %, other guards <= 3 %, floors hold; V = 8 and V = 32 module tests |
| R2-3 | QKV matmul re-tune (id40 at 50 %): minimal_matmul at N = 4608 vs 2D with larger subblocks, one sweep job | -10..-16 ms | low | sweep with pcc_vs_auto, then bench |
| R2-4 | model-local `generic_op` kernel: fused post RMSNorm(o)*gain + residual add + next gain-free pre-norm, writing both the residual and xn | ~-70 ms (-12 %) on both | high (new kernel, HiFi4 norm parity) | micro-benchmark first vs the 3-op chain (<= 1.8 ms at 65536x1280, PCC >= 0.99999 vs the chain), then suite and bench |
| R2-5 | trace capture of the forward (wall only) | device ~0, wall -150..-200 ms | low-medium | fresh-input trace test (PLAN), steady wall in bench.json |
| R2-6 | sharded rms_norm program config (larger blocks) | <= -10..-14 ms | low-medium (reshard cost) | single-norm micro-benchmark |

Not retried: dropping pre-norms (R1-2, D4), Q-window projection (R1-5b). Combined QK-norm (~-4 ms) left out as below the bar.
