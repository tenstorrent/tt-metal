# Round 0 analysis (paper/accuracy, 512 rows; profile job 1020, reduced 2+2, eager)

Written from the read-only analyst's report (Opus, low effort). Per-role sums reproduce the summary (matmul 175.8 ms reduced).

- **Measured:** 691.3 ms traced device time; the profile estimate is 690.5 ms (-0.1 %), and the reduced depth is 234.9 vs 235.1 ms. Op-to-op gaps (151 ms eager) are not in the traced number, so every win has to come from kernel time.
- **Top ops** (per instance, reduced): enc P@V 9.58 ms on 6 cores (0.14 % of peak); cross P@V 8.53 ms on 2 cores; enc scores 7.61 ms on 9 cores (HiFi4, fp32 out); RoPE q@R 7.58 ms on 6 cores (x2 per block, plus 3 in dec); cross K/V `_heads` reshape 5.05 ms (8x64 tile-padded to 32, 20 GB/s); dec cross scores 4.58 ms; dec self scores 4.52 ms on 1 core; dec P@V 3.94 ms. The large-M linears run at 31-58 TF/s (11-16 % of peak).
- **Full-depth time by role:** RoPE rotation matmuls 217 ms, attention P@V 132, scores 100, cross-attn reshape/transpose/slice 93, linears (+ embed, head) 69, eltwise ~54, softmax 9, create/concat heads 8, norms 5.
- **Why it is 40-70x off the napkin:** default batched matmuls over [512, 8, S, 64] land on 1-9 cores; RoPE runs as a dense HiFi4 matmul by a signed permutation; the cross-attn head split pads 8 heads into a 32-row tile; the fp32 [512, 8, 96, 96] scores are DRAM-bound; the linears have no program configs and relu runs as a separate op.
