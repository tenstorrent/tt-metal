# Round 2 -> 3 analysis (iter_02; analyst: Opus, low effort, read-only; jobs 746 bench+profile, 747 multivariate profile)

Baseline 23a61275446: paper/accuracy device 501.6 ms (wall 686.7), long 150.5, multivariate 935.6.

## Paper, one layer (65536 tokens, 15 ops, 25.91 ms)
| op | role | ms | util |
|---|---|---|---|
| 35 | seq wqkv minimal_matmul 1280->4608 | 4.72 | 54 % of HiFi2 peak; DRAM 166 GB/s |
| 36 | NlpCreateHeads | 3.09 | ~390 GB/s |
| 43/45/48 | post_add_norm generic_op (x3) | 4.83 | ~420 GB/s (82 %) |
| 37/38 | RoPE q, k | 3.07 | ~260 GB/s (half of peak) |
| 41 | SDPA causal | 1.90 | |
| 42 | seq wo | 1.49 | 57 % |
| 47/46 | FFN ff1 / ff0 | 1.42 / 1.30 | 50 / 54 % |
| 44 | var w_ov (V = 1) | 1.39 | 51 % |
| 39/40 | QK rms_norm | 1.92 | ~420 GB/s |
| 34 | pre-norm (first layer only) | 0.79 | ~420 GB/s |

Matmuls 10.3 ms (40 %) at 50-57 %; everything else is memory-bound near peak, so only fusion that removes DRAM round
trips helps. Op-to-op gaps ~9 us/layer; two ~0.3/0.7 ms gaps after section signposts are profiler artifacts.

## Estimate (+4.4 %)
The 20x layer count ignores the R1-5 cut: layer 20 runs the full QKV/heads/RoPE/QK-norm (~13.6 ms) and the rest only
on read-out slots (~1.5 ms). Corrected: 19 x 25.91 + 3.8 + 13.6 + 1.5 + 0.2 ~= 511 ms (+1.9 %). Fix for round 3:
profile the cut layer as its own section.

## Multivariate (48.45 ms/layer, 23 ops)
The profiled input took the unpacked path (per-variate padding -> var_mask present), unlike bench job 730 where
packing cut 2821 -> 1014 ms. On that path: two permutes 10.9 ms/layer, var QKV 4.67, heads 3.07, QK norms 1.91,
SDPA 1.99, wo 1.50.

## Candidates (analyst ranking)
A. Packed variate path with the var_mask folded into the block-diagonal mask (per-variate padding): 0 / ~200 ms
   on padded multivariate input; medium.
B. Fused QKV + head split + RoPE + QK-norm kernel: ~115 / ~200 ms; high.
C. One generic_op over q/k doing RoPE + QK-norm (post_add_norm infrastructure): ~60 / ~80 ms; medium-low.
D. Re-tune minimal_matmul blocks for the N = 1280 roles (wo, w_ov, ff0, ff1): ~35 / ~60 ms; low.
E. SDPA on strided non-head-split input: high, op support unclear.
F. Embed residual add into the matmul: ~1 ms; below the bar.
