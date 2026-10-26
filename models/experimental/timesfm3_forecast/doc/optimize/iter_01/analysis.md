# Round 2 analysis (iter_01, after round 1: paper/accuracy 606.9 ms device, 805.2 ms steady wall)

Source: read-only analyst (Opus, low) over `perf_report.txt`, `ops_perf_results.csv`, `multivariate/` (jobs 716, 717).
Reduced-layer estimate 606.7 ms vs measured 606.9 ms (-0.0 %). Peaks 359 TFLOPs, 512 GB/s.

## Paper layer (ids 39-58, 30.06 ms x 20)
| # | id | op / role | ms | utilization |
|---|---|---|---|---|
| 1 | 40 | seq QKV 2D matmul 65536x1280x4608 | 5.05 | 50 % FLOPs (ideal 2.15) |
| 2 | 41 | split_query_key_value_and_split_heads | 3.09 | ~390 GB/s (76 %) |
| 3 | 42, 43 | rotary_embedding_llama q, k | 2.15 + 2.15 | ~187 GB/s (37 %), the least efficient |
| 4 | 46 | SDPA causal, concat heads | 1.90 | |
| 5 | 47 | seq wo minimal_matmul | 1.48 | 57 % |
| 6 | 51 / 56 / 55 | W_ov / ff1 / ff0+relu | 1.39 / 1.38 / 1.31 | 51-54 % |
| 7 | 49, 53, 58 | residual adds (`_post`) | 1.11-1.12 each | ~450 GB/s (88 %) |
| 8 | 44, 45 | QK rms_norm | 0.95 each | ~83 % |
| 9 | 48, 52, 57 | post rms_norm | 0.83-0.84 each | |
| 10 | 39, 50, 54 | pre rms_norm (gain-free) | 0.80 each | ~82 % |

Categories per layer: norms 6.8 ms (8 ops, all > 80 % of DRAM), adds 3.35 ms, matmuls ~11.9 ms.
The post-norm + add + next pre-norm chain (2.76 ms, x3 per layer) is bandwidth-bound on three separate passes; one fused pass would read o and the residual once (~1.5 ms).

## Gaps
In-layer op-to-op gaps 0-1 us. Two ~335 us gaps at window boundaries. Host: 805 ms wall vs 607 ms device (~200 ms host).

## Multivariate (V = 8) layer: 140.7 ms (paper 30.1)
Variate attention ~110 ms/layer, ~100 ms of it tile inflation (8-row blocks padded to 32) and permutes:
permute qkv (id62) 33.2, reshape to (S*R/8, 8, 4608) (id63) 21.8, fill_implicit_tile_padding (id64) 12.6, create heads (id65) 12.3, SDPA (id68) 7.4, permute back (id70) 5.7, reshape back (id69) 5.1, var QKV matmul (id61) 5.06, QK norms 3.70 + 3.65 (4x inflated).
