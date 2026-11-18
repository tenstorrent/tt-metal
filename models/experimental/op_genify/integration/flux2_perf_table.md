# FLUX.2-dev: stock matmuls vs. genify_matmul (QB2, 2x2 mesh, per chip, mean of 4 chips)

M: img 2048, txt 256, joint 2304 rows per chip. Profiled runs averaged: {'stock DiT': 2, 'genify DiT': 2, 'stock TE': 2, 'genify TE': 2}.

Inputs: stock DiT generated/profiler/reports/2026_11_17_21_01_23/ops_perf_results_2026_11_17_21_01_23.csv,generated/profiler/reports/2026_11_17_21_09_40/ops_perf_results_2026_11_17_21_09_40.csv; genify DiT generated/profiler/reports/2026_11_18_11_39_12/ops_perf_results_2026_11_18_11_39_12.csv,generated/profiler/reports/2026_11_18_11_41_22/ops_perf_results_2026_11_18_11_41_22.csv; stock TE generated/profiler/reports/2026_11_17_21_05_56/ops_perf_results_2026_11_17_21_05_56.csv,generated/profiler/reports/2026_11_17_21_12_15/ops_perf_results_2026_11_17_21_12_15.csv; genify TE generated/profiler/reports/2026_11_18_11_40_08/ops_perf_results_2026_11_18_11_40_08.csv,generated/profiler/reports/2026_11_18_11_42_16/ops_perf_results_2026_11_18_11_42_16.csv

## Per denoising step

| matmul | calls | M | K | N | stock_op | stock_us | genify_us | ratio | stock_cores | genify_cores |
|---|---|---|---|---|---|---|---|---|---|---|
| Timestep MLP linear_1 | 1 | 1 | 256 | 6,144 | ttnn.linear 1D | 13.0 | 10.3 | 0.790 | 99 | 110 |
| Timestep MLP linear_2 | 1 | 1 | 6,144 | 6,144 | ttnn.linear 1D | 195.5 | 187.0 | 0.956 | 44 | 110 |
| Guidance MLP linear_1 | 1 | 1 | 256 | 6,144 | ttnn.linear 1D | 10.7 | 14.7 | 1.372 | 99 | 24 |
| Guidance MLP linear_2 | 1 | 1 | 6,144 | 6,144 | ttnn.linear 1D | 192.6 | 189.7 | 0.985 | 44 | 110 |
| time_mod (+bias, split x17) | 1 | 1 | 6,144 | 52,224 | minimal_matmul_split | 2,309.6 | 1,626.2 | 0.704 | 110 | 102 |
| x_embedder | 1 | 2,048 | 128 | 3,072 | minimal_matmul | 63.5 | 48.9 | 0.770 | 110 | 64 |
| Double QKV img (split x3) | 8 | 2,048 | 6,144 | 9,216 | minimal_matmul_split | 1,214.9 | 950.9 | 0.783 | 110 | 110 |
| Double QKV txt (split x3) | 8 | 256 | 6,144 | 9,216 | minimal_matmul_split | 453.1 | 330.7 | 0.730 | 110 | 96 |
| Double attn out img (+addcmul) | 8 | 2,048 | 6,144 | 3,072 | dit_minimal_matmul_addcmul_fused | 564.4 | 351.6 | 0.623 | 110 | 110 |
| Double attn out txt (+addcmul) | 8 | 256 | 6,144 | 3,072 | dit_minimal_matmul_addcmul_fused | 194.5 | 100.3 | 0.516 | 110 | 110 |
| Double FF1 img (+SwiGLU) | 8 | 2,048 | 6,144 | 18,432 | minimal_matmul fuse_swiglu | 2,393.5 | 1,848.7 | 0.772 | 110 | 110 |
| Double FF2 img | 8 | 2,048 | 9,216 | 6,144 | minimal_matmul | 1,256.6 | 955.2 | 0.760 | 110 | 110 |
| Double FF1 txt (+SwiGLU) | 8 | 256 | 6,144 | 18,432 | minimal_matmul fuse_swiglu | 862.8 | 532.5 | 0.617 | 110 | 110 |
| Double FF2 txt | 8 | 256 | 9,216 | 6,144 | minimal_matmul | 455.7 | 278.4 | 0.611 | 110 | 110 |
| Single QKV (split x3) | 47 | 2,304 | 6,144 | 9,216 | minimal_matmul_split | 1,247.5 | 1,103.7 | 0.885 | 110 | 110 |
| Single FF1 (+SwiGLU) | 47 | 2,304 | 6,144 | 18,432 | minimal_matmul fuse_swiglu | 2,398.1 | 2,157.0 | 0.899 | 110 | 110 |
| Single out (concat-K) | 47 | 2,304 | 2,304 | 9,216 | minimal_matmul [x0, x1] | 1,689.9 | 1,412.4 | 0.836 | 110 | 110 |
| Single QKV, final block (split x3) | 1 | 2,304 | 6,144 | 9,216 | minimal_matmul_split | 1,245.1 | 1,103.7 | 0.886 | 110 | 110 |
| Single FF1, final block (+SwiGLU) | 1 | 2,048 | 6,144 | 18,432 | minimal_matmul fuse_swiglu | 2,393.7 | 1,848.5 | 0.772 | 110 | 110 |
| Single out, final block (concat-K) | 1 | 2,048 | 2,048 | 9,216 | minimal_matmul [x0, x1] | 1,644.3 | 1,242.6 | 0.756 | 110 | 110 |
| proj_out | 1 | 2,048 | 3,072 | 128 | minimal_matmul | 60.9 | 58.2 | 0.956 | 110 | 44 |

## Per image (prompt encoding)

| matmul | calls | M | K | N | stock_op | stock_us | genify_us | ratio | stock_cores | genify_cores |
|---|---|---|---|---|---|---|---|---|---|---|
| context_embedder | 1 | 256 | 15,360 | 3,072 | minimal_matmul | 405.0 | 230.4 | 0.569 | 110 | 110 |
| TE qkv_proj | 40 | 512 | 5,120 | 3,072 | minimal_matmul | 139.5 | 93.6 | 0.671 | 110 | 88 |
| TE o_proj | 40 | 512 | 4,096 | 2,560 | minimal_matmul | 79.4 | 69.7 | 0.878 | 110 | 80 |
| TE gate | 40 | 512 | 5,120 | 16,384 | minimal_matmul | 501.3 | 358.9 | 0.716 | 110 | 110 |
| TE up (linear_in) | 40 | 512 | 5,120 | 16,384 | minimal_matmul | 500.2 | 358.5 | 0.717 | 110 | 110 |
| TE down (linear_out) | 40 | 512 | 16,384 | 5,120 | minimal_matmul | 508.7 | 430.0 | 0.845 | 110 | 88 |

## Totals (matmuls x calls)

| per | calls | stock ms | genify ms | genify / stock | glue ms (genify only) | genify + glue ms |
|---|---|---|---|---|---|---|
| denoising step | 215 | 318.06 | 268.75 | 0.845 | 0.00 | 268.75 |
| image (prompt encoding) | 201 | 69.57 | 52.66 | 0.757 | 0.00 | 52.66 |
| image incl. 50 steps | - | 15972.4 | 13490.1 | 0.845 | 0.0 | 13490.1 |

## Whole-section device time (all ops, scaled)

| section | stock ms | genify ms | genify / stock | stock runs | genify runs |
|---|---|---|---|---|---|
| denoising step | 936.35 | 869.92 | 0.929 | 917.06, 955.64 | 869.73, 870.12 |
| denoising step without CCL ops | 370.44 | 321.34 | 0.867 | 370.22, 370.67 | 321.47, 321.21 |
| text encoder (40 layers) | 113.76 | 96.89 | 0.852 | 113.68, 113.85 | 96.82, 96.95 |
| text encoder without CCL ops | 99.18 | 82.74 | 0.834 | 99.10, 99.26 | 82.68, 82.79 |
| context_embedder | 0.405 | 0.230 | 0.569 | | |

## Glue ops (genify run only), per call

Ops only in the stock run: none
