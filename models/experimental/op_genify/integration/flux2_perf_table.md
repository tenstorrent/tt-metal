# FLUX.2-dev: stock matmuls vs. genify_matmul (QB2, 2x2 mesh, per chip, mean of 4 chips)

M: img 2048, txt 256, joint 2304 rows per chip. Profiled runs averaged: {'stock DiT': 2, 'genify DiT': 2, 'stock TE': 2, 'genify TE': 2}.

Inputs: stock DiT generated/profiler/reports/2026_11_17_21_01_23/ops_perf_results_2026_11_17_21_01_23.csv,generated/profiler/reports/2026_11_17_21_09_40/ops_perf_results_2026_11_17_21_09_40.csv; genify DiT generated/profiler/reports/2026_11_17_21_03_54/ops_perf_results_2026_11_17_21_03_54.csv,generated/profiler/reports/2026_11_17_21_11_21/ops_perf_results_2026_11_17_21_11_21.csv; stock TE generated/profiler/reports/2026_11_17_21_05_56/ops_perf_results_2026_11_17_21_05_56.csv,generated/profiler/reports/2026_11_17_21_12_15/ops_perf_results_2026_11_17_21_12_15.csv; genify TE generated/profiler/reports/2026_11_17_21_07_12/ops_perf_results_2026_11_17_21_07_12.csv,generated/profiler/reports/2026_11_17_21_13_05/ops_perf_results_2026_11_17_21_13_05.csv

## Per denoising step

| matmul | calls | M | K | N | stock_op | stock_us | genify_us | ratio | stock_cores | genify_cores |
|---|---|---|---|---|---|---|---|---|---|---|
| Timestep MLP linear_1 | 1 | 1 | 256 | 6,144 | ttnn.linear 1D | 13.0 | 10.5 | 0.806 | 99 | 96 |
| Timestep MLP linear_2 | 1 | 1 | 6,144 | 6,144 | ttnn.linear 1D | 195.5 | 213.1 | 1.090 | 44 | 96 |
| Guidance MLP linear_1 | 1 | 1 | 256 | 6,144 | ttnn.linear 1D | 10.7 | 14.4 | 1.351 | 99 | 24 |
| Guidance MLP linear_2 | 1 | 1 | 6,144 | 6,144 | ttnn.linear 1D | 192.6 | 208.2 | 1.081 | 44 | 96 |
| time_mod (+bias, split x17) | 1 | 1 | 6,144 | 52,224 | minimal_matmul_split | 2,309.6 | 1,619.2 | 0.701 | 110 | 102 |
| x_embedder | 1 | 2,048 | 128 | 3,072 | minimal_matmul | 63.5 | 48.8 | 0.769 | 110 | 64 |
| Double QKV img (split x3) | 8 | 2,048 | 6,144 | 9,216 | minimal_matmul_split | 1,214.9 | 966.0 | 0.795 | 110 | 110 |
| Double QKV txt (split x3) | 8 | 256 | 6,144 | 9,216 | minimal_matmul_split | 453.1 | 327.6 | 0.723 | 110 | 96 |
| Double attn out img (+addcmul) | 8 | 2,048 | 6,144 | 3,072 | dit_minimal_matmul_addcmul_fused | 564.4 | 369.3 | 0.654 | 110 | 110 |
| Double attn out txt (+addcmul) | 8 | 256 | 6,144 | 3,072 | dit_minimal_matmul_addcmul_fused | 194.5 | 145.1 | 0.746 | 110 | 96 |
| Double FF1 img (+SwiGLU) | 8 | 2,048 | 6,144 | 18,432 | minimal_matmul fuse_swiglu | 2,393.5 | 1,848.2 | 0.772 | 110 | 110 |
| Double FF2 img | 8 | 2,048 | 9,216 | 6,144 | minimal_matmul | 1,256.6 | 977.4 | 0.778 | 110 | 110 |
| Double FF1 txt (+SwiGLU) | 8 | 256 | 6,144 | 18,432 | minimal_matmul fuse_swiglu | 862.8 | 635.6 | 0.737 | 110 | 96 |
| Double FF2 txt | 8 | 256 | 9,216 | 6,144 | minimal_matmul | 455.7 | 336.7 | 0.739 | 110 | 96 |
| Single QKV (split x3) | 47 | 2,304 | 6,144 | 9,216 | minimal_matmul_split | 1,247.5 | 1,116.6 | 0.895 | 110 | 110 |
| Single FF1 (+SwiGLU) | 47 | 2,304 | 6,144 | 18,432 | minimal_matmul fuse_swiglu | 2,398.1 | 2,157.3 | 0.900 | 110 | 110 |
| Single out (concat-K) | 47 | 2,304 | 12,288 | 6,144 | minimal_matmul [x0, x1] | 1,689.9 | 1,430.7 | 0.847 | 110 | 110 |
| Single QKV, final block (split x3) | 1 | 2,304 | 6,144 | 9,216 | minimal_matmul_split | 1,245.1 | 1,116.9 | 0.897 | 110 | 110 |
| Single FF1, final block (+SwiGLU) | 1 | 2,048 | 6,144 | 18,432 | minimal_matmul fuse_swiglu | 2,393.7 | 1,848.3 | 0.772 | 110 | 110 |
| Single out, final block (concat-K) | 1 | 2,048 | 12,288 | 6,144 | minimal_matmul [x0, x1] | 1,644.3 | 1,277.4 | 0.777 | 110 | 110 |
| proj_out | 1 | 2,048 | 3,072 | 128 | minimal_matmul | 60.9 | 58.4 | 0.958 | 110 | 44 |

## Per image (prompt encoding)

| matmul | calls | M | K | N | stock_op | stock_us | genify_us | ratio | stock_cores | genify_cores |
|---|---|---|---|---|---|---|---|---|---|---|
| context_embedder | 1 | 256 | 15,360 | 3,072 | minimal_matmul | 405.0 | 348.4 | 0.860 | 110 | 96 |
| TE qkv_proj | 40 | 512 | 5,120 | 3,072 | minimal_matmul | 139.5 | 115.1 | 0.825 | 110 | 88 |
| TE o_proj | 40 | 512 | 4,096 | 2,560 | minimal_matmul | 79.4 | 87.6 | 1.103 | 110 | 80 |
| TE gate | 40 | 512 | 5,120 | 16,384 | minimal_matmul | 501.3 | 509.0 | 1.015 | 110 | 88 |
| TE up (linear_in) | 40 | 512 | 5,120 | 16,384 | minimal_matmul | 500.2 | 508.5 | 1.017 | 110 | 88 |
| TE down (linear_out) | 40 | 512 | 16,384 | 5,120 | minimal_matmul | 508.7 | 532.2 | 1.046 | 110 | 88 |

## Totals (matmuls x calls)

| per | calls | stock ms | genify ms | genify / stock | glue ms (genify only) | genify + glue ms |
|---|---|---|---|---|---|---|
| denoising step | 215 | 318.06 | 272.38 | 0.856 | 25.65 | 298.02 |
| image (prompt encoding) | 201 | 69.57 | 70.44 | 1.013 | 0.00 | 70.44 |
| image incl. 50 steps | - | 15972.4 | 13689.2 | 0.857 | 1282.3 | 14971.6 |

## Whole-section device time (all ops, scaled)

| section | stock ms | genify ms | genify / stock | stock runs | genify runs |
|---|---|---|---|---|---|
| denoising step | 936.35 | 899.89 | 0.961 | 917.06, 955.64 | 902.51, 897.26 |
| denoising step without CCL ops | 370.44 | 350.56 | 0.946 | 370.22, 370.67 | 350.92, 350.20 |
| text encoder (40 layers) | 113.76 | 114.90 | 1.010 | 113.68, 113.85 | 114.92, 114.89 |
| text encoder without CCL ops | 99.18 | 100.43 | 1.013 | 99.10, 99.26 | 100.50, 100.36 |
| context_embedder | 0.405 | 0.348 | 0.860 | | |

## Glue ops (genify run only), per call

| for_matmul | op | in0 | count | us_each | us_sum | calls | ms_per_unit |
|---|---|---|---|---|---|---|---|
| time_mod (+bias, split x17) | SliceDeviceOperation | 1x52224 | 17 | 1.6 | 27.8 | 1 | 0.03 |
| Double QKV img (split x3) | SliceDeviceOperation | 2048x9216 | 3 | 72.8 | 218.5 | 8 | 1.75 |
| Double QKV txt (split x3) | SliceDeviceOperation | 256x9216 | 3 | 9.0 | 27.0 | 8 | 0.22 |
| Single QKV (split x3) | SliceDeviceOperation | 2304x9216 | 3 | 73.0 | 218.9 | 47 | 10.29 |
| Single out (concat-K) | ConcatDeviceOperation | 2304x3072 | 1 | 274.4 | 274.4 | 47 | 12.89 |
| Single QKV, final block (split x3) | SliceDeviceOperation | 2304x9216 | 3 | 72.8 | 218.5 | 1 | 0.22 |
| Single out, final block (concat-K) | ConcatDeviceOperation | 2048x3072 | 1 | 254.5 | 254.5 | 1 | 0.25 |

Ops only in the stock run: none
