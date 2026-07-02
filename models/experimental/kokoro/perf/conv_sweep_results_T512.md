# TextEncoder Conv1d sweep — in=512 out=512 k=5 B=1 T=512 (M=512, K=2560, N=512)

- `conv_us` = dominant (conv2d) device op; `total_us` = sum of all device ops the conv emits (halo + conv2d + reshards).
- `cores` = cores the conv parallelized onto (output shard grid). `in_mem` = requested input shard_layout; `out_mem` = actual conv output layout.
- Program-config knobs: `abh` = act_block_h_override (0=auto), `split_reader`, `act_dbuf` = enable_act_double_buffer, `wt_dbuf` = enable_weights_double_buffer. Fixed: HiFi3, fp32_dest_acc_en=True, config_tensors_in_dram=True.
- `forced_grid` = explicit gx×gy pinned via override_sharding_config (blank = ttnn auto-selects). Rows above the auto core count exist to show the ceiling — they FATAL when the grid is infeasible (block_w<2).

| config | in_mem | out_mem | forced_grid | abh | split_reader | act_dbuf | wt_dbuf | cores | #ops | conv_us | total_us | PCC | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| block_abh64_sr0_adb1_wdb1 **best-conv** | block | block | - | 64 | False | True | True | 64 | 4 | 40.48 | 44.59 | 1.0000 | PASS |
| auto_abh0_sr0_adb1_wdb1 | auto | block | - | auto | False | True | True | 64 | 4 | 40.52 | 44.60 | 1.0000 | PASS |
| block_abh0_sr0_adb1_wdb1 | block | block | - | auto | False | True | True | 64 | 4 | 40.53 | 44.67 | 1.0000 | PASS |
| auto_abh0_sr1_adb1_wdb1 | auto | block | - | auto | True | True | True | 64 | 4 | 40.86 | 44.99 | 1.0000 | PASS |
| block_abh0_sr1_adb1_wdb1 | block | block | - | auto | True | True | True | 64 | 4 | 40.87 | 44.92 | 1.0000 | PASS |
| force_8x8=64c_block | block | block | 8x8 | auto | True | True | True | 64 | 4 | 40.90 | 44.91 | 1.0000 | PASS |
| block_abh64_sr1_adb1_wdb1 | block | block | - | 64 | True | True | True | 64 | 4 | 40.95 | 45.06 | 1.0000 | PASS |
| block_abh32_sr0_adb1_wdb1 | block | block | - | 32 | False | True | True | 64 | 4 | 41.95 | 45.94 | 1.0000 | PASS |
| block_abh32_sr1_adb1_wdb1 | block | block | - | 32 | True | True | True | 64 | 4 | 42.01 | 46.05 | 1.0000 | PASS |
| force_9x9=81c_block | block | block | 9x9 | auto | True | True | True | 72 | 4 | 45.71 | 50.07 | 1.0000 | PASS |
| auto_abh0_sr1_adb0_wdb1 | auto | block | - | auto | True | False | True | 64 | 4 | 46.41 | 50.44 | 1.0000 | PASS |
| block_abh0_sr1_adb0_wdb1 | block | block | - | auto | True | False | True | 64 | 4 | 46.43 | 50.46 | 1.0000 | PASS |
| block_abh64_sr1_adb0_wdb1 | block | block | - | 64 | True | False | True | 64 | 4 | 46.45 | 50.47 | 1.0000 | PASS |
| auto_abh0_sr0_adb0_wdb1 | auto | block | - | auto | False | False | True | 64 | 4 | 46.47 | 50.48 | 1.0000 | PASS |
| block_abh0_sr0_adb0_wdb1 | block | block | - | auto | False | False | True | 64 | 4 | 46.47 | 50.68 | 1.0000 | PASS |
| block_abh64_sr0_adb0_wdb1 | block | block | - | 64 | False | False | True | 64 | 4 | 46.51 | 50.55 | 1.0000 | PASS |
| block_abh32_sr0_adb0_wdb1 | block | block | - | 32 | False | False | True | 64 | 4 | 50.23 | 54.20 | 1.0000 | PASS |
| force_10x9=90c_block | block | block | 10x9 | auto | True | True | True | 72 | 4 | 50.30 | 54.82 | 1.0000 | PASS |
| force_10x10=100c_block | block | block | 10x10 | auto | True | True | True | 80 | 4 | 50.32 | 54.71 | 1.0000 | PASS |
| block_abh32_sr1_adb0_wdb1 | block | block | - | 32 | True | False | True | 64 | 4 | 50.33 | 54.39 | 1.0000 | PASS |
| force_10x8=80c_block | block | block | 10x8 | auto | True | True | True | 64 | 4 | 50.38 | 54.62 | 1.0000 | PASS |
| auto_abh0_sr1_adb0_wdb0 **~baseline** | auto | block | - | auto | True | False | False | 64 | 4 | 50.82 | 54.86 | 1.0000 | PASS |
| block_abh64_sr1_adb1_wdb0 | block | block | - | 64 | True | True | False | 64 | 4 | 50.82 | 54.85 | 1.0000 | PASS |
| auto_abh0_sr1_adb1_wdb0 | auto | block | - | auto | True | True | False | 64 | 4 | 50.89 | 54.95 | 1.0000 | PASS |
| block_abh0_sr1_adb0_wdb0 | block | block | - | auto | True | False | False | 64 | 4 | 50.90 | 54.80 | 1.0000 | PASS |
| block_abh64_sr1_adb0_wdb0 | block | block | - | 64 | True | False | False | 64 | 4 | 50.90 | 55.00 | 1.0000 | PASS |
| block_abh0_sr1_adb1_wdb0 | block | block | - | auto | True | True | False | 64 | 4 | 50.91 | 54.93 | 1.0000 | PASS |
| block_abh0_sr0_adb1_wdb0 | block | block | - | auto | False | True | False | 64 | 4 | 51.06 | 55.21 | 1.0000 | PASS |
| auto_abh0_sr0_adb0_wdb0 | auto | block | - | auto | False | False | False | 64 | 4 | 51.34 | 55.35 | 1.0000 | PASS |
| auto_abh0_sr0_adb1_wdb0 | auto | block | - | auto | False | True | False | 64 | 4 | 51.35 | 55.36 | 1.0000 | PASS |
| block_abh0_sr0_adb0_wdb0 | block | block | - | auto | False | False | False | 64 | 4 | 51.36 | 55.50 | 1.0000 | PASS |
| block_abh64_sr0_adb1_wdb0 | block | block | - | 64 | False | True | False | 64 | 4 | 51.52 | 55.61 | 1.0000 | PASS |
| block_abh64_sr0_adb0_wdb0 | block | block | - | 64 | False | False | False | 64 | 4 | 51.81 | 55.83 | 1.0000 | PASS |
| force_11x10=110c_block | block | block | 11x10 | auto | True | True | True | 80 | 4 | 54.91 | 59.22 | 1.0000 | PASS |
| force_11x9=99c_block | block | block | 11x9 | auto | True | True | True | 72 | 4 | 55.00 | 59.40 | 1.0000 | PASS |
| force_11x8=88c_block | block | block | 11x8 | auto | True | True | True | 64 | 4 | 55.03 | 59.27 | 1.0000 | PASS |
| force_8x7=56c_block | block | block | 8x7 | auto | True | True | True | 56 | 4 | 58.35 | 62.84 | 1.0000 | PASS |
| force_12x10=120c_block | block | block | 12x10 | auto | True | True | True | 80 | 4 | 59.62 | 64.11 | 1.0000 | PASS |
| force_12x9=108c_block | block | block | 12x9 | auto | True | True | True | 72 | 4 | 59.68 | 64.22 | 1.0000 | PASS |
| force_12x8=96c_block | block | block | 12x8 | auto | True | True | True | 64 | 4 | 59.69 | 63.92 | 1.0000 | PASS |
| auto_abh32_sr0_adb1_wdb0 | auto | block | - | 32 | False | True | False | 64 | 4 | 62.63 | 66.56 | 1.0000 | PASS |
| block_abh32_sr1_adb1_wdb0 | block | block | - | 32 | True | True | False | 64 | 4 | 62.68 | 66.76 | 1.0000 | PASS |
| block_abh32_sr0_adb1_wdb0 | block | block | - | 32 | False | True | False | 64 | 4 | 62.88 | 66.99 | 1.0000 | PASS |
| block_abh32_sr1_adb0_wdb0 | block | block | - | 32 | True | False | False | 64 | 4 | 63.00 | 67.14 | 1.0000 | PASS |
| auto_abh32_sr0_adb0_wdb0 | auto | block | - | 32 | False | False | False | 64 | 4 | 63.24 | 67.31 | 1.0000 | PASS |
| auto_abh32_sr1_adb1_wdb0 | auto | block | - | 32 | True | True | False | 64 | 4 | 63.25 | 67.28 | 1.0000 | PASS |
| block_abh32_sr0_adb0_wdb0 | block | block | - | 32 | False | False | False | 64 | 4 | 63.30 | 67.36 | 1.0000 | PASS |
| auto_abh32_sr1_adb0_wdb0 | auto | block | - | 32 | True | False | False | 64 | 4 | 63.43 | 67.58 | 1.0000 | PASS |
| force_13x10=130c_block | block | block | 13x10 | auto | True | True | True | 80 | 4 | 64.34 | 68.92 | 1.0000 | PASS |
| force_13x8=104c_block | block | block | 13x8 | auto | True | True | True | 64 | 4 | 64.36 | 68.61 | 1.0000 | PASS |
| force_13x9=117c_block | block | block | 13x9 | auto | True | True | True | 72 | 4 | 64.38 | 68.87 | 1.0000 | PASS |
| force_9x7=63c_block | block | block | 9x7 | auto | True | True | True | 56 | 4 | 65.24 | 69.87 | 1.0000 | PASS |
| force_9x6=54c_block | block | block | 9x6 | auto | True | True | True | 48 | 4 | 65.31 | 69.87 | 1.0000 | PASS |
| force_10x7=70c_block | block | block | 10x7 | auto | True | True | True | 56 | 4 | 72.16 | 76.77 | 1.0000 | PASS |
| force_7x6=42c_block | block | block | 7x6 | auto | True | True | True | 48 | 4 | 75.01 | 79.55 | 1.0000 | PASS |
| force_7x7=49c_block | block | block | 7x7 | auto | True | True | True | 56 | 4 | 75.06 | 79.65 | 1.0000 | PASS |
| force_8x4=32c_block | block | block | 8x4 | auto | True | True | True | 32 | 4 | 75.98 | 80.20 | 1.0000 | PASS |
| force_11x7=77c_block | block | block | 11x7 | auto | True | True | True | 56 | 4 | 78.97 | 83.67 | 1.0000 | PASS |
| force_11x6=66c_block | block | block | 11x6 | auto | True | True | True | 48 | 4 | 79.03 | 83.64 | 1.0000 | PASS |
| force_9x5=45c_block | block | block | 9x5 | auto | True | True | True | 40 | 4 | 85.00 | 89.39 | 1.0000 | PASS |
| force_12x6=72c_block | block | block | 12x6 | auto | True | True | True | 48 | 4 | 85.82 | 90.66 | 1.0000 | PASS |
| force_12x7=84c_block | block | block | 12x7 | auto | True | True | True | 56 | 4 | 85.94 | 90.62 | 1.0000 | PASS |
| force_13x6=78c_block | block | block | 13x6 | auto | True | True | True | 48 | 4 | 92.73 | 97.47 | 1.0000 | PASS |
| force_13x7=91c_block | block | block | 13x7 | auto | True | True | True | 56 | 4 | 92.80 | 97.53 | 1.0000 | PASS |
| force_10x5=50c_block | block | block | 10x5 | auto | True | True | True | 40 | 4 | 93.97 | 98.29 | 1.0000 | PASS |
| force_10x4=40c_block | block | block | 10x4 | auto | True | True | True | 32 | 4 | 94.00 | 98.31 | 1.0000 | PASS |
| force_5x5=25c_block | block | block | 5x5 | auto | True | True | True | 40 | 4 | 94.03 | 98.70 | 1.0000 | PASS |
| force_7x4=28c_block | block | block | 7x4 | auto | True | True | True | 32 | 4 | 98.08 | 102.78 | 1.0000 | PASS |
| force_7x5=35c_block | block | block | 7x5 | auto | True | True | True | 40 | 4 | 98.10 | 102.86 | 1.0000 | PASS |
| force_11x4=44c_block | block | block | 11x4 | auto | True | True | True | 32 | 4 | 102.98 | 107.31 | 1.0000 | PASS |
| force_11x5=55c_block | block | block | 11x5 | auto | True | True | True | 40 | 4 | 103.05 | 107.35 | 1.0000 | PASS |
| force_12x4=48c_block | block | block | 12x4 | auto | True | True | True | 32 | 4 | 111.93 | 116.29 | 1.0000 | PASS |
| force_12x5=60c_block | block | block | 12x5 | auto | True | True | True | 40 | 4 | 112.06 | 116.47 | 1.0000 | PASS |
| force_13x4=52c_block | block | block | 13x4 | auto | True | True | True | 32 | 4 | 121.02 | 125.47 | 1.0000 | PASS |
| force_13x5=65c_block | block | block | 13x5 | auto | True | True | True | 40 | 4 | 121.03 | 125.48 | 1.0000 | PASS |
| force_9x3=27c_block | block | block | 9x3 | auto | True | True | True | 24 | 4 | 126.45 | 131.76 | 1.0000 | PASS |
| force_5x3=15c_block | block | block | 5x3 | auto | True | True | True | 24 | 4 | 139.57 | 145.51 | 1.0000 | PASS |
| force_10x3=30c_block | block | block | 10x3 | auto | True | True | True | 24 | 4 | 139.85 | 145.10 | 1.0000 | PASS |
| force_7x3=21c_block | block | block | 7x3 | auto | True | True | True | 24 | 4 | 146.03 | 151.25 | 1.0000 | PASS |
| force_8x2=16c_block | block | block | 8x2 | auto | True | True | True | 16 | 4 | 149.81 | 154.66 | 1.0000 | PASS |
| force_11x3=33c_block | block | block | 11x3 | auto | True | True | True | 24 | 4 | 153.29 | 158.63 | 1.0000 | PASS |
| force_12x3=36c_block | block | block | 12x3 | auto | True | True | True | 24 | 4 | 166.78 | 172.02 | 1.0000 | PASS |
| force_9x2=18c_block | block | block | 9x2 | auto | True | True | True | 16 | 4 | 167.75 | 172.67 | 1.0000 | PASS |
| force_13x3=39c_block | block | block | 13x3 | auto | True | True | True | 24 | 4 | 180.34 | 185.68 | 1.0000 | PASS |
| force_10x2=20c_block | block | block | 10x2 | auto | True | True | True | 16 | 4 | 185.78 | 190.71 | 1.0000 | PASS |
| force_7x2=14c_block | block | block | 7x2 | auto | True | True | True | 16 | 4 | 194.02 | 199.56 | 1.0000 | PASS |
| width_abh0_sr1_adb1_wdb1 | width | width | - | auto | True | True | True | 16 | 4 | 201.56 | 207.61 | 1.0000 | PASS |
| force_11x2=22c_block | block | block | 11x2 | auto | True | True | True | 16 | 4 | 203.81 | 208.79 | 1.0000 | PASS |
| width_abh0_sr0_adb1_wdb1 | width | width | - | auto | False | True | True | 16 | 4 | 204.47 | 210.52 | 1.0000 | PASS |
| width_abh64_sr0_adb1_wdb1 | width | width | - | 64 | False | True | True | 16 | 4 | 209.16 | 215.23 | 1.0000 | PASS |
| auto_abh64_sr1_adb1_wdb1 | auto | width | - | 64 | True | True | True | 16 | 4 | 209.44 | 215.51 | 1.0000 | PASS |
| width_abh64_sr1_adb1_wdb1 | width | width | - | 64 | True | True | True | 16 | 4 | 209.49 | 215.55 | 1.0000 | PASS |
| auto_abh64_sr0_adb1_wdb1 | auto | width | - | 64 | False | True | True | 16 | 4 | 209.85 | 215.96 | 1.0000 | PASS |
| width_abh0_sr0_adb1_wdb0 | width | width | - | auto | False | True | False | 16 | 4 | 210.15 | 216.17 | 1.0000 | PASS |
| width_abh0_sr1_adb1_wdb0 | width | width | - | auto | True | True | False | 16 | 4 | 210.98 | 217.06 | 1.0000 | PASS |
| force_12x2=24c_block | block | block | 12x2 | auto | True | True | True | 16 | 4 | 221.76 | 226.74 | 1.0000 | PASS |
| force_13x2=26c_block | block | block | 13x2 | auto | True | True | True | 16 | 4 | 239.73 | 244.90 | 1.0000 | PASS |
| width_abh0_sr0_adb0_wdb1 | width | width | - | auto | False | False | True | 16 | 4 | 242.01 | 248.03 | 1.0000 | PASS |
| width_abh0_sr1_adb0_wdb0 | width | width | - | auto | True | False | False | 16 | 4 | 242.01 | 248.08 | 1.0000 | PASS |
| width_abh0_sr0_adb0_wdb0 | width | width | - | auto | False | False | False | 16 | 4 | 244.98 | 251.04 | 1.0000 | PASS |
| width_abh0_sr1_adb0_wdb1 | width | width | - | auto | True | False | True | 16 | 4 | 245.14 | 251.20 | 1.0000 | PASS |
| auto_abh64_sr1_adb1_wdb0 | auto | width | - | 64 | True | True | False | 16 | 4 | 263.47 | 269.57 | 1.0000 | PASS |
| width_abh64_sr1_adb1_wdb0 | width | width | - | 64 | True | True | False | 16 | 4 | 264.03 | 270.07 | 1.0000 | PASS |
| auto_abh64_sr0_adb1_wdb0 | auto | width | - | 64 | False | True | False | 16 | 4 | 265.30 | 271.36 | 1.0000 | PASS |
| width_abh64_sr0_adb1_wdb0 | width | width | - | 64 | False | True | False | 16 | 4 | 265.57 | 271.65 | 1.0000 | PASS |
| width_abh32_sr0_adb1_wdb1 | width | width | - | 32 | False | True | True | 16 | 4 | 277.36 | 283.36 | 1.0000 | PASS |
| width_abh32_sr1_adb1_wdb1 | width | width | - | 32 | True | True | True | 16 | 4 | 277.52 | 283.52 | 1.0000 | PASS |
| auto_abh32_sr0_adb1_wdb1 | auto | width | - | 32 | False | True | True | 16 | 4 | 277.57 | 283.61 | 1.0000 | PASS |
| auto_abh32_sr1_adb1_wdb1 | auto | width | - | 32 | True | True | True | 16 | 4 | 277.61 | 283.65 | 1.0000 | PASS |
| auto_abh64_sr1_adb0_wdb0 | auto | width | - | 64 | True | False | False | 16 | 4 | 295.66 | 301.71 | 1.0000 | PASS |
| auto_abh64_sr1_adb0_wdb1 | auto | width | - | 64 | True | False | True | 16 | 4 | 295.75 | 301.87 | 1.0000 | PASS |
| auto_abh64_sr0_adb0_wdb1 | auto | width | - | 64 | False | False | True | 16 | 4 | 295.80 | 301.86 | 1.0000 | PASS |
| width_abh64_sr1_adb0_wdb1 | width | width | - | 64 | True | False | True | 16 | 4 | 295.91 | 301.94 | 1.0000 | PASS |
| auto_abh64_sr0_adb0_wdb0 | auto | width | - | 64 | False | False | False | 16 | 4 | 295.92 | 301.98 | 1.0000 | PASS |
| width_abh64_sr1_adb0_wdb0 | width | width | - | 64 | True | False | False | 16 | 4 | 296.16 | 302.27 | 1.0000 | PASS |
| width_abh64_sr0_adb0_wdb1 | width | width | - | 64 | False | False | True | 16 | 4 | 296.29 | 302.32 | 1.0000 | PASS |
| width_abh64_sr0_adb0_wdb0 | width | width | - | 64 | False | False | False | 16 | 4 | 297.08 | 303.13 | 1.0000 | PASS |
| force_8x1=8c_block | block | block | 8x1 | auto | True | True | True | 8 | 4 | 297.82 | 303.85 | 1.0000 | PASS |
| force_9x1=9c_block | block | block | 9x1 | auto | True | True | True | 8 | 4 | 333.64 | 339.79 | 1.0000 | PASS |
| width_abh32_sr1_adb1_wdb0 | width | width | - | 32 | True | True | False | 16 | 4 | 335.02 | 341.05 | 1.0000 | PASS |
| width_abh32_sr0_adb1_wdb0 | width | width | - | 32 | False | True | False | 16 | 4 | 335.09 | 341.06 | 1.0000 | PASS |
| force_10x1=10c_block | block | block | 10x1 | auto | True | True | True | 8 | 4 | 369.64 | 375.72 | 1.0000 | PASS |
| width_abh32_sr0_adb0_wdb1 | width | width | - | 32 | False | False | True | 16 | 4 | 377.47 | 383.52 | 1.0000 | PASS |
| auto_abh32_sr1_adb0_wdb1 | auto | width | - | 32 | True | False | True | 16 | 4 | 377.82 | 383.85 | 1.0000 | PASS |
| width_abh32_sr1_adb0_wdb1 | width | width | - | 32 | True | False | True | 16 | 4 | 377.87 | 383.90 | 1.0000 | PASS |
| auto_abh32_sr0_adb0_wdb1 | auto | width | - | 32 | False | False | True | 16 | 4 | 377.87 | 383.94 | 1.0000 | PASS |
| width_abh32_sr0_adb0_wdb0 | width | width | - | 32 | False | False | False | 16 | 4 | 382.36 | 388.35 | 1.0000 | PASS |
| width_abh32_sr1_adb0_wdb0 | width | width | - | 32 | True | False | False | 16 | 4 | 382.50 | 388.53 | 1.0000 | PASS |
| force_11x1=11c_block | block | block | 11x1 | auto | True | True | True | 8 | 4 | 405.54 | 411.64 | 1.0000 | PASS |
| force_12x1=12c_block | block | block | 12x1 | auto | True | True | True | 8 | 4 | 441.57 | 447.74 | 1.0000 | PASS |
| force_13x1=13c_block | block | block | 13x1 | auto | True | True | True | 8 | 4 | 477.55 | 483.78 | 1.0000 | PASS |
| height_abh0_sr1_adb1_wdb0 | height | - | - | auto | True | True | False | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh0_sr1_adb1_wdb1 | height | - | - | auto | True | True | True | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh0_sr1_adb0_wdb0 | height | - | - | auto | True | False | False | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh0_sr1_adb0_wdb1 | height | - | - | auto | True | False | True | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh0_sr0_adb1_wdb0 | height | - | - | auto | False | True | False | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh0_sr0_adb1_wdb1 | height | - | - | auto | False | True | True | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh0_sr0_adb0_wdb0 | height | - | - | auto | False | False | False | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh0_sr0_adb0_wdb1 | height | - | - | auto | False | False | True | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh32_sr1_adb1_wdb0 | height | - | - | 32 | True | True | False | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh32_sr1_adb1_wdb1 | height | - | - | 32 | True | True | True | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh32_sr1_adb0_wdb0 | height | - | - | 32 | True | False | False | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh32_sr1_adb0_wdb1 | height | - | - | 32 | True | False | True | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh32_sr0_adb1_wdb0 | height | - | - | 32 | False | True | False | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh32_sr0_adb1_wdb1 | height | - | - | 32 | False | True | True | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh32_sr0_adb0_wdb0 | height | - | - | 32 | False | False | False | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh32_sr0_adb0_wdb1 | height | - | - | 32 | False | False | True | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh64_sr1_adb1_wdb0 | height | - | - | 64 | True | True | False | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh64_sr1_adb1_wdb1 | height | - | - | 64 | True | True | True | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh64_sr1_adb0_wdb0 | height | - | - | 64 | True | False | False | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh64_sr1_adb0_wdb1 | height | - | - | 64 | True | False | True | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh64_sr0_adb1_wdb0 | height | - | - | 64 | False | True | False | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh64_sr0_adb1_wdb1 | height | - | - | 64 | False | True | True | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh64_sr0_adb0_wdb0 | height | - | - | 64 | False | False | False | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| height_abh64_sr0_adb0_wdb1 | height | - | - | 64 | False | False | True | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| force_2x1=2c_block | block | - | 2x1 | auto | True | True | True | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| force_3x1=3c_block | block | - | 3x1 | auto | True | True | True | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| force_4x1=4c_block | block | - | 4x1 | auto | True | True | True | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| force_5x1=5c_block | block | - | 5x1 | auto | True | True | True | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| force_6x1=6c_block | block | - | 6x1 | auto | True | True | True | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
| force_7x1=7c_block | block | - | 7x1 | auto | True | True | True | - | - | - | - | - | ERROR: TT_THROW @ /home/ubuntu/samyuktha/tt-metal/tt_metal/impl/program/program.cpp:147 |
