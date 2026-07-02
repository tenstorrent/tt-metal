# TextEncoder Conv1d sweep — in=512 out=512 k=5 B=2 T=48 (M=96, K=2560, N=512)

- `conv_us` = dominant (conv2d) device op; `total_us` = sum of all device ops the conv emits (halo + conv2d + reshards).
- `cores` = cores the conv parallelized onto (output shard grid). `in_mem` = requested input shard_layout; `out_mem` = actual conv output layout.
- Program-config knobs: `abh` = act_block_h_override (0=auto), `split_reader`, `act_dbuf` = enable_act_double_buffer, `wt_dbuf` = enable_weights_double_buffer. `fidelity` = math_fidelity (LoFi→HiFi4), `fp32acc` = fp32_dest_acc_en. Fixed: config_tensors_in_dram=True. Program-knob rows are pinned at HiFi3/fp32acc=True; the `fid_*` rows sweep fidelity on the production-best program config.
- `forced_grid` = explicit gx×gy pinned via override_sharding_config (blank = ttnn auto-selects). Rows above the auto core count exist to show the ceiling — they FATAL when the grid is infeasible (block_w<2).

| config | in_mem | out_mem | forced_grid | abh | split_reader | act_dbuf | wt_dbuf | fidelity | fp32acc | cores | #ops | conv_us | total_us | PCC | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fid_LoFi_fp32acc0_abh32 **best-conv** | block | interleaved | - | 32 | True | True | True | LoFi | False | 0 | 7 | 16.32 | 30.51 | 0.9997 | PASS |
| fid_LoFi_fp32acc1_abh0 | block | interleaved | - | auto | True | True | True | LoFi | True | 0 | 7 | 16.45 | 30.92 | 0.9999 | PASS |
| fid_LoFi_fp32acc0_abh0 | block | interleaved | - | auto | True | True | True | LoFi | False | 0 | 7 | 16.50 | 30.92 | 0.9997 | PASS |
| fid_LoFi_fp32acc1_abh32 | block | interleaved | - | 32 | True | True | True | LoFi | True | 0 | 7 | 16.51 | 30.75 | 0.9999 | PASS |
| fid_HiFi2_fp32acc1_abh0 | block | interleaved | - | auto | True | True | True | HiFi2 | True | 0 | 7 | 16.89 | 31.20 | 1.0000 | PASS |
| fid_HiFi2_fp32acc1_abh32 | block | interleaved | - | 32 | True | True | True | HiFi2 | True | 0 | 7 | 17.06 | 31.43 | 1.0000 | PASS |
| fid_HiFi2_fp32acc0_abh0 | block | interleaved | - | auto | True | True | True | HiFi2 | False | 0 | 7 | 17.09 | 31.52 | 0.9995 | PASS |
| fid_HiFi2_fp32acc0_abh32 | block | interleaved | - | 32 | True | True | True | HiFi2 | False | 0 | 7 | 17.11 | 31.30 | 0.9995 | PASS |
| fid_LoFi_fp32acc1_abh64 | block | interleaved | - | 64 | True | True | True | LoFi | True | 0 | 7 | 18.00 | 32.41 | 0.9999 | PASS |
| fid_LoFi_fp32acc0_abh64 | block | interleaved | - | 64 | True | True | True | LoFi | False | 0 | 7 | 18.35 | 32.66 | 0.9997 | PASS |
| fid_HiFi3_fp32acc1_abh32 | block | interleaved | - | 32 | True | True | True | HiFi3 | True | 0 | 7 | 21.65 | 36.00 | 1.0000 | PASS |
| block_abh32_sr0_adb1_wdb1 | block | interleaved | - | 32 | False | True | True | HiFi3 | True | 0 | 7 | 21.65 | 35.89 | 1.0000 | PASS |
| block_abh0_sr0_adb1_wdb1 | block | interleaved | - | auto | False | True | True | HiFi3 | True | 0 | 7 | 21.67 | 35.83 | 1.0000 | PASS |
| fid_HiFi3_fp32acc0_abh32 | block | interleaved | - | 32 | True | True | True | HiFi3 | False | 0 | 7 | 21.67 | 36.10 | 0.9995 | PASS |
| fid_HiFi3_fp32acc0_abh0 | block | interleaved | - | auto | True | True | True | HiFi3 | False | 0 | 7 | 21.68 | 36.01 | 0.9995 | PASS |
| block_abh32_sr1_adb1_wdb1 | block | interleaved | - | 32 | True | True | True | HiFi3 | True | 0 | 7 | 21.68 | 35.98 | 1.0000 | PASS |
| block_abh0_sr1_adb1_wdb1 | block | interleaved | - | auto | True | True | True | HiFi3 | True | 0 | 7 | 21.68 | 35.98 | 1.0000 | PASS |
| fid_HiFi3_fp32acc1_abh0 | block | interleaved | - | auto | True | True | True | HiFi3 | True | 0 | 7 | 21.79 | 36.21 | 1.0000 | PASS |
| force_9x3=27c_block | block | interleaved | 9x3 | auto | True | True | True | HiFi3 | True | 0 | 7 | 24.27 | 38.88 | 1.0000 | PASS |
| block_abh0_sr0_adb0_wdb1 | block | interleaved | - | auto | False | False | True | HiFi3 | True | 0 | 7 | 25.64 | 39.81 | 1.0000 | PASS |
| block_abh32_sr0_adb0_wdb1 | block | interleaved | - | 32 | False | False | True | HiFi3 | True | 0 | 7 | 25.65 | 39.83 | 1.0000 | PASS |
| block_abh32_sr1_adb0_wdb1 | block | interleaved | - | 32 | True | False | True | HiFi3 | True | 0 | 7 | 25.73 | 39.90 | 1.0000 | PASS |
| block_abh0_sr1_adb0_wdb1 | block | interleaved | - | auto | True | False | True | HiFi3 | True | 0 | 7 | 25.73 | 40.07 | 1.0000 | PASS |
| force_10x3=30c_block | block | interleaved | 10x3 | auto | True | True | True | HiFi3 | True | 0 | 7 | 26.67 | 41.02 | 1.0000 | PASS |
| force_5x3=15c_block | block | interleaved | 5x3 | auto | True | True | True | HiFi3 | True | 0 | 7 | 26.84 | 41.27 | 1.0000 | PASS |
| fid_HiFi4_fp32acc1_abh0 | block | interleaved | - | auto | True | True | True | HiFi4 | True | 0 | 7 | 27.36 | 41.75 | 1.0000 | PASS |
| fid_HiFi4_fp32acc0_abh32 | block | interleaved | - | 32 | True | True | True | HiFi4 | False | 0 | 7 | 27.36 | 41.73 | 0.9995 | PASS |
| fid_HiFi4_fp32acc1_abh32 | block | interleaved | - | 32 | True | True | True | HiFi4 | True | 0 | 7 | 27.36 | 41.63 | 1.0000 | PASS |
| fid_HiFi4_fp32acc0_abh0 | block | interleaved | - | auto | True | True | True | HiFi4 | False | 0 | 7 | 27.42 | 41.73 | 0.9995 | PASS |
| force_7x3=21c_block | block | interleaved | 7x3 | auto | True | True | True | HiFi3 | True | 0 | 7 | 27.64 | 42.15 | 1.0000 | PASS |
| force_11x3=33c_block | block | interleaved | 11x3 | auto | True | True | True | HiFi3 | True | 0 | 7 | 28.95 | 43.53 | 1.0000 | PASS |
| fid_HiFi2_fp32acc1_abh64 | block | interleaved | - | 64 | True | True | True | HiFi2 | True | 0 | 7 | 29.44 | 44.04 | 1.0000 | PASS |
| fid_HiFi2_fp32acc0_abh64 | block | interleaved | - | 64 | True | True | True | HiFi2 | False | 0 | 7 | 29.58 | 43.94 | 0.9995 | PASS |
| force_12x3=36c_block | block | interleaved | 12x3 | auto | True | True | True | HiFi3 | True | 0 | 7 | 31.45 | 45.94 | 1.0000 | PASS |
| auto_abh0_sr0_adb1_wdb0 | auto | interleaved | - | auto | False | True | False | HiFi3 | True | 0 | 7 | 32.33 | 46.64 | 1.0000 | PASS |
| block_abh0_sr0_adb1_wdb0 | block | interleaved | - | auto | False | True | False | HiFi3 | True | 0 | 7 | 32.36 | 46.56 | 1.0000 | PASS |
| block_abh32_sr1_adb1_wdb0 | block | interleaved | - | 32 | True | True | False | HiFi3 | True | 0 | 7 | 32.39 | 46.58 | 1.0000 | PASS |
| auto_abh0_sr1_adb0_wdb0 **~baseline** | auto | interleaved | - | auto | True | False | False | HiFi3 | True | 0 | 7 | 32.41 | 46.76 | 1.0000 | PASS |
| block_abh32_sr0_adb0_wdb0 | block | interleaved | - | 32 | False | False | False | HiFi3 | True | 0 | 7 | 32.42 | 46.52 | 1.0000 | PASS |
| block_abh0_sr1_adb0_wdb0 | block | interleaved | - | auto | True | False | False | HiFi3 | True | 0 | 7 | 32.44 | 46.80 | 1.0000 | PASS |
| block_abh32_sr0_adb1_wdb0 | block | interleaved | - | 32 | False | True | False | HiFi3 | True | 0 | 7 | 32.44 | 46.59 | 1.0000 | PASS |
| block_abh0_sr1_adb1_wdb0 | block | interleaved | - | auto | True | True | False | HiFi3 | True | 0 | 7 | 32.45 | 46.86 | 1.0000 | PASS |
| auto_abh0_sr0_adb0_wdb0 | auto | interleaved | - | auto | False | False | False | HiFi3 | True | 0 | 7 | 32.47 | 46.85 | 1.0000 | PASS |
| block_abh0_sr0_adb0_wdb0 | block | interleaved | - | auto | False | False | False | HiFi3 | True | 0 | 7 | 32.47 | 46.60 | 1.0000 | PASS |
| block_abh32_sr1_adb0_wdb0 | block | interleaved | - | 32 | True | False | False | HiFi3 | True | 0 | 7 | 32.47 | 46.62 | 1.0000 | PASS |
| auto_abh0_sr1_adb1_wdb0 | auto | interleaved | - | auto | True | True | False | HiFi3 | True | 0 | 7 | 32.48 | 46.96 | 1.0000 | PASS |
| force_13x3=39c_block | block | interleaved | 13x3 | auto | True | True | True | HiFi3 | True | 0 | 7 | 33.83 | 48.46 | 1.0000 | PASS |
| block_abh64_sr0_adb1_wdb1 | block | interleaved | - | 64 | False | True | True | HiFi3 | True | 0 | 7 | 40.48 | 54.85 | 1.0000 | PASS |
| fid_HiFi3_fp32acc0_abh64 | block | interleaved | - | 64 | True | True | True | HiFi3 | False | 0 | 7 | 40.83 | 55.25 | 0.9995 | PASS |
| fid_HiFi3_fp32acc1_abh64 | block | interleaved | - | 64 | True | True | True | HiFi3 | True | 0 | 7 | 40.88 | 55.41 | 1.0000 | PASS |
| force_8x2=16c_block | block | interleaved | 8x2 | auto | True | True | True | HiFi3 | True | 0 | 7 | 40.88 | 55.30 | 1.0000 | PASS |
| block_abh64_sr1_adb1_wdb1 | block | interleaved | - | 64 | True | True | True | HiFi3 | True | 0 | 7 | 40.90 | 55.24 | 1.0000 | PASS |
| width_abh0_sr0_adb1_wdb1 | width | interleaved | - | auto | False | True | True | HiFi3 | True | 0 | 7 | 41.80 | 56.02 | 1.0000 | PASS |
| width_abh0_sr1_adb1_wdb1 | width | interleaved | - | auto | True | True | True | HiFi3 | True | 0 | 7 | 41.81 | 56.20 | 1.0000 | PASS |
| auto_abh0_sr1_adb1_wdb1 | auto | interleaved | - | auto | True | True | True | HiFi3 | True | 0 | 7 | 41.84 | 56.22 | 1.0000 | PASS |
| auto_abh0_sr0_adb1_wdb1 | auto | interleaved | - | auto | False | True | True | HiFi3 | True | 0 | 7 | 41.86 | 56.17 | 1.0000 | PASS |
| force_8x2=16c_width | width | interleaved | 8x2 | auto | True | True | True | HiFi3 | True | 0 | 7 | 42.09 | 56.65 | 0.5472 | FAIL |
| force_9x2=18c_block | block | interleaved | 9x2 | auto | True | True | True | HiFi3 | True | 0 | 7 | 45.64 | 60.24 | 1.0000 | PASS |
| block_abh64_sr0_adb0_wdb1 | block | interleaved | - | 64 | False | False | True | HiFi3 | True | 0 | 7 | 46.37 | 60.69 | 1.0000 | PASS |
| block_abh64_sr1_adb0_wdb1 | block | interleaved | - | 64 | True | False | True | HiFi3 | True | 0 | 7 | 46.44 | 60.81 | 1.0000 | PASS |
| width_abh0_sr0_adb1_wdb0 | width | interleaved | - | auto | False | True | False | HiFi3 | True | 0 | 7 | 48.24 | 62.46 | 1.0000 | PASS |
| width_abh0_sr1_adb1_wdb0 | width | interleaved | - | auto | True | True | False | HiFi3 | True | 0 | 7 | 48.73 | 63.09 | 1.0000 | PASS |
| force_10x2=20c_block | block | interleaved | 10x2 | auto | True | True | True | HiFi3 | True | 0 | 7 | 50.42 | 65.10 | 1.0000 | PASS |
| block_abh64_sr1_adb1_wdb0 | block | interleaved | - | 64 | True | True | False | HiFi3 | True | 0 | 7 | 50.64 | 65.10 | 1.0000 | PASS |
| block_abh64_sr1_adb0_wdb0 | block | interleaved | - | 64 | True | False | False | HiFi3 | True | 0 | 7 | 50.93 | 65.35 | 1.0000 | PASS |
| block_abh64_sr0_adb1_wdb0 | block | interleaved | - | 64 | False | True | False | HiFi3 | True | 0 | 7 | 51.13 | 65.61 | 1.0000 | PASS |
| block_abh64_sr0_adb0_wdb0 | block | interleaved | - | 64 | False | False | False | HiFi3 | True | 0 | 7 | 51.28 | 65.62 | 1.0000 | PASS |
| force_7x2=14c_block | block | interleaved | 7x2 | auto | True | True | True | HiFi3 | True | 0 | 7 | 52.19 | 66.71 | 1.0000 | PASS |
| fid_HiFi4_fp32acc0_abh64 | block | interleaved | - | 64 | True | True | True | HiFi4 | False | 0 | 7 | 52.20 | 66.59 | 0.9995 | PASS |
| fid_HiFi4_fp32acc1_abh64 | block | interleaved | - | 64 | True | True | True | HiFi4 | True | 0 | 7 | 52.26 | 66.86 | 1.0000 | PASS |
| width_abh32_sr0_adb1_wdb1 | width | interleaved | - | 32 | False | True | True | HiFi3 | True | 0 | 7 | 53.20 | 67.78 | 1.0000 | PASS |
| auto_abh64_sr1_adb1_wdb1 | auto | interleaved | - | 64 | True | True | True | HiFi3 | True | 0 | 7 | 53.22 | 67.76 | 1.0000 | PASS |
| width_abh64_sr1_adb1_wdb1 | width | interleaved | - | 64 | True | True | True | HiFi3 | True | 0 | 7 | 53.24 | 67.65 | 1.0000 | PASS |
| auto_abh32_sr0_adb1_wdb1 | auto | interleaved | - | 32 | False | True | True | HiFi3 | True | 0 | 7 | 53.27 | 67.53 | 1.0000 | PASS |
| auto_abh32_sr1_adb1_wdb1 | auto | interleaved | - | 32 | True | True | True | HiFi3 | True | 0 | 7 | 53.27 | 67.76 | 1.0000 | PASS |
| auto_abh64_sr0_adb1_wdb1 | auto | interleaved | - | 64 | False | True | True | HiFi3 | True | 0 | 7 | 53.29 | 67.74 | 1.0000 | PASS |
| width_abh64_sr0_adb1_wdb1 | width | interleaved | - | 64 | False | True | True | HiFi3 | True | 0 | 7 | 53.29 | 67.64 | 1.0000 | PASS |
| width_abh32_sr1_adb1_wdb1 | width | interleaved | - | 32 | True | True | True | HiFi3 | True | 0 | 7 | 53.30 | 67.72 | 1.0000 | PASS |
| width_abh0_sr0_adb0_wdb0 | width | interleaved | - | auto | False | False | False | HiFi3 | True | 0 | 7 | 53.75 | 67.90 | 1.0000 | PASS |
| width_abh0_sr0_adb0_wdb1 | width | interleaved | - | auto | False | False | True | HiFi3 | True | 0 | 7 | 53.78 | 68.18 | 1.0000 | PASS |
| width_abh0_sr1_adb0_wdb1 | width | interleaved | - | auto | True | False | True | HiFi3 | True | 0 | 7 | 53.79 | 68.03 | 1.0000 | PASS |
| auto_abh0_sr0_adb0_wdb1 | auto | interleaved | - | auto | False | False | True | HiFi3 | True | 0 | 7 | 53.85 | 68.25 | 1.0000 | PASS |
| width_abh0_sr1_adb0_wdb0 | width | interleaved | - | auto | True | False | False | HiFi3 | True | 0 | 7 | 53.88 | 68.09 | 1.0000 | PASS |
| auto_abh0_sr1_adb0_wdb1 | auto | interleaved | - | auto | True | False | True | HiFi3 | True | 0 | 7 | 53.90 | 68.32 | 1.0000 | PASS |
| force_11x2=22c_block | block | interleaved | 11x2 | auto | True | True | True | HiFi3 | True | 0 | 7 | 55.01 | 69.66 | 1.0000 | PASS |
| force_13x1=13c_width | width | interleaved | 13x1 | auto | True | True | True | HiFi3 | True | 0 | 7 | 57.84 | 72.38 | 1.0000 | PASS |
| force_4x1=4c_block | block | interleaved | 4x1 | auto | True | True | True | HiFi3 | True | 0 | 7 | 58.24 | 72.83 | 1.0000 | PASS |
| force_8x1=8c_block | block | interleaved | 8x1 | auto | True | True | True | HiFi3 | True | 0 | 7 | 58.31 | 72.81 | 1.0000 | PASS |
| force_12x2=24c_block | block | interleaved | 12x2 | auto | True | True | True | HiFi3 | True | 0 | 7 | 59.69 | 74.35 | 1.0000 | PASS |
| force_2x1=2c_block | block | interleaved | 2x1 | auto | True | True | True | HiFi3 | True | 0 | 7 | 60.81 | 76.47 | 1.0000 | PASS |
| force_13x2=26c_block | block | interleaved | 13x2 | auto | True | True | True | HiFi3 | True | 0 | 7 | 64.30 | 79.12 | 1.0000 | PASS |
| auto_abh64_sr1_adb1_wdb0 | auto | interleaved | - | 64 | True | True | False | HiFi3 | True | 0 | 7 | 64.48 | 78.82 | 1.0000 | PASS |
| width_abh32_sr0_adb1_wdb0 | width | interleaved | - | 32 | False | True | False | HiFi3 | True | 0 | 7 | 64.78 | 79.39 | 1.0000 | PASS |
| auto_abh32_sr0_adb1_wdb0 | auto | interleaved | - | 32 | False | True | False | HiFi3 | True | 0 | 7 | 64.81 | 80.17 | 1.0000 | PASS |
| width_abh64_sr0_adb1_wdb0 | width | interleaved | - | 64 | False | True | False | HiFi3 | True | 0 | 7 | 64.88 | 79.26 | 1.0000 | PASS |
| force_6x1=6c_block | block | interleaved | 6x1 | auto | True | True | True | HiFi3 | True | 0 | 7 | 64.91 | 79.46 | 1.0000 | PASS |
| width_abh32_sr1_adb1_wdb0 | width | interleaved | - | 32 | True | True | False | HiFi3 | True | 0 | 7 | 65.15 | 79.67 | 1.0000 | PASS |
| auto_abh32_sr1_adb1_wdb0 | auto | interleaved | - | 32 | True | True | False | HiFi3 | True | 0 | 7 | 65.25 | 79.72 | 1.0000 | PASS |
| force_9x1=9c_block | block | interleaved | 9x1 | auto | True | True | True | HiFi3 | True | 0 | 7 | 65.28 | 79.80 | 1.0000 | PASS |
| width_abh64_sr1_adb1_wdb0 | width | interleaved | - | 64 | True | True | False | HiFi3 | True | 0 | 7 | 65.29 | 79.98 | 1.0000 | PASS |
| auto_abh64_sr0_adb1_wdb0 | auto | interleaved | - | 64 | False | True | False | HiFi3 | True | 0 | 7 | 65.60 | 79.92 | 1.0000 | PASS |
| force_3x1=3c_block | block | interleaved | 3x1 | auto | True | True | True | HiFi3 | True | 0 | 7 | 66.11 | 81.33 | 1.0000 | PASS |
| force_5x1=5c_block | block | interleaved | 5x1 | auto | True | True | True | HiFi3 | True | 0 | 7 | 71.59 | 86.39 | 1.0000 | PASS |
| force_10x1=10c_block | block | interleaved | 10x1 | auto | True | True | True | HiFi3 | True | 0 | 7 | 72.06 | 87.51 | 1.0000 | PASS |
| width_abh64_sr1_adb0_wdb1 | width | interleaved | - | 64 | True | False | True | HiFi3 | True | 0 | 7 | 72.10 | 87.18 | 1.0000 | PASS |
| auto_abh64_sr0_adb0_wdb1 | auto | interleaved | - | 64 | False | False | True | HiFi3 | True | 0 | 7 | 72.12 | 86.52 | 1.0000 | PASS |
| auto_abh64_sr1_adb0_wdb1 | auto | interleaved | - | 64 | True | False | True | HiFi3 | True | 0 | 7 | 72.12 | 86.56 | 1.0000 | PASS |
| width_abh32_sr0_adb0_wdb1 | width | interleaved | - | 32 | False | False | True | HiFi3 | True | 0 | 7 | 72.13 | 86.61 | 1.0000 | PASS |
| auto_abh32_sr1_adb0_wdb1 | auto | interleaved | - | 32 | True | False | True | HiFi3 | True | 0 | 7 | 72.16 | 86.50 | 1.0000 | PASS |
| auto_abh32_sr0_adb0_wdb1 | auto | interleaved | - | 32 | False | False | True | HiFi3 | True | 0 | 7 | 72.18 | 87.08 | 1.0000 | PASS |
| width_abh64_sr0_adb0_wdb1 | width | interleaved | - | 64 | False | False | True | HiFi3 | True | 0 | 7 | 72.21 | 87.21 | 1.0000 | PASS |
| width_abh32_sr1_adb0_wdb1 | width | interleaved | - | 32 | True | False | True | HiFi3 | True | 0 | 7 | 72.24 | 87.12 | 1.0000 | PASS |
| auto_abh32_sr1_adb0_wdb0 | auto | interleaved | - | 32 | True | False | False | HiFi3 | True | 0 | 7 | 72.56 | 87.90 | 1.0000 | PASS |
| auto_abh64_sr1_adb0_wdb0 | auto | interleaved | - | 64 | True | False | False | HiFi3 | True | 0 | 7 | 72.56 | 87.53 | 1.0000 | PASS |
| width_abh64_sr1_adb0_wdb0 | width | interleaved | - | 64 | True | False | False | HiFi3 | True | 0 | 7 | 72.58 | 87.66 | 1.0000 | PASS |
| width_abh32_sr0_adb0_wdb0 | width | interleaved | - | 32 | False | False | False | HiFi3 | True | 0 | 7 | 72.62 | 87.02 | 1.0000 | PASS |
| width_abh64_sr0_adb0_wdb0 | width | interleaved | - | 64 | False | False | False | HiFi3 | True | 0 | 7 | 72.65 | 87.51 | 1.0000 | PASS |
| auto_abh32_sr0_adb0_wdb0 | auto | interleaved | - | 32 | False | False | False | HiFi3 | True | 0 | 7 | 72.69 | 87.17 | 1.0000 | PASS |
| width_abh32_sr1_adb0_wdb0 | width | interleaved | - | 32 | True | False | False | HiFi3 | True | 0 | 7 | 72.89 | 87.94 | 1.0000 | PASS |
| auto_abh64_sr0_adb0_wdb0 | auto | interleaved | - | 64 | False | False | False | HiFi3 | True | 0 | 7 | 72.93 | 87.81 | 1.0000 | PASS |
| force_7x1=7c_block | block | interleaved | 7x1 | auto | True | True | True | HiFi3 | True | 0 | 7 | 74.91 | 90.24 | 1.0000 | PASS |
| force_11x1=11c_block | block | interleaved | 11x1 | auto | True | True | True | HiFi3 | True | 0 | 7 | 78.93 | 93.42 | 1.0000 | PASS |
| force_12x1=12c_block | block | interleaved | 12x1 | auto | True | True | True | HiFi3 | True | 0 | 7 | 85.75 | 100.18 | 1.0000 | PASS |
| force_13x1=13c_block | block | interleaved | 13x1 | auto | True | True | True | HiFi3 | True | 0 | 7 | 92.60 | 108.02 | 1.0000 | PASS |
| height_abh0_sr1_adb1_wdb0 | height | - | - | auto | True | True | False | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh0_sr1_adb1_wdb1 | height | - | - | auto | True | True | True | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh0_sr1_adb0_wdb0 | height | - | - | auto | True | False | False | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh0_sr1_adb0_wdb1 | height | - | - | auto | True | False | True | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh0_sr0_adb1_wdb0 | height | - | - | auto | False | True | False | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh0_sr0_adb1_wdb1 | height | - | - | auto | False | True | True | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh0_sr0_adb0_wdb0 | height | - | - | auto | False | False | False | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh0_sr0_adb0_wdb1 | height | - | - | auto | False | False | True | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh32_sr1_adb1_wdb0 | height | - | - | 32 | True | True | False | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh32_sr1_adb1_wdb1 | height | - | - | 32 | True | True | True | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh32_sr1_adb0_wdb0 | height | - | - | 32 | True | False | False | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh32_sr1_adb0_wdb1 | height | - | - | 32 | True | False | True | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh32_sr0_adb1_wdb0 | height | - | - | 32 | False | True | False | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh32_sr0_adb1_wdb1 | height | - | - | 32 | False | True | True | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh32_sr0_adb0_wdb0 | height | - | - | 32 | False | False | False | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh32_sr0_adb0_wdb1 | height | - | - | 32 | False | False | True | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh64_sr1_adb1_wdb0 | height | - | - | 64 | True | True | False | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh64_sr1_adb1_wdb1 | height | - | - | 64 | True | True | True | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh64_sr1_adb0_wdb0 | height | - | - | 64 | True | False | False | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh64_sr1_adb0_wdb1 | height | - | - | 64 | True | False | True | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh64_sr0_adb1_wdb0 | height | - | - | 64 | False | True | False | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh64_sr0_adb1_wdb1 | height | - | - | 64 | False | True | True | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh64_sr0_adb0_wdb0 | height | - | - | 64 | False | False | False | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| height_abh64_sr0_adb0_wdb1 | height | - | - | 64 | False | False | True | HiFi3 | True | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |

## Fidelity sweep (production-best program config: auto/block 24c, double-buffered)

Lowest device time among PCC-passing rows is the recommended conv `math_fidelity`. Inputs are bf16, so PCC saturates by HiFi2 and extra passes only cost time.

**Recommended: `LoFi` (fp32_acc=False) — 16.32µs conv, PCC=0.9997**

| fidelity | fp32_acc | abh | cores | conv_us | total_us | PCC | result |
|---|---|---|---|---|---|---|---|
| LoFi **(recommended)** | False | 32 | 0 | 16.32 | 30.51 | 0.9997 | PASS |
| LoFi | True | auto | 0 | 16.45 | 30.92 | 0.9999 | PASS |
| LoFi | False | auto | 0 | 16.50 | 30.92 | 0.9997 | PASS |
| LoFi | True | 32 | 0 | 16.51 | 30.75 | 0.9999 | PASS |
| HiFi2 | True | auto | 0 | 16.89 | 31.20 | 1.0000 | PASS |
| HiFi2 | True | 32 | 0 | 17.06 | 31.43 | 1.0000 | PASS |
| HiFi2 | False | auto | 0 | 17.09 | 31.52 | 0.9995 | PASS |
| HiFi2 | False | 32 | 0 | 17.11 | 31.30 | 0.9995 | PASS |
| LoFi | True | 64 | 0 | 18.00 | 32.41 | 0.9999 | PASS |
| LoFi | False | 64 | 0 | 18.35 | 32.66 | 0.9997 | PASS |
| HiFi3 | True | 32 | 0 | 21.65 | 36.00 | 1.0000 | PASS |
| HiFi3 | False | 32 | 0 | 21.67 | 36.10 | 0.9995 | PASS |
| HiFi3 | False | auto | 0 | 21.68 | 36.01 | 0.9995 | PASS |
| HiFi3 | True | auto | 0 | 21.79 | 36.21 | 1.0000 | PASS |
| HiFi4 | True | auto | 0 | 27.36 | 41.75 | 1.0000 | PASS |
| HiFi4 | False | 32 | 0 | 27.36 | 41.73 | 0.9995 | PASS |
| HiFi4 | True | 32 | 0 | 27.36 | 41.63 | 1.0000 | PASS |
| HiFi4 | False | auto | 0 | 27.42 | 41.73 | 0.9995 | PASS |
| HiFi2 | True | 64 | 0 | 29.44 | 44.04 | 1.0000 | PASS |
| HiFi2 | False | 64 | 0 | 29.58 | 43.94 | 0.9995 | PASS |
| HiFi3 | False | 64 | 0 | 40.83 | 55.25 | 0.9995 | PASS |
| HiFi3 | True | 64 | 0 | 40.88 | 55.41 | 1.0000 | PASS |
| HiFi4 | False | 64 | 0 | 52.20 | 66.59 | 0.9995 | PASS |
| HiFi4 | True | 64 | 0 | 52.26 | 66.86 | 1.0000 | PASS |

## Core-count saturation (same shape) — more cores does NOT improve device time

Holding the row split at its max (gy = m_tiles = 3) and raising requested cores via the forced grid (block sharding; width can't be forced — few-core width overflows L1 and segfaults). `compute_cores` (the conv kernel's grid, = tracy `Cores`) scales with the request above the natural pick, yet `conv_us`/`total_us` rise monotonically — the conv is overhead-bound, not core-starved. `output_shard_cores` (the output tensor layout) stays pinned at the legal max (block_w>=2 => gx<=n_tiles//2).

Global fastest (unforced, ttnn auto): **fid_LoFi_fp32acc0_abh32 -> 0 cores, 16.32µs** — every forced grid below is slower; forcing MORE cores than the auto pick only makes it worse.

| shard | forced_grid | requested_cores | compute_cores | output_shard_cores | conv_us | total_us |
|---|---|---|---|---|---|---|
| block | 5x3 | 15 | 15 | 0 | 26.84 | 41.27 |
| block | 7x3 | 21 | 21 | 0 | 27.64 | 42.15 |
| block | 9x3 **(fastest)** | 27 | 27 | 0 | 24.27 | 38.88 |
| block | 10x3 | 30 | 30 | 0 | 26.67 | 41.02 |
| block | 11x3 | 33 | 33 | 0 | 28.95 | 43.53 |
| block | 12x3 | 36 | 36 | 0 | 31.45 | 45.94 |
| block | 13x3 | 39 | 39 | 0 | 33.83 | 48.46 |
