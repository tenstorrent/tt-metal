Perf baseline
DTYPE: dtype_bf16, FP32 ACC: fp32_acc, FIDELITY: HiFi2
| M, K, N | math util (%) | measured perf (ms) | attributes |
|---|---:|---:|---:|
| (512, 512, 512) | 7.7 | 0.012 | 'MinimalMatmulConfig(M_block_size=8;K_block_size=8;N_block_size=2;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (512, 1024, 1024) | 17.6 | 0.021 | 'MinimalMatmulConfig(M_block_size=2;K_block_size=4;N_block_size=4;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (512, 1024, 2048) | 24.0 | 0.031 | 'MinimalMatmulConfig(M_block_size=2;K_block_size=4;N_block_size=8;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (1024, 1024, 1024) | 22.5 | 0.033 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=2;N_block_size=4;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';
|
| (1024, 1024, 2048) | 35.6 | 0.041 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=2;N_block_size=8;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';
|
| (1024, 2048, 2048) | 47.1 | 0.062 | 'MinimalMatmulConfig(M_block_size=8;K_block_size=4;N_block_size=8;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';
|
| (2048, 2048, 2048) | 51.8 | 0.113 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=4;N_block_size=8;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';
|
| (2048, 2048, 3072) | 51.0 | 0.173 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=4;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';
 |
| (2048, 3072, 3072) | 55.8 | 0.237 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=2;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';
 |
| (3072, 3072, 3072) | 60.2 | 0.329 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=4;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';
 |
| (3072, 3072, 4096) | 65.8 | 0.401 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=4;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';
 |
| (3072, 4096, 4096) | 69.4 | 0.507 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=8;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';
 |
| (4096, 4096, 4096) | 72.0 | 0.652 | 'MinimalMatmulConfig(M_block_size=8;K_block_size=4;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';
 |
| (8192, 8192, 8192) | 84.1 | 4.465 | 'MinimalMatmulConfig(M_block_size=16;K_block_size=4;N_block_size=8;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';
 |
| (16384, 16384, 16384) | 90.3 | 33.245 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=16;N_block_size=8;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10)
)';  |

New perf
DTYPE: dtype_bf16, FP32 ACC: fp32_acc, FIDELITY: HiFi2
| M, K, N | math util (%) | measured perf (ms) | attributes |
|---|---:|---:|---:|
| (512, 512, 512) | 8.2 | 0.011 | 'MinimalMatmulConfig(M_block_size=2;K_block_size=4;N_block_size=2;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (512, 1024, 1024) | 18.2 | 0.020 | 'MinimalMatmulConfig(M_block_size=2;K_block_size=4;N_block_size=4;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (512, 1024, 2048) | 25.0 | 0.029 | 'MinimalMatmulConfig(M_block_size=2;K_block_size=4;N_block_size=8;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (1024, 1024, 1024) | 23.0 | 0.032 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=8;N_block_size=4;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (1024, 1024, 2048) | 36.3 | 0.040 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=4;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (1024, 2048, 2048) | 48.7 | 0.060 | 'MinimalMatmulConfig(M_block_size=16;K_block_size=4;N_block_size=8;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (2048, 2048, 2048) | 53.9 | 0.109 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=4;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (2048, 2048, 3072) | 54.5 | 0.161 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=4;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (2048, 3072, 3072) | 57.9 | 0.228 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=4;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (3072, 3072, 3072) | 64.1 | 0.309 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=4;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (3072, 3072, 4096) | 70.7 | 0.373 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=4;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (3072, 4096, 4096) | 72.8 | 0.483 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=8;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (4096, 4096, 4096) | 75.3 | 0.624 | 'MinimalMatmulConfig(M_block_size=8;K_block_size=4;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (8192, 8192, 8192) | 87.3 | 4.302 | 'MinimalMatmulConfig(M_block_size=16;K_block_size=4;N_block_size=8;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (16384, 16384, 16384) | 90.7 | 33.103 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=16;N_block_size=8;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |

New perf with no in1 reuse
DTYPE: dtype_bf16, FP32 ACC: fp32_acc, FIDELITY: HiFi2
| M, K, N | math util (%) | measured perf (ms) | attributes |
|---|---:|---:|---:|
| (512, 512, 512) | 8.2 | 0.011 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=4;N_block_size=4;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (512, 1024, 1024) | 18.7 | 0.020 | 'MinimalMatmulConfig(M_block_size=2;K_block_size=4;N_block_size=8;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (512, 1024, 2048) | 25.2 | 0.029 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=4;N_block_size=8;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (1024, 1024, 1024) | 23.2 | 0.032 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=8;N_block_size=4;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (1024, 1024, 2048) | 36.4 | 0.040 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=4;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (1024, 2048, 2048) | 48.6 | 0.060 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=4;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (2048, 2048, 2048) | 53.7 | 0.109 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=4;N_block_size=8;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (2048, 2048, 3072) | 54.0 | 0.163 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=4;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (2048, 3072, 3072) | 58.0 | 0.228 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=4;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (3072, 3072, 3072) | 64.2 | 0.308 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=8;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (3072, 3072, 4096) | 70.5 | 0.374 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=4;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (3072, 4096, 4096) | 73.1 | 0.481 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=8;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (4096, 4096, 4096) | 75.3 | 0.623 | 'MinimalMatmulConfig(M_block_size=8;K_block_size=4;N_block_size=16;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (8192, 8192, 8192) | 87.4 | 4.297 | 'MinimalMatmulConfig(M_block_size=16;K_block_size=4;N_block_size=8;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |
| (16384, 16384, 16384) | 90.7 | 33.110 | 'MinimalMatmulConfig(M_block_size=4;K_block_size=16;N_block_size=8;subblock_h=2;subblock_w=2;compute_with_storage_grid_size=(x=11;y=10))';  |

---

## Performance Comparison: Baseline vs New vs No in1 Reuse

| M, K, N | Baseline Time (ms) | Baseline Util (%) | New Time (ms) | New Util (%) | Speedup | No in1 Time (ms) | No in1 Util (%) | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (512, 512, 512) | 0.012 | 7.7 | 0.011 | 8.2 | 1.09x | 0.011 | 8.2 | 1.09x |
| (512, 1024, 1024) | 0.021 | 17.6 | 0.020 | 18.2 | 1.05x | 0.020 | 18.7 | 1.05x |
| (512, 1024, 2048) | 0.031 | 24.0 | 0.029 | 25.0 | 1.07x | 0.029 | 25.2 | 1.07x |
| (1024, 1024, 1024) | 0.033 | 22.5 | 0.032 | 23.0 | 1.03x | 0.032 | 23.2 | 1.03x |
| (1024, 1024, 2048) | 0.041 | 35.6 | 0.040 | 36.3 | 1.03x | 0.040 | 36.4 | 1.03x |
| (1024, 2048, 2048) | 0.062 | 47.1 | 0.060 | 48.7 | 1.03x | 0.060 | 48.6 | 1.03x |
| (2048, 2048, 2048) | 0.113 | 51.8 | 0.109 | 53.9 | 1.04x | 0.109 | 53.7 | 1.04x |
| (2048, 2048, 3072) | 0.173 | 51.0 | 0.161 | 54.5 | 1.07x | 0.163 | 54.0 | 1.06x |
| (2048, 3072, 3072) | 0.237 | 55.8 | 0.228 | 57.9 | 1.04x | 0.228 | 58.0 | 1.04x |
| (3072, 3072, 3072) | 0.329 | 60.2 | 0.309 | 64.1 | 1.06x | 0.308 | 64.2 | 1.07x |
| (3072, 3072, 4096) | 0.401 | 65.8 | 0.373 | 70.7 | 1.08x | 0.374 | 70.5 | 1.07x |
| (3072, 4096, 4096) | 0.507 | 69.4 | 0.483 | 72.8 | 1.05x | 0.481 | 73.1 | 1.05x |
| (4096, 4096, 4096) | 0.652 | 72.0 | 0.624 | 75.3 | 1.04x | 0.623 | 75.3 | 1.05x |
| (8192, 8192, 8192) | 4.465 | 84.1 | 4.302 | 87.3 | 1.04x | 4.297 | 87.4 | 1.04x |
| (16384, 16384, 16384) | 33.245 | 90.3 | 33.103 | 90.7 | 1.00x | 33.110 | 90.7 | 1.00x |

### Speedup Summary: New vs Baseline
- **Min Speedup:** 1.00x (16384, 16384, 16384)
- **Mean Speedup:** 1.05x
- **Max Speedup:** 1.09x (512, 512, 512)

### Speedup Summary: No in1 Reuse vs Baseline
- **Min Speedup:** 1.00x (16384, 16384, 16384)
- **Mean Speedup:** 1.05x
- **Max Speedup:** 1.09x (512, 512, 512)
