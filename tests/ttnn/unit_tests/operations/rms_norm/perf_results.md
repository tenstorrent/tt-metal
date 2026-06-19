# rms_norm device-kernel-time baseline (Refinement 6)

Per-op on-device kernel time in microseconds (best of 7), 8x8 Wormhole grid.
Measured via ttnn.ReadDeviceProfiler + get_latest_programs_perf_data
(DEVICE KERNEL DURATION [ns]). Heuristic-selected regime (REGIME_B_MIN_WT=160).

| shape | dtype | layout | no_gamma (us) | gamma (us) |
|-------|-------|--------|---------------|------------|
| A: (1,1,2048,256) | bf16 | TILE |    17.64 |    19.09 |
| A: (1,1,2048,256) | bf16 | RM |    18.37 |    20.48 |
| A: (1,1,2048,256) | fp32 | TILE |    30.61 |    31.00 |
| A: (1,1,2048,256) | fp32 | RM |    28.14 |    30.36 |
| A: (4,1,512,512) | bf16 | TILE |    30.89 |    32.38 |
| A: (4,1,512,512) | bf16 | RM |    31.83 |    35.86 |
| A: (4,1,512,512) | fp32 | TILE |    57.60 |    59.08 |
| A: (4,1,512,512) | fp32 | RM |    50.02 |    54.85 |
| A: (1024,1024) | bf16 | TILE |    38.18 |    42.58 |
| A: (1024,1024) | bf16 | RM |    41.71 |    46.24 |
| A: (1024,1024) | fp32 | TILE |    64.05 |    70.33 |
| A: (1024,1024) | fp32 | RM |    60.07 |    66.97 |
| B: (1,1,32,4096) | bf16 | TILE |    99.84 |   115.06 |
| B: (1,1,32,4096) | bf16 | RM |   150.29 |   170.09 |
| B: (1,1,32,4096) | fp32 | TILE |   138.91 |   120.47 |
| B: (1,1,32,4096) | fp32 | RM |   170.39 |   120.41 |
| B: (1,1,32,8192) | bf16 | TILE |   109.90 |   113.13 |
| B: (1,1,32,8192) | bf16 | RM |   111.17 |   111.78 |
| B: (1,1,32,8192) | fp32 | TILE |   128.58 |   129.62 |
| B: (1,1,32,8192) | fp32 | RM |   127.46 |   128.87 |
| B: (1,1,32,16384) | bf16 | TILE |   117.12 |   120.34 |
| B: (1,1,32,16384) | bf16 | RM |   118.16 |   120.74 |
| B: (1,1,32,16384) | fp32 | TILE |   143.10 |   146.50 |
| B: (1,1,32,16384) | fp32 | RM |   141.06 |   143.57 |
