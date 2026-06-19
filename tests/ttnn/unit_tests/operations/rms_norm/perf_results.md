# rms_norm device-kernel-time baseline (Refinement 6)

Per-op on-device kernel time in microseconds (best of 7), 8x8 Wormhole grid.
Measured via ttnn.ReadDeviceProfiler + get_latest_programs_perf_data
(DEVICE KERNEL DURATION [ns]). Heuristic regime: REGIME_B_MIN_WT_TILE=160, REGIME_B_MIN_WT_RM=96.

| shape | dtype | layout | no_gamma (us) | gamma (us) |
|-------|-------|--------|---------------|------------|
| A: (1,1,2048,256) | bf16 | TILE |    17.61 |    19.17 |
| A: (1,1,2048,256) | bf16 | RM |    18.46 |    20.20 |
| A: (1,1,2048,256) | fp32 | TILE |    30.63 |    32.89 |
| A: (1,1,2048,256) | fp32 | RM |    28.00 |    30.48 |
| A: (4,1,512,512) | bf16 | TILE |    30.55 |    33.02 |
| A: (4,1,512,512) | bf16 | RM |    32.49 |    35.32 |
| A: (4,1,512,512) | fp32 | TILE |    58.42 |    57.47 |
| A: (4,1,512,512) | fp32 | RM |    50.47 |    55.23 |
| A: (1024,1024) | bf16 | TILE |    38.70 |    42.61 |
| A: (1024,1024) | bf16 | RM |    41.83 |    46.24 |
| A: (1024,1024) | fp32 | TILE |    63.04 |    70.84 |
| A: (1024,1024) | fp32 | RM |    60.50 |    67.07 |
| B: (1,1,32,4096) | bf16 | TILE |    99.88 |   115.35 |
| B: (1,1,32,4096) | bf16 | RM |   108.79 |   107.29 |
| B: (1,1,32,4096) | fp32 | TILE |   138.82 |   120.38 |
| B: (1,1,32,4096) | fp32 | RM |   119.23 |   120.46 |
| B: (1,1,32,8192) | bf16 | TILE |   110.11 |   112.98 |
| B: (1,1,32,8192) | bf16 | RM |   111.13 |   111.74 |
| B: (1,1,32,8192) | fp32 | TILE |   128.53 |   130.37 |
| B: (1,1,32,8192) | fp32 | RM |   127.61 |   128.40 |
| B: (1,1,32,16384) | bf16 | TILE |   116.89 |   119.86 |
| B: (1,1,32,16384) | bf16 | RM |   118.33 |   120.75 |
| B: (1,1,32,16384) | fp32 | TILE |   143.04 |   145.88 |
| B: (1,1,32,16384) | fp32 | RM |   141.38 |   143.93 |
