# rms_norm device-kernel-time baseline (Refinement 6)

Per-op on-device kernel time in microseconds (best of 7), 8x8 Wormhole grid.
Measured via ttnn.ReadDeviceProfiler + get_latest_programs_perf_data
(DEVICE KERNEL DURATION [ns]). Heuristic regime: REGIME_B_MIN_WT_TILE=16, REGIME_B_MIN_WT_RM=16.

| shape | dtype | layout | no_gamma (us) | gamma (us) |
|-------|-------|--------|---------------|------------|
| A: (1,1,2048,256) | bf16 | TILE |    17.75 |    19.36 |
| A: (1,1,2048,256) | bf16 | RM |    18.48 |    20.07 |
| A: (1,1,2048,256) | fp32 | TILE |    30.27 |    31.38 |
| A: (1,1,2048,256) | fp32 | RM |    28.46 |    30.02 |
| A: (4,1,512,512) | bf16 | TILE |    30.08 |    33.39 |
| A: (4,1,512,512) | bf16 | RM |    32.29 |    35.40 |
| A: (4,1,512,512) | fp32 | TILE |    58.80 |    59.04 |
| A: (4,1,512,512) | fp32 | RM |    50.89 |    55.24 |
| A: (1024,1024) | bf16 | TILE |    38.36 |    42.39 |
| A: (1024,1024) | bf16 | RM |    41.78 |    46.06 |
| A: (1024,1024) | fp32 | TILE |    64.55 |    71.26 |
| A: (1024,1024) | fp32 | RM |    60.70 |    66.82 |
| B: (1,1,32,4096) | bf16 | TILE |    16.31 |    17.56 |
| B: (1,1,32,4096) | bf16 | RM |    18.48 |    20.15 |
| B: (1,1,32,4096) | fp32 | TILE |    24.31 |    26.09 |
| B: (1,1,32,4096) | fp32 | RM |    24.29 |    26.59 |
| B: (1,1,32,8192) | bf16 | TILE |    22.09 |    23.53 |
| B: (1,1,32,8192) | bf16 | RM |    22.69 |    24.13 |
| B: (1,1,32,8192) | fp32 | TILE |    36.42 |    39.23 |
| B: (1,1,32,8192) | fp32 | RM |    35.97 |    38.04 |
| B: (1,1,32,16384) | bf16 | TILE |    30.68 |    33.31 |
| B: (1,1,32,16384) | bf16 | RM |    31.48 |    34.10 |
| B: (1,1,32,16384) | fp32 | TILE |    52.81 |    57.54 |
| B: (1,1,32,16384) | fp32 | RM |    50.67 |    54.71 |
