# rms_norm device-kernel-time baseline (Refinement 6)

Per-op on-device kernel time in microseconds (best of 7), 8x8 Wormhole grid.
Measured via ttnn.ReadDeviceProfiler + get_latest_programs_perf_data
(DEVICE KERNEL DURATION [ns]). Heuristic regime: REGIME_B_MIN_WT_TILE=16, REGIME_B_MIN_WT_RM=16.

| shape | dtype | layout | no_gamma (us) | gamma (us) |
|-------|-------|--------|---------------|------------|
| A: (1,1,2048,256) | bf16 | TILE |    17.85 |    19.32 |
| A: (1,1,2048,256) | bf16 | RM |    19.07 |    20.66 |
| A: (1,1,2048,256) | fp32 | TILE |    31.48 |    31.18 |
| A: (1,1,2048,256) | fp32 | RM |    28.36 |    30.40 |
| A: (4,1,512,512) | bf16 | TILE |    30.41 |    32.68 |
| A: (4,1,512,512) | bf16 | RM |    32.68 |    36.02 |
| A: (4,1,512,512) | fp32 | TILE |    58.40 |    59.11 |
| A: (4,1,512,512) | fp32 | RM |    50.58 |    54.12 |
| A: (1024,1024) | bf16 | TILE |    38.02 |    42.83 |
| A: (1024,1024) | bf16 | RM |    41.89 |    46.26 |
| A: (1024,1024) | fp32 | TILE |    63.57 |    70.52 |
| A: (1024,1024) | fp32 | RM |    60.40 |    66.10 |
| B: (1,1,32,4096) | bf16 | TILE |    22.93 |    25.39 |
| B: (1,1,32,4096) | bf16 | RM |    29.19 |    31.95 |
| B: (1,1,32,4096) | fp32 | TILE |    30.41 |    32.90 |
| B: (1,1,32,4096) | fp32 | RM |    34.05 |    38.52 |
| B: (1,1,32,8192) | bf16 | TILE |    34.17 |    36.91 |
| B: (1,1,32,8192) | bf16 | RM |    38.81 |    41.44 |
| B: (1,1,32,8192) | fp32 | TILE |    45.46 |    48.47 |
| B: (1,1,32,8192) | fp32 | RM |    44.92 |    49.47 |
| B: (1,1,32,16384) | bf16 | TILE |    48.10 |    52.62 |
| B: (1,1,32,16384) | bf16 | RM |    56.37 |    61.48 |
| B: (1,1,32,16384) | fp32 | TILE |    68.01 |    72.81 |
| B: (1,1,32,16384) | fp32 | RM |    65.51 |    74.31 |
