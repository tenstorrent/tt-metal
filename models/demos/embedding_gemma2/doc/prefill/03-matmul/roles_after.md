
#### b1_s128: kernel sum 4.518 ms, matmul 1.366 ms (30.2 %)

| role | M x K x N | calls | device us (per call) | share of forward | cores | family | in/out mem | bound (model) | floor us | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| down | 128x2048x512 | 24 | 294.1 (12.3) | 6.5 % | 32 | Matmul | L1/L1 | memory | 53.7 | 5.5x |
| gate | 128x512x2048 | 24 | 216.2 (9.0) | 4.8 % | 44 | Matmul | L1/L1 | memory | 53.7 | 4.0x |
| up | 128x512x2048 | 24 | 215.7 (9.0) | 4.8 % | 44 | Matmul | L1/L1 | memory | 53.7 | 4.0x |
| qkv_sliding | 128x512x2048 | 20 | 169.2 (8.5) | 3.7 % | 64 | Matmul | L1/L1 | memory | 44.8 | 3.8x |
| o_sliding | 128x1024x512 | 20 | 140.2 (7.0) | 3.1 % | 32 | Matmul | L1/L1 | memory | 22.5 | 6.2x |
| ple_proj | 128x512x512 | 24 | 105.4 (4.4) | 2.3 % | 32 | Matmul | L1/L1 | memory | 13.7 | 7.7x |
| ple_gate | 128x512x512 | 24 | 95.1 (4.0) | 2.1 % | 32 | Matmul | L1/L1 | memory | 13.7 | 7.0x |
| o_full | 128x2048x512 | 4 | 49.2 (12.3) | 1.1 % | 32 | Matmul | L1/L1 | memory | 9.0 | 5.5x |
| qkv_full | 128x512x3072 | 4 | 37.8 (9.5) | 0.8 % | 96 | Matmul | L1/L1 | memory | 13.4 | 2.8x |
| ple_wide | 128x512x12288 | 1 | 34.2 (34.2) | 0.8 % | 96 | Matmul | DRAM/DRAM | memory | 19.5 | 1.8x |
| final | 128x512x768 | 1 | 5.0 (5.0) | 0.1 % | 48 | Matmul | L1/DRAM | memory | 1.2 | 4.2x |
| pooler | 32x128x768 | 1 | 3.8 (3.8) | 0.1 % | 24 | Matmul | DRAM/DRAM | memory | 0.6 | 6.4x |

#### b1_s2048: kernel sum 23.573 ms, matmul 5.918 ms (25.1 %)

| role | M x K x N | calls | device us (per call) | share of forward | cores | family | in/out mem | bound (model) | floor us | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| down | 2048x2048x512 | 24 | 1101.3 (45.9) | 4.7 % | 120 | MinimalMatmul | DRAM/L1 | memory | 450.3 | 2.4x |
| gate | 2048x512x2048 | 24 | 1030.4 (42.9) | 4.4 % | 110 | Matmul | L1/DRAM | memory | 450.3 | 2.3x |
| up | 2048x512x2048 | 24 | 1026.9 (42.8) | 4.4 % | 110 | Matmul | L1/DRAM | memory | 450.3 | 2.3x |
| qkv_sliding | 2048x512x2048 | 20 | 859.8 (43.0) | 3.6 % | 110 | Matmul | L1/DRAM | memory | 375.2 | 2.3x |
| o_sliding | 2048x1024x512 | 20 | 430.3 (21.5) | 1.8 % | 80 | Matmul | L1/L1 | compute | 129.5 | 3.3x |
| ple_proj | 2048x512x512 | 24 | 342.4 (14.3) | 1.5 % | 80 | Matmul | L1/L1 | compute | 77.7 | 4.4x |
| ple_gate | 2048x512x512 | 24 | 339.0 (14.1) | 1.4 % | 80 | Matmul | L1/L1 | compute | 77.7 | 4.4x |
| ple_wide | 2048x512x12288 | 1 | 279.3 (279.3) | 1.2 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 115.5 | 2.4x |
| qkv_full | 2048x512x3072 | 4 | 247.1 (61.8) | 1.0 % | 120 | Matmul | L1/DRAM | memory | 112.2 | 2.2x |
| o_full | 2048x2048x512 | 4 | 203.3 (50.8) | 0.9 % | 120 | MinimalMatmul | DRAM/L1 | memory | 75.0 | 2.7x |
| pooler | 32x2048x768 | 1 | 29.7 (29.7) | 0.1 % | 24 | Matmul | DRAM/DRAM | memory | 6.6 | 4.5x |
| final | 2048x512x768 | 1 | 28.4 (28.4) | 0.1 % | 120 | MinimalMatmul | L1/DRAM | memory | 7.2 | 4.0x |

#### b1_s4096: kernel sum 41.337 ms, matmul 11.410 ms (27.6 %)

| role | M x K x N | calls | device us (per call) | share of forward | cores | family | in/out mem | bound (model) | floor us | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| up | 4096x512x2048 | 24 | 1923.7 (80.2) | 4.7 % | 110 | Matmul | L1/DRAM | memory | 848.4 | 2.3x |
| gate | 4096x512x2048 | 24 | 1922.6 (80.1) | 4.7 % | 110 | Matmul | L1/DRAM | memory | 848.4 | 2.3x |
| down | 4096x2048x512 | 24 | 1750.8 (72.9) | 4.2 % | 120 | MinimalMatmul | DRAM/L1 | memory | 848.4 | 2.1x |
| qkv_sliding | 4096x512x2048 | 20 | 1607.2 (80.4) | 3.9 % | 110 | Matmul | L1/DRAM | memory | 707.0 | 2.3x |
| ple_gate | 4096x512x512 | 24 | 1261.7 (52.6) | 3.1 % | 80 | Matmul +gelu | L1/L1 | compute | 155.3 | 8.1x |
| o_sliding | 4096x1024x512 | 20 | 919.5 (46.0) | 2.2 % | 120 | MinimalMatmul | DRAM/L1 | memory | 357.5 | 2.6x |
| ple_proj | 4096x512x512 | 24 | 582.9 (24.3) | 1.4 % | 80 | Matmul | L1/L1 | compute | 155.3 | 3.8x |
| ple_wide | 4096x512x12288 | 1 | 538.3 (538.3) | 1.3 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 217.9 | 2.5x |
| qkv_full | 4096x512x3072 | 4 | 449.9 (112.5) | 1.1 % | 120 | Matmul | L1/DRAM | memory | 211.3 | 2.1x |
| o_full | 4096x2048x512 | 4 | 343.1 (85.8) | 0.8 % | 120 | MinimalMatmul | DRAM/L1 | memory | 141.4 | 2.4x |
| pooler | 32x4096x768 | 1 | 58.0 (58.0) | 0.1 % | 24 | Matmul | DRAM/DRAM | memory | 13.0 | 4.5x |
| final | 4096x512x768 | 1 | 52.3 (52.3) | 0.1 % | 120 | MinimalMatmul | L1/DRAM | memory | 13.5 | 3.9x |

#### b1_s8192: kernel sum 94.426 ms, matmul 23.724 ms (25.1 %)

| role | M x K x N | calls | device us (per call) | share of forward | cores | family | in/out mem | bound (model) | floor us | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| gate | 8192x512x2048 | 24 | 4479.8 (186.7) | 4.7 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 2018.3 | 2.2x |
| up | 8192x512x2048 | 24 | 4280.1 (178.3) | 4.5 % | 110 | Matmul | DRAM/DRAM | memory | 2018.3 | 2.1x |
| qkv_sliding | 8192x512x2048 | 20 | 3750.3 (187.5) | 4.0 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 1681.9 | 2.2x |
| down | 8192x2048x512 | 24 | 3285.2 (136.9) | 3.5 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 2018.3 | 1.6x |
| o_sliding | 8192x1024x512 | 20 | 1573.9 (78.7) | 1.7 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 1004.8 | 1.6x |
| ple_proj | 8192x512x512 | 24 | 1518.1 (63.3) | 1.6 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 799.5 | 1.9x |
| ple_gate | 8192x512x512 | 24 | 1517.8 (63.2) | 1.6 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 799.5 | 1.9x |
| ple_layer | 8192x512x512 | 24 | 1433.5 (59.7) | 1.5 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 799.5 | 1.8x |
| qkv_full | 8192x512x3072 | 4 | 1109.4 (277.3) | 1.2 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 471.8 | 2.4x |
| o_full | 8192x2048x512 | 4 | 578.2 (144.6) | 0.6 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 336.4 | 1.7x |
| pooler | 32x8192x768 | 1 | 114.0 (114.0) | 0.1 % | 24 | Matmul | DRAM/DRAM | memory | 25.8 | 4.4x |
| final | 8192x512x768 | 1 | 83.4 (83.4) | 0.1 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 41.8 | 2.0x |

#### b1_s16384: kernel sum 200.708 ms, matmul 44.689 ms (22.3 %)

| role | M x K x N | calls | device us (per call) | share of forward | cores | family | in/out mem | bound (model) | floor us | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| gate | 16384x512x2048 | 24 | 8489.5 (353.7) | 4.2 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 3984.4 | 2.1x |
| up | 16384x512x2048 | 24 | 8205.7 (341.9) | 4.1 % | 110 | Matmul | DRAM/DRAM | memory | 3984.4 | 2.1x |
| qkv_sliding | 16384x512x2048 | 20 | 7057.5 (352.9) | 3.5 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 3320.3 | 2.1x |
| down | 16384x2048x512 | 24 | 6080.8 (253.4) | 3.0 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 3984.4 | 1.5x |
| o_sliding | 16384x1024x512 | 20 | 3094.3 (154.7) | 1.5 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 1987.8 | 1.6x |
| ple_gate | 16384x512x512 | 24 | 2821.4 (117.6) | 1.4 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 1585.9 | 1.8x |
| ple_proj | 16384x512x512 | 24 | 2798.1 (116.6) | 1.4 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 1585.9 | 1.8x |
| ple_layer | 16384x512x512 | 24 | 2606.0 (108.6) | 1.3 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 1585.9 | 1.6x |
| qkv_full | 16384x512x3072 | 4 | 2060.1 (515.0) | 1.0 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 930.6 | 2.2x |
| o_full | 16384x2048x512 | 4 | 1092.4 (273.1) | 0.5 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 664.1 | 1.6x |
| pooler | 32x16384x768 | 1 | 226.9 (226.9) | 0.1 % | 24 | Matmul | DRAM/DRAM | memory | 51.4 | 4.4x |
| final | 16384x512x768 | 1 | 156.2 (156.2) | 0.1 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 82.7 | 1.9x |

#### b1_s65536: kernel sum 1418.856 ms, matmul 209.134 ms (14.7 %)

| role | M x K x N | calls | device us (per call) | share of forward | cores | family | in/out mem | bound (model) | floor us | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| gate | 65536x512x2048 | 24 | 66465.2 (2769.4) | 4.7 % | 120 | MinimalMatmul +gelu | DRAM/DRAM | memory | 15780.9 | 4.2x |
| up | 65536x512x2048 | 24 | 33666.1 (1402.8) | 2.4 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 15780.9 | 2.1x |
| qkv_sliding | 65536x512x2048 | 20 | 28132.2 (1406.6) | 2.0 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 13150.7 | 2.1x |
| down | 65536x2048x512 | 24 | 25242.8 (1051.8) | 1.8 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 15780.9 | 1.6x |
| o_sliding | 65536x1024x512 | 20 | 10659.5 (533.0) | 0.8 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 7886.1 | 1.4x |
| ple_layer | 65536x512x512 | 24 | 10462.0 (435.9) | 0.7 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 6304.5 | 1.7x |
| ple_proj | 65536x512x512 | 24 | 10418.2 (434.1) | 0.7 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 6304.5 | 1.7x |
| ple_gate | 65536x512x512 | 24 | 10398.9 (433.3) | 0.7 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 6304.5 | 1.6x |
| qkv_full | 65536x512x3072 | 4 | 7895.4 (1973.8) | 0.6 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 3683.1 | 2.1x |
| o_full | 65536x2048x512 | 4 | 4247.6 (1061.9) | 0.3 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 2630.1 | 1.6x |
| pooler | 32x65536x768 | 1 | 929.4 (929.4) | 0.1 % | 24 | Matmul | DRAM/DRAM | memory | 205.0 | 4.5x |
| final | 65536x512x768 | 1 | 616.5 (616.5) | 0.0 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 328.5 | 1.9x |
