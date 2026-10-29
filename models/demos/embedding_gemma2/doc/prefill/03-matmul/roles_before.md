
#### b1_s128: kernel sum 4.555 ms, matmul 1.406 ms (30.9 %)

| role | M x K x N | calls | device us (per call) | share of forward | cores | family | in/out mem | bound (model) | floor us | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| down | 128x2048x512 | 24 | 294.0 (12.3) | 6.5 % | 32 | Matmul | L1/L1 | memory | 53.7 | 5.5x |
| up | 128x512x2048 | 24 | 216.0 (9.0) | 4.7 % | 44 | Matmul | L1/L1 | memory | 53.7 | 4.0x |
| gate | 128x512x2048 | 24 | 215.1 (9.0) | 4.7 % | 44 | Matmul | L1/L1 | memory | 53.7 | 4.0x |
| qkv_sliding | 128x512x2048 | 20 | 179.5 (9.0) | 3.9 % | 44 | Matmul | L1/L1 | memory | 44.8 | 4.0x |
| o_sliding | 128x1024x512 | 20 | 140.1 (7.0) | 3.1 % | 32 | Matmul | L1/L1 | memory | 22.5 | 6.2x |
| ple_gate | 128x512x512 | 24 | 106.1 (4.4) | 2.3 % | 32 | Matmul | L1/L1 | memory | 13.7 | 7.8x |
| ple_proj | 128x512x512 | 24 | 106.0 (4.4) | 2.3 % | 32 | Matmul | L1/L1 | memory | 13.7 | 7.8x |
| o_full | 128x2048x512 | 4 | 49.2 (12.3) | 1.1 % | 32 | Matmul | L1/L1 | memory | 9.0 | 5.5x |
| qkv_full | 128x512x3072 | 4 | 46.3 (11.6) | 1.0 % | 48 | Matmul | L1/L1 | memory | 13.4 | 3.5x |
| ple_wide | 128x512x12288 | 1 | 44.7 (44.7) | 1.0 % | 48 | Matmul | DRAM/DRAM | memory | 19.5 | 2.3x |
| final | 128x512x768 | 1 | 5.2 (5.2) | 0.1 % | 48 | Matmul | L1/DRAM | memory | 1.2 | 4.3x |
| pooler | 32x128x768 | 1 | 3.7 (3.7) | 0.1 % | 24 | Matmul | DRAM/DRAM | memory | 0.6 | 6.3x |

#### b1_s256: kernel sum 5.565 ms, matmul 1.461 ms (26.3 %)

| role | M x K x N | calls | device us (per call) | share of forward | cores | family | in/out mem | bound (model) | floor us | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| down | 256x2048x512 | 24 | 304.4 (12.7) | 5.5 % | 64 | Matmul | L1/L1 | memory | 55.3 | 5.5x |
| up | 256x512x2048 | 24 | 222.2 (9.3) | 4.0 % | 88 | Matmul | L1/L1 | memory | 55.3 | 4.0x |
| gate | 256x512x2048 | 24 | 221.4 (9.2) | 4.0 % | 88 | Matmul | L1/L1 | memory | 55.3 | 4.0x |
| qkv_sliding | 256x512x2048 | 20 | 185.2 (9.3) | 3.3 % | 88 | Matmul | L1/L1 | memory | 46.0 | 4.0x |
| o_sliding | 256x1024x512 | 20 | 144.5 (7.2) | 2.6 % | 64 | Matmul | L1/L1 | memory | 23.3 | 6.2x |
| ple_proj | 256x512x512 | 24 | 109.5 (4.6) | 2.0 % | 64 | Matmul | L1/L1 | memory | 14.3 | 7.7x |
| ple_gate | 256x512x512 | 24 | 108.9 (4.5) | 2.0 % | 64 | Matmul | L1/L1 | memory | 14.3 | 7.6x |
| ple_wide | 256x512x12288 | 1 | 54.3 (54.3) | 1.0 % | 96 | Matmul | DRAM/DRAM | memory | 25.9 | 2.1x |
| o_full | 256x2048x512 | 4 | 52.1 (13.0) | 0.9 % | 64 | Matmul | L1/L1 | memory | 9.2 | 5.7x |
| qkv_full | 256x512x3072 | 4 | 47.2 (11.8) | 0.8 % | 96 | Matmul | L1/L1 | memory | 13.8 | 3.4x |
| final | 256x512x768 | 1 | 6.1 (6.1) | 0.1 % | 96 | Matmul | L1/DRAM | memory | 1.6 | 3.8x |
| pooler | 32x256x768 | 1 | 5.6 (5.6) | 0.1 % | 24 | Matmul | DRAM/DRAM | memory | 1.0 | 5.6x |

#### b1_s512: kernel sum 6.944 ms, matmul 1.803 ms (26.0 %)

| role | M x K x N | calls | device us (per call) | share of forward | cores | family | in/out mem | bound (model) | floor us | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| down | 512x2048x512 | 24 | 361.1 (15.0) | 5.2 % | 64 | Matmul | L1/L1 | compute | 77.7 | 4.6x |
| up | 512x512x2048 | 24 | 271.7 (11.3) | 3.9 % | 88 | Matmul | L1/L1 | compute | 77.7 | 3.5x |
| gate | 512x512x2048 | 24 | 271.3 (11.3) | 3.9 % | 88 | Matmul | L1/L1 | compute | 77.7 | 3.5x |
| qkv_sliding | 512x512x2048 | 20 | 226.0 (11.3) | 3.3 % | 88 | Matmul | L1/L1 | compute | 64.7 | 3.5x |
| o_sliding | 512x1024x512 | 20 | 176.6 (8.8) | 2.5 % | 64 | Matmul | L1/L1 | compute | 32.4 | 5.5x |
| ple_proj | 512x512x512 | 24 | 141.1 (5.9) | 2.0 % | 64 | Matmul | L1/L1 | compute | 19.4 | 7.3x |
| ple_gate | 512x512x512 | 24 | 139.9 (5.8) | 2.0 % | 64 | Matmul | L1/L1 | compute | 19.4 | 7.2x |
| ple_wide | 512x512x12288 | 1 | 79.7 (79.7) | 1.1 % | 96 | Matmul | DRAM/DRAM | memory | 38.7 | 2.1x |
| o_full | 512x2048x512 | 4 | 60.1 (15.0) | 0.9 % | 64 | Matmul | L1/L1 | compute | 12.9 | 4.6x |
| qkv_full | 512x512x3072 | 4 | 58.0 (14.5) | 0.8 % | 96 | Matmul | L1/L1 | compute | 19.4 | 3.0x |
| pooler | 32x512x768 | 1 | 8.9 (8.9) | 0.1 % | 24 | Matmul | DRAM/DRAM | memory | 1.8 | 4.9x |
| final | 512x512x768 | 1 | 8.6 (8.6) | 0.1 % | 96 | Matmul | L1/DRAM | memory | 2.4 | 3.6x |

#### b1_s1024: kernel sum 11.446 ms, matmul 2.924 ms (25.5 %)

| role | M x K x N | calls | device us (per call) | share of forward | cores | family | in/out mem | bound (model) | floor us | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| down | 1024x2048x512 | 24 | 593.4 (24.7) | 5.2 % | 64 | Matmul | L1/L1 | compute | 155.3 | 3.8x |
| gate | 1024x512x2048 | 24 | 430.4 (17.9) | 3.8 % | 88 | Matmul | L1/L1 | compute | 155.3 | 2.8x |
| up | 1024x512x2048 | 24 | 425.7 (17.7) | 3.7 % | 88 | Matmul | L1/L1 | compute | 155.3 | 2.7x |
| qkv_sliding | 1024x512x2048 | 20 | 348.1 (17.4) | 3.0 % | 88 | Matmul | L1/L1 | compute | 129.5 | 2.7x |
| o_sliding | 1024x1024x512 | 20 | 283.3 (14.2) | 2.5 % | 64 | Matmul | L1/L1 | compute | 64.7 | 4.4x |
| ple_proj | 1024x512x512 | 24 | 219.9 (9.2) | 1.9 % | 64 | Matmul | L1/L1 | compute | 38.8 | 5.7x |
| ple_gate | 1024x512x512 | 24 | 218.9 (9.1) | 1.9 % | 64 | Matmul | L1/L1 | compute | 38.8 | 5.6x |
| ple_wide | 1024x512x12288 | 1 | 187.9 (187.9) | 1.6 % | 96 | Matmul | DRAM/DRAM | memory | 64.3 | 2.9x |
| o_full | 1024x2048x512 | 4 | 96.7 (24.2) | 0.8 % | 64 | Matmul | L1/L1 | compute | 25.9 | 3.7x |
| qkv_full | 1024x512x3072 | 4 | 90.1 (22.5) | 0.8 % | 96 | Matmul | L1/L1 | compute | 38.8 | 2.3x |
| pooler | 32x1024x768 | 1 | 15.8 (15.8) | 0.1 % | 24 | Matmul | DRAM/DRAM | memory | 3.4 | 4.7x |
| final | 1024x512x768 | 1 | 14.0 (14.0) | 0.1 % | 96 | Matmul | L1/DRAM | memory | 4.0 | 3.5x |

#### b1_s2048: kernel sum 24.875 ms, matmul 8.126 ms (32.7 %)

| role | M x K x N | calls | device us (per call) | share of forward | cores | family | in/out mem | bound (model) | floor us | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| gate | 2048x512x2048 | 24 | 1853.0 (77.2) | 7.4 % | 110 | Matmul +gelu | L1/DRAM | memory | 450.3 | 4.1x |
| down | 2048x2048x512 | 24 | 1210.6 (50.4) | 4.9 % | 120 | MinimalMatmul | DRAM/L1 | memory | 450.3 | 2.7x |
| up | 2048x512x2048 | 24 | 1168.6 (48.7) | 4.7 % | 110 | Matmul | L1/DRAM | memory | 450.3 | 2.6x |
| qkv_sliding | 2048x512x2048 | 20 | 972.5 (48.6) | 3.9 % | 110 | Matmul | L1/DRAM | memory | 375.2 | 2.6x |
| o_sliding | 2048x1024x512 | 20 | 732.5 (36.6) | 2.9 % | 120 | MinimalMatmul | L1/L1 | compute | 129.5 | 5.7x |
| ple_gate | 2048x512x512 | 24 | 729.6 (30.4) | 2.9 % | 80 | Matmul +gelu | L1/L1 | compute | 77.7 | 9.4x |
| ple_proj | 2048x512x512 | 24 | 622.2 (25.9) | 2.5 % | 120 | MinimalMatmul | L1/L1 | compute | 77.7 | 8.0x |
| ple_wide | 2048x512x12288 | 1 | 296.3 (296.3) | 1.2 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 115.5 | 2.6x |
| qkv_full | 2048x512x3072 | 4 | 277.9 (69.5) | 1.1 % | 120 | Matmul | L1/DRAM | memory | 112.2 | 2.5x |
| o_full | 2048x2048x512 | 4 | 204.0 (51.0) | 0.8 % | 120 | MinimalMatmul | DRAM/L1 | memory | 75.0 | 2.7x |
| pooler | 32x2048x768 | 1 | 29.7 (29.7) | 0.1 % | 24 | Matmul | DRAM/DRAM | memory | 6.6 | 4.5x |
| final | 2048x512x768 | 1 | 28.6 (28.6) | 0.1 % | 120 | MinimalMatmul | L1/DRAM | memory | 7.2 | 4.0x |

#### b1_s4096: kernel sum 43.727 ms, matmul 14.787 ms (33.8 %)

| role | M x K x N | calls | device us (per call) | share of forward | cores | family | in/out mem | bound (model) | floor us | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| gate | 4096x512x2048 | 24 | 3490.5 (145.4) | 8.0 % | 110 | Matmul +gelu | L1/DRAM | memory | 848.4 | 4.1x |
| up | 4096x512x2048 | 24 | 2322.4 (96.8) | 5.3 % | 110 | Matmul | L1/DRAM | memory | 848.4 | 2.7x |
| down | 4096x2048x512 | 24 | 2041.8 (85.1) | 4.7 % | 120 | MinimalMatmul | DRAM/L1 | memory | 848.4 | 2.4x |
| qkv_sliding | 4096x512x2048 | 20 | 1932.9 (96.6) | 4.4 % | 110 | Matmul | L1/DRAM | memory | 707.0 | 2.7x |
| ple_gate | 4096x512x512 | 24 | 1299.2 (54.1) | 3.0 % | 80 | Matmul +gelu | L1/L1 | compute | 155.3 | 8.4x |
| o_sliding | 4096x1024x512 | 20 | 1137.8 (56.9) | 2.6 % | 120 | MinimalMatmul | DRAM/L1 | memory | 357.5 | 3.2x |
| ple_proj | 4096x512x512 | 24 | 1033.1 (43.0) | 2.4 % | 120 | MinimalMatmul | L1/L1 | compute | 155.3 | 6.7x |
| qkv_full | 4096x512x3072 | 4 | 540.5 (135.1) | 1.2 % | 120 | Matmul | L1/DRAM | memory | 211.3 | 2.6x |
| ple_wide | 4096x512x12288 | 1 | 535.6 (535.6) | 1.2 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 217.9 | 2.5x |
| o_full | 4096x2048x512 | 4 | 343.3 (85.8) | 0.8 % | 120 | MinimalMatmul | DRAM/L1 | memory | 141.4 | 2.4x |
| pooler | 32x4096x768 | 1 | 57.8 (57.8) | 0.1 % | 24 | Matmul | DRAM/DRAM | memory | 13.0 | 4.5x |
| final | 4096x512x768 | 1 | 52.2 (52.2) | 0.1 % | 120 | MinimalMatmul | L1/DRAM | memory | 13.5 | 3.9x |

#### b1_s8192: kernel sum 99.983 ms, matmul 31.791 ms (31.8 %)

| role | M x K x N | calls | device us (per call) | share of forward | cores | family | in/out mem | bound (model) | floor us | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| gate | 8192x512x2048 | 24 | 9988.2 (416.2) | 10.0 % | 120 | MinimalMatmul +gelu | DRAM/DRAM | memory | 2018.3 | 4.9x |
| up | 8192x512x2048 | 24 | 4510.4 (187.9) | 4.5 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 2018.3 | 2.2x |
| qkv_sliding | 8192x512x2048 | 20 | 3745.0 (187.3) | 3.7 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 1681.9 | 2.2x |
| down | 8192x2048x512 | 24 | 3484.8 (145.2) | 3.5 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 2018.3 | 1.7x |
| ple_gate | 8192x512x512 | 24 | 3390.0 (141.2) | 3.4 % | 80 | Matmul +gelu | DRAM/DRAM | memory | 799.5 | 4.2x |
| o_sliding | 8192x1024x512 | 20 | 1748.3 (87.4) | 1.7 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 1004.8 | 1.7x |
| ple_layer | 8192x512x512 | 24 | 1526.6 (63.6) | 1.5 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 799.5 | 1.9x |
| ple_proj | 8192x512x512 | 24 | 1522.1 (63.4) | 1.5 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 799.5 | 1.9x |
| qkv_full | 8192x512x3072 | 4 | 1091.1 (272.8) | 1.1 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 471.8 | 2.3x |
| o_full | 8192x2048x512 | 4 | 585.3 (146.3) | 0.6 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 336.4 | 1.7x |
| pooler | 32x8192x768 | 1 | 113.7 (113.7) | 0.1 % | 24 | Matmul | DRAM/DRAM | memory | 25.8 | 4.4x |
| final | 8192x512x768 | 1 | 85.5 (85.5) | 0.1 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 41.8 | 2.0x |

#### b1_s16384: kernel sum 211.306 ms, matmul 60.309 ms (28.5 %)

| role | M x K x N | calls | device us (per call) | share of forward | cores | family | in/out mem | bound (model) | floor us | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| gate | 16384x512x2048 | 24 | 18654.4 (777.3) | 8.8 % | 120 | MinimalMatmul +gelu | DRAM/DRAM | memory | 3984.4 | 4.7x |
| up | 16384x512x2048 | 24 | 8560.3 (356.7) | 4.1 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 3984.4 | 2.1x |
| ple_gate | 16384x512x512 | 24 | 7112.9 (296.4) | 3.4 % | 80 | Matmul +gelu | DRAM/DRAM | memory | 1585.9 | 4.5x |
| qkv_sliding | 16384x512x2048 | 20 | 7091.8 (354.6) | 3.4 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 3320.3 | 2.1x |
| down | 16384x2048x512 | 24 | 6586.5 (274.4) | 3.1 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 3984.4 | 1.7x |
| o_sliding | 16384x1024x512 | 20 | 3079.6 (154.0) | 1.5 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 1987.8 | 1.5x |
| ple_layer | 16384x512x512 | 24 | 2827.2 (117.8) | 1.3 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 1585.9 | 1.8x |
| ple_proj | 16384x512x512 | 24 | 2825.9 (117.7) | 1.3 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 1585.9 | 1.8x |
| qkv_full | 16384x512x3072 | 4 | 2083.2 (520.8) | 1.0 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 930.6 | 2.2x |
| o_full | 16384x2048x512 | 4 | 1102.2 (275.5) | 0.5 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 664.1 | 1.7x |
| pooler | 32x16384x768 | 1 | 225.6 (225.6) | 0.1 % | 24 | Matmul | DRAM/DRAM | memory | 51.4 | 4.4x |
| final | 16384x512x768 | 1 | 159.4 (159.4) | 0.1 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 82.7 | 1.9x |

#### b1_s32768: kernel sum 512.119 ms, matmul 125.212 ms (24.4 %)

| role | M x K x N | calls | device us (per call) | share of forward | cores | family | in/out mem | bound (model) | floor us | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| gate | 32768x512x2048 | 24 | 32876.5 (1369.9) | 6.4 % | 120 | MinimalMatmul +gelu | DRAM/DRAM | memory | 7916.5 | 4.2x |
| ple_gate | 32768x512x512 | 24 | 26518.4 (1104.9) | 5.2 % | 120 | MinimalMatmul +gelu | DRAM/DRAM | memory | 3158.8 | 8.4x |
| up | 32768x512x2048 | 24 | 16333.1 (680.5) | 3.2 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 7916.5 | 2.1x |
| qkv_sliding | 32768x512x2048 | 20 | 13618.4 (680.9) | 2.7 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 6597.1 | 2.1x |
| down | 32768x2048x512 | 24 | 12536.3 (522.3) | 2.4 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 7916.5 | 1.6x |
| o_sliding | 32768x1024x512 | 20 | 5655.9 (282.8) | 1.1 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 3953.9 | 1.4x |
| ple_proj | 32768x512x512 | 24 | 5418.6 (225.8) | 1.1 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 3158.8 | 1.7x |
| ple_layer | 32768x512x512 | 24 | 5413.9 (225.6) | 1.1 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 3158.8 | 1.7x |
| qkv_full | 32768x512x3072 | 4 | 4013.2 (1003.3) | 0.8 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 1848.1 | 2.2x |
| o_full | 32768x2048x512 | 4 | 2075.9 (519.0) | 0.4 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 1319.4 | 1.6x |
| pooler | 32x32768x768 | 1 | 450.0 (450.0) | 0.1 % | 24 | Matmul | DRAM/DRAM | memory | 102.6 | 4.4x |
| final | 32768x512x768 | 1 | 301.9 (301.9) | 0.1 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 164.7 | 1.8x |

#### b1_s65536: kernel sum 1420.502 ms, matmul 246.399 ms (17.3 %)

| role | M x K x N | calls | device us (per call) | share of forward | cores | family | in/out mem | bound (model) | floor us | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| gate | 65536x512x2048 | 24 | 64578.0 (2690.8) | 4.5 % | 120 | MinimalMatmul +gelu | DRAM/DRAM | memory | 15780.9 | 4.1x |
| ple_gate | 65536x512x512 | 24 | 52950.1 (2206.3) | 3.7 % | 120 | MinimalMatmul +gelu | DRAM/DRAM | memory | 6304.5 | 8.4x |
| up | 65536x512x2048 | 24 | 32722.5 (1363.4) | 2.3 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 15780.9 | 2.1x |
| qkv_sliding | 65536x512x2048 | 20 | 27173.9 (1358.7) | 1.9 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 13150.7 | 2.1x |
| down | 65536x2048x512 | 24 | 24592.6 (1024.7) | 1.7 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 15780.9 | 1.6x |
| o_sliding | 65536x1024x512 | 20 | 10471.1 (523.6) | 0.7 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 7886.1 | 1.3x |
| ple_layer | 65536x512x512 | 24 | 10220.5 (425.9) | 0.7 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 6304.5 | 1.6x |
| ple_proj | 65536x512x512 | 24 | 10139.8 (422.5) | 0.7 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 6304.5 | 1.6x |
| qkv_full | 65536x512x3072 | 4 | 7934.2 (1983.5) | 0.6 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 3683.1 | 2.2x |
| o_full | 65536x2048x512 | 4 | 4126.9 (1031.7) | 0.3 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 2630.1 | 1.6x |
| pooler | 32x65536x768 | 1 | 904.4 (904.4) | 0.1 % | 24 | Matmul | DRAM/DRAM | memory | 205.0 | 4.4x |
| final | 65536x512x768 | 1 | 584.7 (584.7) | 0.0 % | 120 | MinimalMatmul | DRAM/DRAM | memory | 328.5 | 1.8x |
