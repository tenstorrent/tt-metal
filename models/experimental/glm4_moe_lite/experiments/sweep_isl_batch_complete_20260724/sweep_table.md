# GLM-4.7-Flash sweep: ISL × batch

## Per-user TPS (tokens/sec per sequence; steady-state when available)

| ISL \ batch | 1 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---||
| 128 | 19.49 | 18.32 | 17.48 | 16.05 | 14.04 | 8.28 | 4.62 |
| 512 | 19.05 | 17.79 | 16.75 | 15.48 | 13.46 | 7.87 | OOM/fail |
| 1024 | 18.73 | 17.30 | 16.45 | 14.93 | 12.74 | OOM/fail | OOM/fail |
| 2048 | 18.55 | 17.01 | 15.80 | 14.08 | OOM/fail | OOM/fail | OOM/fail |
| 4096 | 18.18 | 16.37 | 14.79 | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 8192 | 17.48 | 15.34 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 16384 | 16.23 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 32768 | 14.16 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 65536 | 11.38 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |

## Aggregate TPS (total tokens/sec)

| ISL \ batch | 1 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---||
| 128 | 19.49 | 73.26 | 139.86 | 256.82 | 449.44 | 529.80 | 591.50 |
| 512 | 19.05 | 71.17 | 134.00 | 247.68 | 430.69 | 503.54 | OOM/fail |
| 1024 | 18.73 | 69.20 | 131.58 | 238.81 | 407.64 | OOM/fail | OOM/fail |
| 2048 | 18.55 | 68.03 | 126.38 | 225.35 | OOM/fail | OOM/fail | OOM/fail |
| 4096 | 18.18 | 65.47 | 118.34 | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 8192 | 17.48 | 61.35 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 16384 | 16.23 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 32768 | 14.16 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 65536 | 11.38 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |

## TTFT (time to first token, ms)

| ISL \ batch | 1 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---||
| 128 | 4623.40 | 6498.50 | 9128.40 | 13098.80 | 20649.80 | 38351.30 | 83127.10 |
| 512 | 7298.10 | 13979.20 | 20563.50 | 33452.30 | 61502.60 | 133254.70 | OOM/fail |
| 1024 | 10928.70 | 20518.60 | 34596.20 | 64402.20 | 131937.60 | OOM/fail | OOM/fail |
| 2048 | 13723.30 | 33827.80 | 66853.00 | 134547.20 | OOM/fail | OOM/fail | OOM/fail |
| 4096 | 22247.00 | 71024.20 | 144060.10 | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 8192 | 42328.60 | 162150.80 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 16384 | 104188.40 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 32768 | 297128.90 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 65536 | 911593.90 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |

## Decode latency mean (ms) — ISL × batch

| ISL \ batch | 1 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---||
| 128 | 51.30 | 54.60 | 57.20 | 62.30 | 71.20 | 120.80 | 216.40 |
| 512 | 52.50 | 56.20 | 59.70 | 64.60 | 74.30 | 127.10 | OOM/fail |
| 1024 | 53.40 | 57.80 | 60.80 | 67.00 | 78.50 | OOM/fail | OOM/fail |
| 2048 | 53.90 | 58.80 | 63.30 | 71.00 | OOM/fail | OOM/fail | OOM/fail |
| 4096 | 55.00 | 61.10 | 67.60 | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 8192 | 57.20 | 65.20 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 16384 | 61.60 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 32768 | 70.60 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 65536 | 87.90 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |

## Prefill time (s) — ISL × batch

| ISL \ batch | 1 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---||
| 128 | 1.84 | 3.28 | 5.49 | 9.30 | 16.82 | 32.41 | 73.18 |
| 512 | 4.41 | 9.41 | 16.33 | 29.67 | 57.58 | 128.00 | OOM/fail |
| 1024 | 7.42 | 15.99 | 29.70 | 59.38 | 125.54 | OOM/fail | OOM/fail |
| 2048 | 10.25 | 29.36 | 61.95 | 129.46 | OOM/fail | OOM/fail | OOM/fail |
| 4096 | 18.72 | 66.55 | 139.21 | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 8192 | 38.86 | 157.62 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 16384 | 100.71 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 32768 | 293.60 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |
| 65536 | 908.03 | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail | OOM/fail |

## OOM/Failure Details

| ISL | Batch | Status | Detail |
|---|---|---|---|
| 512 | 128 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 1024 | 64 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 1024 | 128 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 2048 | 32 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 2048 | 64 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 2048 | 128 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 4096 | 16 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 4096 | 32 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 4096 | 64 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 4096 | 128 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 8192 | 8 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 8192 | 16 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 8192 | 32 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 8192 | 64 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 8192 | 128 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
