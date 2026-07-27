# GLM-4.7-Flash sweep: ISL × batch

## Per-user TPS (tokens/sec per sequence; steady-state when available)

| ISL \ batch | 8 | 16 | 32 |
|---|---|---|---||
| 2048 | OOM/fail | OOM/fail | 12.74 |
| 4096 | OOM/fail | 14.04 | OOM/fail |
| 8192 | 14.73 | OOM/fail | OOM/fail |

## Aggregate TPS (total tokens/sec)

| ISL \ batch | 8 | 16 | 32 |
|---|---|---|---||
| 2048 | OOM/fail | OOM/fail | 407.64 |
| 4096 | OOM/fail | 224.72 | OOM/fail |
| 8192 | 117.82 | OOM/fail | OOM/fail |

## TTFT (time to first token, ms)

| ISL \ batch | 8 | 16 | 32 |
|---|---|---|---||
| 2048 | OOM/fail | OOM/fail | 286559.10 |
| 4096 | OOM/fail | 297141.10 | OOM/fail |
| 8192 | 338301.50 | OOM/fail | OOM/fail |

## Decode latency mean (ms) — ISL × batch

| ISL \ batch | 8 | 16 | 32 |
|---|---|---|---||
| 2048 | OOM/fail | OOM/fail | 78.50 |
| 4096 | OOM/fail | 71.20 | OOM/fail |
| 8192 | 67.90 | OOM/fail | OOM/fail |

## Prefill time (s) — ISL × batch

| ISL \ batch | 8 | 16 | 32 |
|---|---|---|---||
| 2048 | OOM/fail | OOM/fail | 280.82 |
| 4096 | OOM/fail | 291.99 | OOM/fail |
| 8192 | 333.37 | OOM/fail | OOM/fail |

## OOM/Failure Details

| ISL | Batch | Status | Detail |
|---|---|---|---|
| 512 | 128 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 1024 | 64 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 1024 | 128 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 2048 | 64 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 2048 | 128 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 4096 | 32 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 4096 | 64 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 4096 | 128 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 8192 | 16 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 8192 | 32 | exit_-15 | — |
| 8192 | 64 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
| 8192 | 128 | OOM | RuntimeError: TT_FATAL @ /home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false |
