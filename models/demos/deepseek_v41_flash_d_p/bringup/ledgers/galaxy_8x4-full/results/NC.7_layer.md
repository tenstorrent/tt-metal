# NC.7 layer check: the per-chunk programs as the native C++ ttnn.chunk_program vs the Python prepared route

Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_chunk_programs.py` (op level on the held mesh, no model load; fixed-seed bytes in the model's io shapes and the callers' arguments). Model-wide verification is deferred to the end of the port series (owner, NC.8). LaneRows and ChipRows are not on the shipped path (BREADCRUMBS NC.7, OV-82) and were not ported.

| case | program | mismatch | repeat | cache growth (C++) | host us python (before) | host us native (after) | device us python | device us native | calls a chunk (single / packed) |
|---|---|---|---|---|---|---|---|---|---|
| pc_hist320 | PageCopy | 0.0 | 0.0 | 0 | 55.0 | 19.8 | 44.07 | 44.05 | ~40-80 / ~40-80 |
| pc_tile_bf16 | PageCopy | 0.0 | 0.0 | 0 | 55.1 | 19.9 | 6.8 | 6.8 | ~40-80 / ~40-80 |
| pc_row_u32 | PageCopy | 0.0 | 0.0 | 0 | 54.4 | 19.3 | 1.31 | 1.31 | ~40-80 / ~40-80 |
| pc_slab | PageCopy | 0.0 | 0.0 | 0 | 54.2 | 19.6 | 5.13 | 5.13 | ~40-80 / ~40-80 |
| rope8192 | GluedRows x 2 (cos + sin) -> 1 | 0.0 | 0.0 | 0 | 303.2 | 83.0 | 7.81 | 6.85 | 1 pair / 1 pair |
| crope8192 | GluedRows x 2 (cos + sin) -> 1 | 0.0 | 0.0 | 0 | 300.9 | 82.4 | 11.91 | 11.44 | 1-2 pairs / 1-2 pairs |
| rope5120 | GluedRows x 2 (cos + sin) -> 1 | 0.0 | 0.0 | 0 | 328.5 | 90.2 | 6.26 | 5.5 | 1 pair / 1 pair |
| crope5120 | GluedRows x 2 (cos + sin) -> 1 | 0.0 | 0.0 | 0 | 300.8 | 81.7 | 8.34 | 7.84 | 1-2 pairs / 1-2 pairs |
| rope4x2k | GluedRows x 2 (cos + sin) -> 1 | 0.0 | 0.0 | 0 | 306.4 | 86.8 | 8.19 | 7.17 | 1 pair / 1 pair |
| ids8192 | IdsReorder | 0.0 | 0.0 | 0 | 117.6 | 89.0 | 31.51 | 31.51 | 1 / 0 |
| ids5120 | IdsReorder | 0.0 | 0.0 | 0 | 117.7 | 89.0 | 26.99 | 27.24 | 1 / 0 |
| gids4x2k_cap | GluedIdsReorder | 0.0 | 0.0 | 0 | 63.8 | 41.2 | 27.68 | 27.68 | 0 / 1 |
| gids4x2k_end | GluedIdsReorder | 0.0 | 0.0 | 0 | 63.2 | 40.6 | 27.67 | 27.68 | 0 / 1 |
| derive4x2k | Derive (a chunk: every section) | 0.0 | 0.0 | 0 | 500.5 | 246.2 | 284.72 | 219.06 | 0 / 1 chunk |
| owned_hist | OwnedSend | 0.0 | 0.0 | 0 | 107.1 | 88.1 | 159.69 | 159.44 | 0 / ~20 |
| owned_keys | OwnedSend | 0.0 | 0.0 | 0 | 106.6 | 88.1 | 76.57 | 76.76 | 0 / ~20 |

Host us: a warm call, perf_counter over 40 calls, the median of 5 rounds alternating the routes (python = the class's prepared route: ttnn.bringup.prepare_program / run_prepared; native = ttnn.chunk_program). Device us: real-time profiler, slowest chip, 20 calls a section. rope / crope: a call = the cos and the sin rows of one start (python: two programs; native: one). derive4x2k: a call = one chunk (begin + every section's run: python 8 programs; native 1 begin program + region + 2 srv).
