# NC.5 layer check: the low-bit forks' programs as native C++ ops vs the Python prepared routes

Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_lowbit.py` (op level on the held mesh, no model load; fixed-seed data of the model's shapes and the callers' options). Model-wide verification is deferred to the end of the port series (owner, NC.8).

| case | program | call | mismatch | repeat | cache growth (C++) | host us generic (before) | host us native (after) | host us native_direct | device us generic | device us native |
|---|---|---|---|---|---|---|---|---|---|---|
| pair_pool1024 | ttnn.pair_pool | x [1024, 1024] fp32 -> [512, 512] | 0.0 | 0.0 | 0 | 73.3 | 24.1 | 23.7 | 108.25 | 108.38 |
| pair_pool640 | ttnn.pair_pool | x [640, 1024] fp32 -> [320, 512] | 0.0 | 0.0 | 0 | 60.2 | 20.0 | 19.3 | 76.3 | 76.34 |
| pack512 | ttnn.pack_fp8_rows | [512, 128] e4m3 + scale -> [512, 192] uint8 | 0.0 | 0.0 | 0 | 88.7 | 31.9 | 30.6 | 10.67 | 10.68 |
| pack1024 | ttnn.pack_fp8_rows | [1024, 128] e4m3 + scale -> [1024, 192] uint8 | 0.0 | 0.0 | 0 | 83.6 | 30.8 | 29.8 | 12.2 | 12.2 |
| pack320 | ttnn.pack_fp8_rows | [320, 128] e4m3 + scale -> [320, 192] uint8 | 0.0 | 0.0 | 0 | 84.6 | 30.6 | 30.1 | 10.08 | 10.08 |
| pack640 | ttnn.pack_fp8_rows | [640, 128] e4m3 + scale -> [640, 192] uint8 | 0.0 | 0.0 | 0 | 86.5 | 31.6 | 30.4 | 11.29 | 11.28 |
| unpack_hist | ttnn.unpack_fp8_rows | rows [3, 1, 4096, 192] ND (shard 16) -> [4096, 128] bfp8 | 0.0 | 0.0 | 0 | 64.7 | 21.4 | 20.3 | 23.85 | 23.88 |
| unpack_chunk | ttnn.unpack_fp8_rows | rows [3, 1, 4096, 192] ND (shard 16) -> [4096, 128] bfp8 | 0.0 | 0.0 | 0 | 63.8 | 20.4 | 19.6 | 20.74 | 20.75 |
| rows_hist | ttnn.ckv_bfp4_rows | slab [3, 1, 4096, 512] bf16 ND (shard 16) -> [4096, 320] uint8 | 0.0 | 0.0 | 0 | 62.7 | 20.2 | 19.5 | 25.16 | 25.22 |
| rows_tail | ttnn.ckv_bfp4_rows | slab [3, 1, 4096, 512] bf16 ND (shard 16) -> [4096, 320] uint8 | 0.0 | 0.0 | 0 | 58.7 | 20.0 | 19.4 | 20.49 | 20.52 |

| program | calls a single 8192 chunk (est.) | host us saved a call (median of its cases) | host ms a chunk saved |
|---|---|---|---|
| ttnn.pair_pool | 3 (TtCompressor._pool, ratio-2 sources 2 / 8 / 14) | 44.7 | 0.134 |
| ttnn.pack_fp8_rows | 4 (fp8_rows_tile, index sources 2 / 8 / 14 / 20) | 54.4 | 0.218 |
| ttnn.unpack_fp8_rows | 8 (index_keys._decode, history + chunk a source) | 43.4 | 0.347 |
| ttnn.ckv_bfp4_rows | 8 (history_gather._pack_rows, history + tail a source) | 40.6 | 0.325 |
| total | | | 1.024 |

Host us: a warm call, perf_counter over 40 calls, the median of 5 rounds alternating the routes. native = ttnn.bringup.<op> (the model's entry), native_direct = ttnn.<op> (the C++ op), generic = the pre-port Python route (prepared programs). Device us: real-time profiler, slowest chip, 20 calls a section.
