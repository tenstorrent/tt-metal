# NC.6 layer check: the indexer block-select fork's programs as native C++ ops vs the Python prepared routes

Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_block_select.py` (op level on the held mesh, no model load; fixed-seed data of the model's shapes and the callers' options). Model-wide verification is deferred to the end of the port series (owner, NC.8).

| case | program | call | mismatch | repeat | cache growth (C++) | host us generic (before) | host us native (after) | host us native_direct | device us generic | device us native |
|---|---|---|---|---|---|---|---|---|---|---|
| bmax_s8192 | ttnn.block_max_bf16 | strip [16384, 256] bf16 -> [256, 16384] bf16, n | 0.0 | 0.0 | 0 | 62.7 | 20.8 | 19.9 | 118.02 | 117.89 |
| bmax_p8192 | ttnn.block_max_bf16 | strip [16384, 256] bf16 -> [256, 16384] bf16, n_tensor | 0.0 | 0.0 | 0 | 61.2 | 20.9 | 20.3 | 118.12 | 118.18 |
| bmax_s5120 | ttnn.block_max_bf16 | strip [10200, 256] bf16 -> [160, 16320] bf16, n | 0.0 | 0.0 | 0 | 58.4 | 20.3 | 19.7 | 74.2 | 74.27 |
| bmax_fold | ttnn.block_max_bf16 | [256, 65536] fp32 (factor 4) -> [256, 16384] bf16, n | 0.0 | 0.0 | 0 | 151.7 | 32.2 | 44.2 | 389.35 | 389.39 |
| gather8192 | ttnn.block_gather | scores [256, 524288] bf16 + ids [256, 512] -> [256, 16384] bf16 | 0.0 | 0.0 | 0 | 122.4 | 30.0 | 30.6 | 246.05 | 245.8 |
| gather5120 | ttnn.block_gather | scores [160, 522240] bf16 + ids [160, 512] -> [160, 16384] bf16 | 0.0 | 0.0 | 0 | 79.2 | 21.4 | 20.3 | 135.48 | 135.49 |
| keys8192 | ttnn.block_keys | pos [256, 512] + ids [256, 512] uint32 -> [256, 512] uint32 | 0.0 | 0.0 | 0 | 87.1 | 30.9 | 29.4 | 33.33 | 33.33 |
| keys5120 | ttnn.block_keys | pos [160, 512] + ids [160, 512] uint32 -> [160, 512] uint32 | 0.0 | 0.0 | 0 | 88.3 | 31.1 | 29.7 | 22.71 | 22.71 |

| program | calls a chunk on the block path (est.) | host us saved a call (median of its shipped cases) | host ms a chunk saved |
|---|---|---|---|
| ttnn.block_max_bf16 | 3 (TtIndexer.block_topk, the strip transpose) | 40.3 | 0.121 |
| ttnn.block_gather | 3 (TtIndexer.block_topk) | 75.1 | 0.225 |
| ttnn.block_keys | 3 (TtIndexer.block_topk) | 56.7 | 0.170 |
| total | | | 0.516 |

Calls a chunk: the 3 plain indexers (layers 2 / 8 / 14 under DECODER_SKIP=decode) take the block path from 163840 compressed keys (single) / 196608 (packed glued); below that 0. bmax_fold is INDEXER_BMAX_COMPUTE=off (not shipped): ported with the entry, left out of the saving.

Host us: a warm call, perf_counter over 40 calls, the median of 5 rounds alternating the routes. native = ttnn.bringup.<op> (the model's entry), native_direct = ttnn.<op> (the C++ op), generic = the pre-port Python route (prepared programs). Device us: real-time profiler, slowest chip, 20 calls a section.
