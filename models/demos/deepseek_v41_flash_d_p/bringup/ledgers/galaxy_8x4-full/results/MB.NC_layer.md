# MB.NC layer check: the batching branch's C++ ports on model_bringup (5120 chunk, 160 rows a chip)

Written by the gate's op-level tests (no model load; the resident runner's held mesh, BATCH_LAYERS none): every program native vs the Python prepared route it replaces, max abs diff on every output and chip, repeats on fresh buffers, program-cache growth after warm-up. One row a program; host us a warm call (perf_counter, the median of rounds alternating the routes); calls a 5120 chunk estimated from the call sites (layers 0-20, DECODER_SKIP=decode). Model-wide check: task MB.NCL (the 5120 ladder).

| program | host us before (Python prepared) | host us after (native) | calls a 5120 chunk | host ms a 5120 chunk saved | note |
|---|---|---|---|---|---|
| ttnn.grouped_matmul (attn.wo_a, interleaved160) | 61.6 | 21.4 | 0 | 0.000 | not shipped (WOA_BANK) |
| ttnn.grouped_matmul (attn.wo_a, bank160) | 67.9 | 25.6 | 21 | 0.888 | the shipped path (WOA_BANK) |
| ttnn.fabric_reduce_scatter (attn_out640: [640, 5120] bf16 dim 3) | 69.8 | 43.5 | 21 | 0.552 | est. calls |
| ttnn.fabric_reduce_scatter (q_a640: [640, 1792] bf16 dim 3) | 69.3 | 44.7 | 21 | 0.517 | est. calls |
| ttnn.fabric_reduce_scatter (comp640: [640, 1024] bf16 dim 3) | 69.4 | 44.6 | 19 | 0.471 | est. calls |
| ttnn.fabric_reduce_scatter (idx640: [640, 32] bf16 dim 2) | 80.5 | 51.8 | 4 | 0.115 | est. calls |
| ttnn.fabric_reduce_scatter (exact640: [640, 5120] fp32 dim 3) | 171.0 | 69.6 | 0 | 0.000 | not shipped |
| ttnn.fabric_reduce_scatter (moe640: [640, 5120] bf16 dim 2) | 7811.4 | 44.3 | 0 | 0.000 | not shipped |
| ttnn.layer_send (WindowSend.fused, window_fused5120) | 55.2 | 44.7 | 21 | 0.221 | shipped form |
| ttnn.layer_send (WindowSend.__call__, window5120) | 44.4 | 40.2 | 0 | 0.000 | not shipped |
| ttnn.layer_send (SnapshotSend._call_dyn, snap_dyn_own) | 63.1 | 46.8 | 21 | 0.342 | shipped form |
| ttnn.layer_send (SnapshotSend static, snap_static_own) | 63.4 | 49.5 | 0 | 0.000 | not shipped |
| ttnn.layer_send (SnapshotSend._call_dyn, snap_dyn_mixed) | 62.5 | 47.3 | 21 | 0.319 | shipped form |
| ttnn.layer_send (SnapshotSend static, snap_static_mixed) | 69.1 | 52.9 | 0 | 0.000 | not shipped |
| ttnn.pair_pool (pair_pool640: median) | 56.6 | 20.8 | 3 | 0.107 | est. calls: TtCompressor._pool, ratio-2 sources 2 / 8 / 14 |
| ttnn.pack_fp8_rows (pack320, pack640: median) | 79.2 | 31.5 | 4 | 0.191 | est. calls: fp8_rows_tile, index sources 2 / 8 / 14 / 20 |
| ttnn.unpack_fp8_rows (unpack_hist, unpack_chunk: median) | 336.2 | 22.7 | 8 | 2.508 | est. calls: index_keys._decode, history + chunk a source |
| ttnn.ckv_bfp4_rows (rows_hist, rows_tail: median) | 59.4 | 20.8 | 8 | 0.309 | est. calls: history_gather._pack_rows, history + tail a source |
| ttnn.block_max_bf16 (bmax_s5120) | 58.1 | 21.6 | 3 | 0.110 | est. calls on the block path (long history), 0 below INDEXER_BLOCK_SELECT_MIN |
| ttnn.block_gather (gather5120) | 70.5 | 22.1 | 3 | 0.145 | est. calls on the block path (long history), 0 below INDEXER_BLOCK_SELECT_MIN |
| ttnn.block_keys (keys5120) | 88.5 | 32.3 | 3 | 0.169 | est. calls on the block path (long history), 0 below INDEXER_BLOCK_SELECT_MIN |
| ttnn.chunk_program (PageCopy, pc_hist320, pc_tile_bf16, pc_row_u32, pc_slab: median) | 55.0 | 19.3 | 60 | 2.142 | est. calls (the batching branch's NC.7: ~40-80) |
| ttnn.chunk_program (PageCopy x 2 (sp_stride) -> GluedRows (cos + sin), rope5120) | 172.9 | 74.8 | 1 | 0.098 | est. calls |
| ttnn.chunk_program (PageCopy x 2 (sp_stride) -> GluedRows (cos + sin), crope5120) | 164.0 | 69.5 | 2 | 0.189 | est. calls |
| ttnn.chunk_program (IdsReorder, ids5120) | 120.1 | 89.5 | 1 | 0.031 | est. calls |
| total | | | | 9.423 | sections present: gmm, reduce_scatter, sends, lowbit, block_select, chunk_programs |

## ttnn.grouped_matmul (NC.1): native C++ vs the Python prepared route (wo_a)

Written by `models/demos/deepseek_v41_flash_d_p/tests/woa/test_gmm_native_layer.py` (op level on the held mesh, no model load; fixed-seed bfloat8_b inputs of the real shapes, the model's knobs, dtypes, config and grid). Model-wide verification: task MB.NCL.

| case | knobs | mismatch | repeat | cache growth (C++) | Python LRU growth | host us native | host us native_direct | host us generic | device us native | device us generic |
|---|---|---|---|---|---|---|---|---|---|---|
| interleaved160 | block_w 2, rt 1, depth 4 | 0.0 | 0.0 | 0 | 0 | 21.4 | 20.6 | 61.6 | 127.21 | 127.1 |
| bank160 | block_w 2, rt 1, depth 4, w_inflight 2, x_ahead 1, one_group True | 0.0 | 0.0 | 0 | 0 | 25.6 | 24.5 | 67.9 | 108.37 | 108.13 |

Output dtype DataType.BFLOAT8_B, grid 11-10. Host us: a warm call with the output buffer passed (as attention._out_wo), perf_counter over 50 calls, the median of 5 rounds alternating the routes. native = ttnn.bringup.grouped_matmul (the model's entry, the C++ op under the ttnn.bringup registration), native_direct = ttnn.grouped_matmul. Device us: real-time profiler, slowest chip, 20 calls a section.

## ttnn.fabric_reduce_scatter (NC.2): native C++ vs the Python prepared route

Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_reduce_scatter.py` (op level on the held mesh, no model load; fixed-seed partials of the model's shapes and the callers' options). Model-wide verification: task MB.NCL.

| case | partial a chip | options | mismatch | repeat | cache growth (C++) | host us native | host us native_direct | host us generic | device us native | device us generic |
|---|---|---|---|---|---|---|---|---|---|---|
| attn_out640 | [640, 5120] bf16 dim 3 | cluster_axis 1, num_links 2, dim 3, cache_descriptors True, program_hash True, prepared True | 0.0 | 0.0 | 0 | 43.5 | 43.2 | 69.8 | 117.45 | 117.66 |
| q_a640 | [640, 1792] bf16 dim 3 | cluster_axis 1, num_links 2, dim 3, cache_descriptors True, program_hash True, prepared True | 0.0 | 0.0 | 0 | 44.7 | 43.8 | 69.3 | 48.92 | 66.79 |
| comp640 | [640, 1024] bf16 dim 3 | cluster_axis 1, num_links 2, dim 3, cache_descriptors True, program_hash True, prepared True | 0.0 | 0.0 | 0 | 44.6 | 43.7 | 69.4 | 45.76 | 54.46 |
| idx640 | [640, 32] bf16 dim 2 | cluster_axis 1, num_links 2, dim 2, cache_descriptors True, program_hash True, prepared True | 0.0 | 0.0 | 0 | 51.8 | 50.4 | 80.5 | 39.38 | 45.62 |
| exact640 | [640, 5120] fp32 dim 3 | cluster_axis 1, num_links 2, dim 3, cache_descriptors True, program_hash True, prepared True, exact_fp32_add True | 0.0 | 0.0 | 0 | 69.6 | 107.9 | 171.0 | 303.59 | 304.28 |
| moe640 | [640, 5120] bf16 dim 2 | cluster_axis 1, topology Topology.Ring, num_links 2 | 0.0 | 0.0 | 0 | 44.3 | 43.0 | 7811.4 | 92.11 | 109.61 |

Host us: a warm call with the output buffer passed, perf_counter over 50 calls, the median of 5 rounds alternating the routes. native = ttnn.bringup.fabric_reduce_scatter (the model's entry), native_direct = ttnn.fabric_reduce_scatter (the C++ op), generic = the pre-port Python route (cached descriptors, prepared programs). Device us: real-time profiler, slowest chip, 20 calls a section.

## ttnn.layer_send (NC.3): the per-layer fabric sends, native C++ vs the Python prepared route

Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_sends.py` (op level on the held mesh, no model load; fixed-seed bytes in the model's io shapes and the callers' arguments). Model-wide verification is task MB.NCL.

| case | program | mismatch | repeat | cache growth (C++) | host us python (before) | host us native (after) | device us python | device us native | calls a chunk | host ms a chunk saved |
|---|---|---|---|---|---|---|---|---|---|---|
| window_fused5120 | WindowSend.fused | 0.0 | 0.0 | 0 | 55.2 | 44.7 | 39.55 | 39.78 | 21 | 0.22 |
| window5120 | WindowSend.__call__ | 0.0 | 0.0 | 0 | 44.4 | 40.2 | 38.76 | 39.55 | 0 | 0.00 |
| snap_dyn_own | SnapshotSend._call_dyn | 0.0 | 0.0 | 0 | 63.1 | 46.8 | 59.59 | 47.35 | 21 | 0.34 |
| snap_static_own | SnapshotSend static | 0.0 | 0.0 | 0 | 63.4 | 49.5 | 61.03 | 50.68 | 0 | 0.00 |
| snap_dyn_mixed | SnapshotSend._call_dyn | 0.0 | 0.0 | 0 | 62.5 | 47.3 | 60.75 | 47.58 | 21 | 0.32 |
| snap_static_mixed | SnapshotSend static | 0.0 | 0.0 | 0 | 69.1 | 52.9 | 58.82 | 51.98 | 0 | 0.00 |

Host us: a warm call, perf_counter over 40 calls, the median of 5 rounds alternating the routes (python = the class's prepared route: ttnn.bringup.prepare_program / run_prepared; native = ttnn.layer_send). Device us: real-time profiler, slowest chip, 20 calls a section. Calls a chunk: one a layer for layers 0-20 of the shipped forms (ATTN_FUSE=on: the fused window write; POS_DYNAMIC=on: the dynamic snapshot send); the other forms 0.

## ttnn.pair_pool / pack_fp8_rows / unpack_fp8_rows / ckv_bfp4_rows (NC.5): native C++ vs the Python prepared routes

Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_lowbit.py` (op level on the held mesh, no model load; fixed-seed data of the model's shapes and the callers' options). Model-wide verification is task MB.NCL.

| case | program | call | mismatch | repeat | cache growth (C++) | host us generic (before) | host us native (after) | host us native_direct | device us generic | device us native |
|---|---|---|---|---|---|---|---|---|---|---|
| pair_pool640 | ttnn.pair_pool | x [640, 1024] fp32 -> [320, 512] | 0.0 | 0.0 | 0 | 56.6 | 20.8 | 20.1 | 76.26 | 76.27 |
| pack320 | ttnn.pack_fp8_rows | [320, 128] e4m3 + scale -> [320, 192] uint8 | 0.0 | 0.0 | 0 | 79.1 | 31.8 | 30.1 | 10.08 | 10.09 |
| pack640 | ttnn.pack_fp8_rows | [640, 128] e4m3 + scale -> [640, 192] uint8 | 0.0 | 0.0 | 0 | 79.4 | 31.1 | 30.7 | 11.28 | 11.3 |
| unpack_hist | ttnn.unpack_fp8_rows | rows [3, 1, 4096, 192] ND (shard 16) -> [4096, 128] bfp8 | 0.0 | 0.0 | 0 | 335.9 | 22.9 | 21.7 | 23.85 | 23.93 |
| unpack_chunk | ttnn.unpack_fp8_rows | rows [3, 1, 4096, 192] ND (shard 16) -> [4096, 128] bfp8 | 0.0 | 0.0 | 0 | 336.5 | 22.5 | 21.5 | 20.77 | 20.77 |
| rows_hist | ttnn.ckv_bfp4_rows | slab [3, 1, 4096, 512] bf16 ND (shard 16) -> [4096, 320] uint8 | 0.0 | 0.0 | 0 | 59.4 | 20.8 | 19.9 | 25.15 | 25.23 |
| rows_tail | ttnn.ckv_bfp4_rows | slab [3, 1, 4096, 512] bf16 ND (shard 16) -> [4096, 320] uint8 | 0.0 | 0.0 | 0 | 59.4 | 20.8 | 20.1 | 20.5 | 20.49 |

| program | calls a single 5120 chunk (est.) | host us saved a call (median of its cases) | host ms a chunk saved |
|---|---|---|---|
| ttnn.pair_pool | 3 (TtCompressor._pool, ratio-2 sources 2 / 8 / 14) | 35.8 | 0.107 |
| ttnn.pack_fp8_rows | 4 (fp8_rows_tile, index sources 2 / 8 / 14 / 20) | 47.8 | 0.191 |
| ttnn.unpack_fp8_rows | 8 (index_keys._decode, history + chunk a source) | 313.5 | 2.508 |
| ttnn.ckv_bfp4_rows | 8 (history_gather._pack_rows, history + tail a source) | 38.6 | 0.309 |
| total | | | 3.115 |

Host us: a warm call, perf_counter over 40 calls, the median of 5 rounds alternating the routes. native = ttnn.bringup.<op> (the model's entry), native_direct = ttnn.<op> (the C++ op), generic = the pre-port Python route (prepared programs). Device us: real-time profiler, slowest chip, 20 calls a section.

## ttnn.block_max_bf16 / block_gather / block_keys (NC.6): native C++ vs the Python prepared routes

Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_block_select.py` (op level on the held mesh, no model load; fixed-seed data of the model's shapes and the callers' options). Model-wide verification: task MB.NCL.

| case | program | call | mismatch | repeat | cache growth (C++) | host us generic (before) | host us native (after) | host us native_direct | device us generic | device us native |
|---|---|---|---|---|---|---|---|---|---|---|
| bmax_s5120 | ttnn.block_max_bf16 | strip [10200, 256] bf16 -> [160, 16320] bf16, n | 0.0 | 0.0 | 0 | 58.1 | 21.6 | 20.7 | 74.2 | 74.24 |
| gather5120 | ttnn.block_gather | scores [160, 522240] bf16 + ids [160, 512] -> [160, 16384] bf16 | 0.0 | 0.0 | 0 | 70.5 | 22.1 | 21.6 | 135.17 | 135.9 |
| keys5120 | ttnn.block_keys | pos [160, 512] + ids [160, 512] uint32 -> [160, 512] uint32 | 0.0 | 0.0 | 0 | 88.5 | 32.3 | 31.1 | 22.73 | 22.72 |

| program | calls a chunk on the block path (est.) | host us saved a call (median of its shipped cases) | host ms a chunk saved |
|---|---|---|---|
| ttnn.block_max_bf16 | 3 (TtIndexer.block_topk, the strip transpose) | 36.5 | 0.110 |
| ttnn.block_gather | 3 (TtIndexer.block_topk) | 48.4 | 0.145 |
| ttnn.block_keys | 3 (TtIndexer.block_topk) | 56.2 | 0.169 |
| total | | | 0.423 |

Calls a chunk: the 3 plain indexers (layers 2 / 8 / 14 under DECODER_SKIP=decode) take the block path from INDEXER_BLOCK_SELECT_MIN's compressed keys on; below that 0.

Host us: a warm call, perf_counter over 40 calls, the median of 5 rounds alternating the routes. native = ttnn.bringup.<op> (the model's entry), native_direct = ttnn.<op> (the C++ op), generic = the pre-port Python route (prepared programs). Device us: real-time profiler, slowest chip, 20 calls a section.

## ttnn.chunk_program (NC.7): the per-chunk programs, native C++ vs the Python prepared route

Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_chunk_programs.py` (op level on the held mesh, no model load; fixed-seed bytes in the model's io shapes and the callers' arguments). Model-wide verification: task MB.NCL.

| case | program | mismatch | repeat | cache growth (C++) | host us python (before) | host us native (after) | device us python | device us native | calls a 5120 chunk (est.) |
|---|---|---|---|---|---|---|---|---|---|
| pc_hist320 | PageCopy | 0.0 | 0.0 | 0 | 54.8 | 19.3 | 43.79 | 44.08 | 60 |
| pc_tile_bf16 | PageCopy | 0.0 | 0.0 | 0 | 54.8 | 19.3 | 6.81 | 6.8 | 60 |
| pc_row_u32 | PageCopy | 0.0 | 0.0 | 0 | 55.4 | 19.3 | 1.31 | 1.31 | 60 |
| pc_slab | PageCopy | 0.0 | 0.0 | 0 | 55.1 | 19.8 | 5.14 | 5.14 | 60 |
| rope5120 | PageCopy x 2 (sp_stride) -> GluedRows (cos + sin) | 0.0 | 0.0 | 0 | 172.9 | 74.8 | 6.13 | 5.4 | 1 |
| crope5120 | PageCopy x 2 (sp_stride) -> GluedRows (cos + sin) | 0.0 | 0.0 | 0 | 164.0 | 69.5 | 9.43 | 7.83 | 2 |
| ids5120 | IdsReorder | 0.0 | 0.0 | 0 | 120.1 | 89.5 | 27.09 | 26.97 | 1 |

Host us: a warm call, perf_counter over 40 calls, the median of 5 rounds alternating the routes (python = the class's prepared route: ttnn.bringup.prepare_program / run_prepared; native = ttnn.chunk_program). Device us: real-time profiler, slowest chip, 20 calls a section. rope / crope: a call = the cos and the sin rows of one start (python: two page copies; native: one GLUED_ROWS program).
