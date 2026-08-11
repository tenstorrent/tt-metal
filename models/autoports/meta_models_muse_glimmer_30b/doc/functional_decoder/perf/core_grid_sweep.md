# Blackhole core-grid sweep — decoder SDPA call sites

Every row uses the layer's own compute-kernel config (HiFi4, `math_approx_mode=False`,
`fp32_dest_acc_en=True`, `packer_l1_acc=True`), so the ranking holds for the policy the
model actually runs.

Measured with `scripts/sweep_core_grids.py` on one Blackhole p300c chip
(11 x 10 compute grid) at the layer's real shapes: 32 Q heads / 2 KV heads,
head_dim 128, sliding window 2048, prefill seq 8192,
decode context 4096, block size 64.
Wall clock over 10 calls after 3 warmup calls, device synchronized.

Best configuration per call site:

| call site (batch) | grid | q_chunk | k_chunk | ms/call |
|---|---|---|---|---|
| decode_sdpa_full (batch 1) | 4x4 | 32 | 128 | 0.050 |
| decode_sdpa_full (batch 32) | 11x8 | 32 | 64 | 0.379 |
| decode_sdpa_sliding (batch 1) | 8x4 | 32 | 256 | 0.041 |
| decode_sdpa_sliding (batch 32) | 11x8 | 32 | 64 | 0.206 |
| prefill_sdpa_chunked (batch 1) | 11x10 | 256 | 256 | 7.546 |
| prefill_sdpa_sliding (batch 1) | 11x10 | 256 | 256 | 7.459 |

Full grid (11x10) vs the Wormhole-shaped 8x8, same chunk sizes:

| call site | q_chunk | k_chunk | 11x10 ms | 8x8 ms |
|---|---|---|---|---|
| decode_sdpa_full | 32 | 32 | 0.063 | 0.060 |
| decode_sdpa_full | 32 | 64 | 0.067 | 0.052 |
| decode_sdpa_full | 32 | 128 | 0.055 | 0.053 |
| decode_sdpa_full | 32 | 256 | 0.057 | 0.056 |
| decode_sdpa_sliding | 32 | 32 | 0.052 | 0.048 |
| decode_sdpa_sliding | 32 | 64 | 0.052 | 0.045 |
| decode_sdpa_sliding | 32 | 128 | 0.051 | 0.045 |
| decode_sdpa_sliding | 32 | 256 | 0.051 | 0.042 |
| prefill_sdpa_chunked | 128 | 128 | 12.156 | 13.768 |
| prefill_sdpa_chunked | 256 | 128 | 8.436 | 12.160 |
| prefill_sdpa_chunked | 256 | 256 | 7.546 | 10.613 |
| prefill_sdpa_chunked | 512 | 128 | 10.314 | 12.390 |
| prefill_sdpa_chunked | 512 | 256 | 8.506 | 10.784 |
| prefill_sdpa_sliding | 128 | 64 | 12.765 | 16.526 |
| prefill_sdpa_sliding | 128 | 128 | 12.023 | 13.396 |
| prefill_sdpa_sliding | 256 | 128 | 8.587 | 12.546 |
| prefill_sdpa_sliding | 256 | 256 | 7.459 | 11.004 |
| prefill_sdpa_sliding | 512 | 128 | 10.344 | 12.603 |

Rejected combinations (op raised):

* `prefill_sdpa_sliding` grid 11x10 q1024/k128: RuntimeError
* `prefill_sdpa_sliding` grid 11x10 q512/k256: RuntimeError
* `prefill_sdpa_sliding` grid 11x8 q1024/k128: RuntimeError
* `prefill_sdpa_sliding` grid 11x8 q512/k256: RuntimeError
* `prefill_sdpa_sliding` grid 10x10 q1024/k128: RuntimeError
* `prefill_sdpa_sliding` grid 10x10 q512/k256: RuntimeError
* `prefill_sdpa_sliding` grid 8x10 q1024/k128: RuntimeError
* `prefill_sdpa_sliding` grid 8x10 q512/k256: RuntimeError
* `prefill_sdpa_sliding` grid 11x5 q1024/k128: RuntimeError
* `prefill_sdpa_sliding` grid 11x5 q512/k256: RuntimeError
* `prefill_sdpa_sliding` grid 8x8 q1024/k128: RuntimeError
* `prefill_sdpa_sliding` grid 8x8 q512/k256: RuntimeError
* `prefill_sdpa_sliding` grid 8x4 q1024/k128: RuntimeError
* `prefill_sdpa_sliding` grid 8x4 q512/k256: RuntimeError
* `prefill_sdpa_sliding` grid 4x4 q1024/k128: RuntimeError
* `prefill_sdpa_sliding` grid 4x4 q512/k256: RuntimeError
* `prefill_sdpa_chunked` grid 11x10 q1024/k128: RuntimeError
* `prefill_sdpa_chunked` grid 11x8 q1024/k128: RuntimeError
* `prefill_sdpa_chunked` grid 10x10 q1024/k128: RuntimeError
* `prefill_sdpa_chunked` grid 8x10 q1024/k128: RuntimeError
* `prefill_sdpa_chunked` grid 11x5 q1024/k128: RuntimeError
* `prefill_sdpa_chunked` grid 8x8 q1024/k128: RuntimeError
* `prefill_sdpa_chunked` grid 8x4 q1024/k128: RuntimeError
* `prefill_sdpa_chunked` grid 4x4 q1024/k128: RuntimeError
* `decode_sdpa_sliding` grid 11x5 q32/k32: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_full` grid 11x5 q32/k32: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_sliding` grid 11x5 q32/k64: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_full` grid 11x5 q32/k64: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_sliding` grid 11x5 q32/k128: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_full` grid 11x5 q32/k128: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_sliding` grid 11x5 q32/k256: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_full` grid 11x5 q32/k256: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_sliding` grid 8x4 q32/k32: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_full` grid 8x4 q32/k32: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_sliding` grid 8x4 q32/k64: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_full` grid 8x4 q32/k64: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_sliding` grid 8x4 q32/k128: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_full` grid 8x4 q32/k128: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_sliding` grid 8x4 q32/k256: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_full` grid 8x4 q32/k256: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_sliding` grid 4x4 q32/k32: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_full` grid 4x4 q32/k32: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_sliding` grid 4x4 q32/k64: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_full` grid 4x4 q32/k64: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_sliding` grid 4x4 q32/k128: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_full` grid 4x4 q32/k128: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_sliding` grid 4x4 q32/k256: illegal_cores_lt_batch_times_kv_heads
* `decode_sdpa_full` grid 4x4 q32/k256: illegal_cores_lt_batch_times_kv_heads

Raw data: `core_grid_sweep.csv`.
