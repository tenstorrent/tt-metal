# DSA indexer: single-chip op shapes (Kimi K3.1, Galaxy 8x4, S = 5120)

Derived statically (nothing was run) from commit `63d761645de`
("Kimi K3.1 sparse-MLA tests; sparse reshard at every TP > 1", #64403).
Code read: `models/demos/deepseek_v3_d_p/tt/mla/indexer.py` (`TtIndexerBase.write_k`, `.score`,
`.select_local`, `.finalize_distribution`), `tt/mla/mla_config.py`, `tt/mla/mla.py`,
`reference/kimi_k3_1_config.py`, `tests/sparse_mla/test_sparse_mla.py`, `tests/test_mla.py`,
`tt/tt_ccl.py`, `utils/kv_cache_utils.py`.

## Test case chosen

`test_sparse_mla_accuracy_chunked[kimi_k3_1-shape-8x4-seq5120-kv_scaled_fp8]`
(main KV format for K3.1; the `kv_bf16` twin runs the same indexer shapes).
One full-sequence chunk at offset 0: `is_chunked=True, active_seq_len=5120, actual_start=0`,
scalar path (`metadata=None`), `actual_end=None`.

## Parameters

| Symbol | Value | Source |
|---|---|---|
| S (global seq / chunk) | 5120 | test |
| mesh | 8 x 4 -> `sp = 8` (axis 0), `tp = 4` (axis 1), 32 chips | `SPARSE_MESH_BY_DEVICES[32]` |
| Sl = S/sp | 640 | `seq_len` the indexer receives (local) |
| Sq = Sl/tp = S/(sp*tp) | 160 (5 tiles) | `sq_local` |
| hidden (EMB_SIZE) | 7168 (1792 per TP rank) | `KimiK3Config` |
| q_lora_rank | 1536 (48 tiles) | `KimiK3Config` |
| num_attention_heads | 96 | gates the matmul config entries |
| index heads H_idx x D_idx | 32 x 128 | `KimiK31Config` |
| index_topk / capacity | 2048 / `min(2048, 5120)` = 2048 | `index_topk_capacity` |
| index_rope_dim | 0 (NoPE indexer) | `KimiK31Config` |
| compute grid | (11, 10) | `COMPUTE_GRID` |

Compute kernel config (`default_compute_kernel_config`, from ttMLA): HiFi2, `math_approx_mode=False`,
`fp32_dest_acc_en=False`, `packer_l1_acc=True`.

Conventions: shapes are per chip. "tiles" = 32x32. Layouts are TILE unless noted.

## Persistent state per chip

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| index key cache | `[1, 1, 160, 128]` | bfloat8_b | 1 user x 1 layer slot; `S/(sp*tp)` rows (KV dedup: striped over SP and TP, block-cyclic). Assumes `num_full_indexer_layers` is None for K3.1 so the slot count falls back to 1. |
| ring K buffer (gathered) | `[1, 1, 5120, 128]` | bfloat8_b | DRAM, replicated, shared scratch (`get_indexer_ring_k_buffer`) |
| `wq_b` | `[1536, 4096]` | bfloat8_b | **replicated** on every chip (all 32 heads) |
| `wk` | `[1792, 128]` | bfloat8_b | sharded on the hidden (input) dim across TP |
| `weights_proj` | `[1792, 32]` | bfloat16 | sharded on hidden dim across TP; scaled once at init by `32**-0.5 * 128**-0.5` via `ttnn.multiply` |
| `k_norm` weight / bias | 128-vector | bfloat16 | LayerNorm with bias; eps 1e-6 |
| Hadamard matrix | `[1, 1, 128, 128]` | bfloat16 | replicated, normalized Sylvester |

## Inputs to the indexer

| Tensor | Shape | Notes |
|---|---|---|
| `hidden_states` | `[1, 1, 640, 1792]` | sharded S over SP, hidden over TP (`test_mla.py` `shard_dims`) |
| `qr` (shared q_a latent) | `[1, 1, 640, 1536]` | already TP all-reduced and layer-normed by ttMLA; replicated across TP |

## Single-chip ops, in call order

Order within `score()`: `write_k` first (K stem), then the Q stem, then scoring (multi-chip), then
`select_local`. Multi-chip steps are shown in italics only to explain where each input comes from.

### K stem (`write_k`)

*`matmul_reduce_scatter_prefill` (wk), multi-chip:* `[1,1,640,1792] x [1792,128]` -> `[1,1,640,32]`
(128/tp columns per chip; no program config is chosen in the indexer for this fused op).
*`high_bw_all_gather` over TP, dim 3:* `[1,1,640,32]` -> `[1,1,640,128]` bf16.

| # | Op | Input shape(s) | Output | Config |
|---|---|---|---|---|
| K1 | `ttnn.layer_norm` | x `[1,1,640,128]` bf16; weight, bias 128-vectors | `[1,1,640,128]` bf16, DRAM | `epsilon=1e-6`, default compute config, `DRAM_MEMORY_CONFIG` |
| K2 | RoPE (`rotary_embedding_indexed`) | - | - | **skipped**: NoPE, `_bc_rope_pe` returns `x` |
| K3 | `ttnn.matmul` k x Hadamard (`indexer.k_hadamard`, key 640) | `[1,1,640,128] x [1,1,128,128]` | `[1,1,640,128]` bfloat8_b, DRAM | `MatmulMultiCoreReuseMultiCastProgramConfig`: grid (11,10), `in0_block_w=4`, `out_subblock=2x1`, `per_core_M=2`, `per_core_N=1`, `transpose_mcast=False`, `fuse_batch=False`. M=20 tiles, K=4, N=4 -> 10 x 4 = 40 cores. Default compute config. |
| K4 | `update_padded_kv_cache` (`deepseek_prefill`) | cache `[1,1,160,128]` bf8; k `[1,1,640,128]` bf8 | in-place write | `slot_idx=0, layer_idx=0, num_layers=1, kv_actual_global=0, cluster_axis=0 (SP), valid_global=None, tp_axis=1`. Each chip writes only its 1/tp window of the 640 rows (160 rows) at the block-cyclic offset. |

### Q stem (`score`)

| # | Op | Input shape(s) | Output | Config |
|---|---|---|---|---|
| Q1 | `ttnn.mesh_partition`, dim 2, `cluster_axis=tp_axis` | `qr [1,1,640,1536]` | `[1,1,160,1536]` | local slice of this TP rank's 160 rows (no data movement) |
| Q2 | `ttnn.linear` (`indexer.wq_b`, key 160) | `[1,1,160,1536] x [1536,4096]` | `[1,1,160,4096]` bf16, **L1** | `MatmulMultiCoreReuseMultiCast1DProgramConfig`: grid (11,10), `in0_block_w=16`, `out_subblock=1x2`, `per_core_M=5`, `per_core_N=2`, `fuse_batch=True`, `mcast_in0=True`; `dtype=bfloat16`, `out_mem_config=L1`. M=5 tiles, K=48, N=128 -> 64 cores (N-parallel). Default compute config. |
| Q3 | `nlp_create_qkv_heads` | `[1,1,160,4096]` | q `[1,32,160,128]` bf16, DRAM | `num_heads=32, num_kv_heads=0, transpose_k_heads=False`, `DRAM_MEMORY_CONFIG`. The two other outputs are unused. |
| Q4 | RoPE on q | - | - | **skipped** (NoPE) |
| Q5 | `ttnn.matmul` q x Hadamard (`indexer.q_hadamard`, key 160) | `[1,32,160,128] x [1,1,128,128]` | `[1,32,160,128]` **bfloat8_b**, DRAM | `MatmulMultiCoreReuseMultiCast1DProgramConfig`: grid (11,10), `in0_block_w=2`, `out_subblock=2x4`, `per_core_M=2`, `per_core_N=4`, `fuse_batch=True`, `mcast_in0=False`. 32 heads x 5 tiles = 160 M-tiles, K=4, N=4 -> 80 cores. Default compute config. |

*`matmul_reduce_scatter_prefill` (weights_proj), multi-chip:* `[1,1,640,1792] x [1792,32]` -> `[1,1,160,32]`
(scatter on the sequence dim), used by the ring op below.

### Scoring (multi-chip, for context)

`ring_indexer_score_dsa`: q `[1,32,160,128]` bf8, k_full `[1,1,5120,128]` bf8, weights `[1,1,160,32]`,
k_local `[1,1,160,128]` -> logits `[1,1,160,5120]` bf16 ROW_MAJOR.
`IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=320, head_group_size=0)`:
`q_chunk_size = 64 if 160 % 64 == 0 else 32` -> 32; `k_chunk_size = min(get_indexer_key_chunk(32)=320, 5120)` = 320.
Ring args: `cluster_axis=None` (full-mesh snake), `chunk_start_idx=0`, `kv_len=5120`, `cache_batch_idx=0`,
`index_cache_num_layers=1`, `index_cache_layer_idx=0`, `block_cyclic_chunk_local=160`,
`block_cyclic_cache_tp_sharded=False`.

### Selection (`select_local`)

| # | Op | Input | Output | Config |
|---|---|---|---|---|
| T1 | `ttnn.experimental.topk_large_indices` | logits `[1,1,160,5120]` bf16 ROW_MAJOR | indices `[1,1,160,2048]` uint32 ROW_MAJOR | `k=2048`, `valid_length=5120` (`end_pos`, scalar path), no metadata tensors, default subdevice / core grid / NoC |

`finalize_distribution` is a no-op here: at TP > 1 K3.1 has `output_tp_sequence_sharded=True`
(head-to-sequence reshard), so `requires_tp_redistribution` is False and the top-k indices stay
`[1,1,160,2048]`. Real pruning applies: 2048 of 5120 keys are selected, causally masked.

## Other single-chip ops (not in the per-chunk chain)

- `ttnn.multiply(weights_proj, 32**-0.5 * 128**-0.5)`: once at init, `[1792,32]`.
- `ttnn.deallocate`: about 21 call sites, memory management only.

## Assumptions and open points

- Scalar path (`metadata=None`) and `actual_end=None`. Under trace, `topk_large_indices` takes
  `valid_length_tensor` / `valid_end_tensor` instead of the scalar `valid_length`.
- Cache slot count of 1 (K3.1 has no `indexer_types`; the test builds a single-layer MLA).
- `layer_norm` weight/bias exact padded shape was not checked (128 values, tile-padded).
- Config entries are picked by `_resolve_mm_cfg`: BH only, `num_heads == 96` and `q_lora_rank == 1536`
  for K3.1 (`_K31_INDEXER_TAGS`). `wk` and `weights_proj` have no per-shape config on TP > 1 because
  they run inside the fused matmul + reduce-scatter op.
- Other meshes change the numbers: Sl = S/sp, Sq = Sl/tp, cache rows = S/(sp*tp), and a `wq_b` /
  `q_hadamard` config is only found where an entry exists for that Sq (160 and 640 are tuned).
- Nothing here was executed on hardware or checked against a profiler run.
