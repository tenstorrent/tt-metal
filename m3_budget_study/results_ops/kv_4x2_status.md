# M3 prefill on a (4,2) sub-mesh: KV correctness and what KV-head sharding needs

SP=4 (axis 0) x TP=2 (axis 1), EP=8, 16 experts per chip. 2 of the 4 K/V heads per chip. Stage 0 of
`create_submeshes(MeshShape(4, 2))` carved from the 8x4 galaxy, 1d fabric, bf4 experts,
M3_MOE_W_NDSHARD=1 M3_MOE_HYBRID_THRESHOLD=128, v1 dispatch/combine.

## Correctness (harness patches, no model change)

`tools/profile_4x2.py` monkeypatches two things. First, `allocate_kv_caches` gets n_kv/TP = 2 K/V heads per chip.
Second, `msa_sp_attention_cache_read` slices the multi-head slot one head at a time to DRAM interleaved, gathers
each head with `high_bw_all_gather`, then concats them.

Test run: layers 0-6, one 5120-token chunk at a 51200-token cache, golden `longbook_56320` (55218 tokens), no tracy.

| run | vs | worst layer | min PCC |
|---|---|---|---:|
| kv_4x2_L0-6 (4,2) | golden | layer 6 V | 0.99524 |
| kv_2x4_L0-6 (2,4) | golden | layer 6 V | 0.99531 |
| (4,2) vs (2,4) dumps | each other | layer 6 V | 0.99854 |

Every layer is within 1e-4 of the (2,4) PCC vs golden, so the (4,2) is correct. Layer 0 is bit-identical to
(2,4) (k=v=1.00000), and the K / V / index_k PCC of layers 1-6 vs (2,4) is >= 0.9985. The per-layer lines are in
logs/kv_4x2_L0-6.log, logs/kv_2x4_L0-6.log and logs/kv_compare_4x2_vs_2x4.log.

## What the KV-head sharding change takes

The model today assumes one K/V head per chip (n_kv == TP == 4). Three places break at TP=2:

1. `tt/attention/kv_cache.py` `allocate_kv_caches` allocates `[users*layers, 1, seq_local, hd]`, and
   `update_padded_kv_cache` asserts that cache heads == input heads.
2. `tt/attention/msa.py` `msa_sp_attention_cache_read` passes the slot to `high_bw_all_gather`'s selected-batch
   path. That path TT_FATALs on a non-singleton dim between batch and the gather dim
   (`high_bw_all_gather_device_operation.cpp:365-373`).
3. `tt/runners/kv_chunk_table.py:96` asserts `num_kv_heads == cols` (head h -> TP column h). The M3 adapter
   `allocate_kv_cache` (`tt/runners/adapters/minimax_m3.py:128-134`) inherits (1).

These parts already work unchanged at TP=2: head sharding, the q/k/v split, the dense ring_joint cache read
(its gather buffer shards the global n_kv over the cols), index_k (a single shared head), `read_slot_kv`, and MoE.

| step | work | estimate |
|---|---|---|
| (a) upstream the harness patches | `n_kv_local` in `allocate_kv_caches`; per-head slice / gather / concat in `msa_sp_attention_cache_read` behind `kv_cache.k.shape[1] > 1`; (4,2) cases with num_groups=2 in test_kv_cache_write_vs_ref, test_msa_sp_cache_read_vs_ref and test_attention_chunked_vs_ref (all MSA unit tests use num_groups=1 and (8,4) today) | 1-1.5 d |
| (b) native multi-head gather | let the selected-batch path take `[1, H, rows, hd]` as H contiguous page ranges (one per head, stride seq_local), or gather over the flattened head x rows dim with a per-head output offset. This removes the per-head slice + concat copies, i.e. 2 x (2 heads x seq_local x hd) of DRAM traffic per K and per V per sparse layer, plus the extra CCL launches. The measured cost is in per_op_4x2.csv, `ag_kv` with and without `head_slice` / `head_concat` | 2-3 d |
| (c) runner / migration | generalize kv_chunk_table (head h -> col h // (n_kv/tp), plus a head offset inside the chip), the adapter's allocate_kv_cache, and the decode-side layout agreement for migrated KV | 1-2 d |

The total to make (4,2) a production layout is about 4-6.5 d, on top of the runner pieces listed in recipe_4x2.md
(topology yaml, 4-mesh [4,2] MGD, per-rank TT_VISIBLE_DEVICES, a full 60-layer [4,2] weight cache, and a 2d-fabric
check). Only (a) is needed for correctness. (b) is a perf item, whose size is the head_slice + head_concat rows.

The measured cost of (b), per sparse layer, is the worst-chip `ag_kv_head_copies` in per_op_4x2.csv, mean of layers 3-6:

| point | head copies ms | share of layer | ag_kv with copies | ag_kv without (native est.) | (2,4) ag_kv |
|---|---:|---:|---:|---:|---:|
| W=4096 h=139264 | 0.53 | 3.6% | 2.10 | 1.57 | 0.49 |
| W=4096 h=548864 | 2.12 | 9.4% | 8.20 | 6.08 | 1.85 |
| W=8192 h=548864 | 2.03 | 5.3% | 12.83 | 10.80 | 1.94 |
| W=8192 packed 4x2048 | 6.16 | 16.6% | 13.00 | 6.84 | 2.31 |

The slices copy the whole slot capacity, not the written prefix, so a packed forward pays for every slot that has
history. Even without the copies, ag_kv stays about 3x the (2,4) value. That factor is structural: each chip
gathers (SP-1)/SP x kv_len for 2 heads (3/4 x 2) instead of 1 head (1/2 x 1), and the roofline is 3x as well.
Fixing (b) removes the copies but not that factor.
Layer 3's ag_kv also absorbs the skew of the dense ring before it, which is why the W=8192 h=548864 mean is high.
Over layers 4-6 alone the native estimate there is 5.2 ms (43% eff).

## Weight cache

`/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill/tensor_cache_bfp8_MeshShape([4, 2])` holds
embed + layers 0-6 (bf4 experts, 20 GB). The build including the bf16 read and tilize took 231 s, and the whole
run 300 s. That is far below the recipe's 20-30 min estimate, because expert conversion ran at ~45 s per sparse
layer. By extrapolation, a full 60-layer [4,2] cache is ~45 min of conversion, split across ranks.
