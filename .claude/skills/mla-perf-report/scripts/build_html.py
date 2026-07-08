#!/usr/bin/env python3
"""Generate the self-contained DeepSeek V3.2 sparse-MLA performance report HTML."""
import json, os, html

SP = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(SP + "/perf_data.json"))

# ------------------------------------------------------------------ block metadata
# Authored from the two code-verified trace specs. file:line + snippet feed the
# appendix (req 8.1); col/row drive the graph layout; ops list is the audit trail
# for the node-duration attribution (each op-code total is split across the blocks
# that emit it, per this mapping).
F = "models/demos/deepseek_v3_d_p/tt/mla/mla.py"
FI = "models/demos/deepseek_v3_d_p/tt/mla/indexer.py"

SPARSE_BLOCKS = [
    {
        "id": "in",
        "label": "hidden_states",
        "kind": "io",
        "col": 0,
        "row": 2,
        "desc": "Layer input activation (this chunk).",
        "outputs": [["hidden_states", "[1,1,1280,1792]", "bf16", "TILE", "SP dim2 · TP dim3"]],
    },
    {
        "id": "s1",
        "label": "q_a latent",
        "fn": "_q_a_latent",
        "file": F,
        "lines": "731-775",
        "col": 1,
        "row": 0,
        "desc": "Down-project to the q latent, TP all-reduce, q_a RMSNorm.",
        "snippet": "qr = ttnn.linear(hidden_states, self.q_a_proj_weight, ...)\nqr = ttnn.experimental.reduce_scatter_minimal_async(qr, dim=3, cluster_axis=self.tp_axis)\nqr = ttnn.experimental.all_gather_async(qr, dim=3, cluster_axis=self.tp_axis)\nreturn ttnn.rms_norm(qr, weight=self.q_a_layernorm_weight, ...)",
        "weights": [["q_a_proj", "[7168, 1536]", "TP-shard on hidden"], ["q_a_layernorm", "[1536]", "replicated"]],
        "ops": ["Matmul×1", "ReduceScatterMinimalAsync×1", "AllGatherAsync×1", "LayerNorm×1"],
    },
    {
        "id": "s2",
        "label": "indexer · key stem + K-cache write",
        "fn": "TtIndexer.write_k",
        "file": FI,
        "lines": "412-454",
        "col": 1,
        "row": 2,
        "desc": "Lightning-indexer key projection, TP all-reduce, k-norm, block-cyclic RoPE, write to the bf8 index K-cache.",
        "snippet": "k = ttnn.linear(hidden_states, self._idx_wk, ...)\nk = self._tp_rs_ag(k)                 # TP all-reduce\nk = ttnn.layer_norm(k, weight=self._idx_knorm_w, bias=self._idx_knorm_b, ...)\nk = self._bc_rope_pe(k, rope_tensors, start_pos)\nttnn.experimental.deepseek_prefill.update_padded_kv_cache(index_kbuf, k, ...)",
        "weights": [
            ["indexer.wk", "[7168, 128]", "TP-shard on hidden"],
            ["indexer.k_norm(+bias)", "[128]", "replicated"],
            ["_rope_perm", "[64, 64]", "replicated"],
        ],
        "ops": [
            "Matmul×2",
            "ReduceScatterMinimalAsync/ReduceScatter",
            "AllGatherAsync",
            "LayerNorm×1",
            "Slice×2",
            "RotaryEmbeddingIndexed×1",
            "Concat×1",
            "Typecast×1",
            "UpdatePaddedKvCache×1",
        ],
    },
    {
        "id": "s5",
        "label": "kv stem",
        "fn": "_kv_stem",
        "file": F,
        "lines": "827-906",
        "col": 1,
        "row": 4,
        "desc": "kv_a projection with MQA, TP all-reduce (all-gather+fast_reduce_nc), split nope/rope, kv-norm, RoPE, concat, to ROW_MAJOR cache format.",
        "snippet": "tt_kv = ttnn.linear(hidden_states, self.kv_a_proj_with_mqa_weight, ...)\ntt_kv = ttnn.experimental.all_gather_async(tt_kv, dim=1, cluster_axis=self.tp_axis)\ntt_kv = ttnn.experimental.fast_reduce_nc(tt_kv, dims=[1], ...)\ntt_kv_nope = ttnn.rms_norm(tt_kv_nope, weight=self.kv_a_layernorm_weight, ...)\ntt_kv_rope = self._apply_rope(tt_kv_rope, rope_tensors, kv_actual_isl)\ntt_kvpe = self._to_cache_format(ttnn.concat([tt_kv_nope, tt_kv_rope], dim=-1), kvpe_cache)",
        "weights": [
            ["kv_a_proj_with_mqa", "[7168, 576]", "TP-shard on hidden"],
            ["kv_a_layernorm", "[512]", "replicated"],
        ],
        "ops": [
            "Matmul×1",
            "AllGatherAsync×1",
            "FastReduceNC×1",
            "Slice×2",
            "LayerNorm×1",
            "RotaryEmbeddingIndexed×1",
            "Concat×1",
            "Untilize×1",
        ],
    },
    {
        "id": "s3",
        "label": "indexer · query + score + top-k",
        "fn": "TtIndexer.forward",
        "file": FI,
        "lines": "484-628",
        "col": 2,
        "row": 1,
        "desc": "Query stem + block-cyclic RoPE, per-head weights, gather full-T index keys, indexer_score_dsa (causal, in-kernel block-cyclic), then top-k → sparse indices.",
        "snippet": "q = ttnn.linear(qr, self._idx_wq_b, ...); q,_,_ = nlp_create_qkv_heads(q, ...)\nq_dev = self._bc_rope_pe(q, rope_tensors, start_pos)\nwts = self._tp_rs_ag(ttnn.linear(hidden_states, self._idx_wproj, ...), rs_only=True)\nk_full = self._gather_index_kbuf(index_kv_cache)          # SP all-gather → full T\nlogits = ttnn.experimental.indexer_score_dsa(q_dev, k_full, weights, chunk_start_idx=start_pos, ...)\nreturn ttnn.experimental.topk_large_indices(logits, k=min(index_topk, end_pos))",
        "weights": [
            ["indexer.wq_b", "[1536, 8192]", "TP col-shard (heads)"],
            ["indexer.weights_proj", "[7168, 64]", "TP-shard on hidden"],
        ],
        "ops": [
            "Matmul×3",
            "NlpCreateHeads×1",
            "Slice×2(+resid)",
            "RotaryEmbedding×1",
            "Concat×1(+resid)",
            "BinaryNg×1",
            "Permute×1(+resid)",
            "AllGather / AllBroadcast (index gather)",
            "IndexerScore×1",
            "Tilize×1",
            "ReduceScatter/AllGather (logits AR)",
            "Untilize×1",
            "TopkLargeIndices×1",
            "MeshPartition(reshard)",
        ],
    },
    {
        "id": "s4",
        "label": "q stem (absorbed)",
        "fn": "_q_stem",
        "file": F,
        "lines": "777-825",
        "col": 2,
        "row": 3,
        "desc": "q_b projection → heads → split nope/rope → absorb nope into the latent (wkv_b1) → RoPE → concat to the 576-wide absorbed query.",
        "snippet": "tt_q = ttnn.linear(qr, self.q_b_proj_weight, ...)\ntt_q,_,_ = ttnn.experimental.nlp_create_qkv_heads(tt_q, num_heads=num_heads_local, num_kv_heads=0, ...)\ntt_q_nope = ttnn.linear(tt_q_nope, self.wkv_b1_weight, ...)   # absorb into kv_lora latent\ntt_q_rope = self._apply_rope(tt_q_rope, rope_tensors, kv_actual_isl)\ntt_q = ttnn.concat([tt_q_nope, tt_q_rope], dim=-1)",
        "weights": [
            ["q_b_proj", "[1536, 24576]", "TP col-shard (heads)"],
            ["wkv_b1", "[128, 512] ×128 heads", "batched, per-head"],
        ],
        "ops": ["Matmul×2", "NlpCreateHeads×1", "Slice×2", "RotaryEmbeddingIndexed×1", "Concat×1", "Permute(resid)"],
    },
    {
        "id": "s6",
        "label": "KVPE cache write",
        "fn": "_sparse_chunked_attn",
        "file": F,
        "lines": "1186-1194",
        "col": 2,
        "row": 5,
        "desc": "Write this chunk's KVPE into its block-cyclic slot in the multi-user cache.",
        "snippet": "ttnn.experimental.deepseek_prefill.update_padded_kv_cache(\n    kvpe_cache, tt_kvpe, slot_idx=cache_user_id, layer_idx=cache_layer_idx,\n    num_layers=self.layer_num, kv_actual_global=kv_actual_isl, cluster_axis=self.sp_axis)",
        "weights": [],
        "ops": ["UpdatePaddedKvCache×1", "FillPad(resid)"],
    },
    {
        "id": "s7",
        "label": "KVPE prefix gather",
        "fn": "_gather_kvpe_prefix",
        "file": F,
        "lines": "1404-1421",
        "col": 3,
        "row": 4,
        "desc": "Reshard ND→interleaved and SP all-gather the full-T latent prefix (block-cyclic order preserved; sparse_sdpa remaps indices→pages in-kernel). Scales with prefix length.",
        "snippet": "cache_i = ttnn.to_memory_config(kvpe_cache, ttnn.DRAM_MEMORY_CONFIG)  # ND→interleaved\nfull = self._all_gather(cache_i, dim=2, cluster_axis=self.sp_axis)     # → [B,1,T,576] replicated",
        "weights": [],
        "ops": [
            "AllGatherAsync / AllBroadcast (prefix gather)",
            "MeshPartition(reshard)",
            "Copy×2",
            "Slice(resid)",
            "Concat(resid)",
            "Untilize(resid)",
            "UntilizeWithUnpadding",
        ],
    },
    {
        "id": "s8",
        "label": "sparse SDPA (top-k)",
        "fn": "_sparse_mla",
        "file": F,
        "lines": "1314-1402",
        "col": 4,
        "row": 2,
        "desc": "Absorbed MQA over the top-k=2048 selected latents (FlashMLA sparse contract, no causal mask — indices carry it). Per-chip single-chip sparse_sdpa; bounded by k, not prefix length.",
        "snippet": "q_rm = ttnn.to_layout(q_seq_sharded, ttnn.ROW_MAJOR_LAYOUT)\nout = ttnn.transformer.sparse_sdpa(q_rm, kvpe, idx, v_dim=self.kv_lora_rank, scale=self.scale,\n          k_chunk_size=k_chunk, block_cyclic_sp_axis=self.sp_axis,\n          block_cyclic_chunk_local=block_cyclic_chunk_local, cache_batch_idx=cache_batch_idx)\nret = ttnn.to_layout(out, ttnn.TILE_LAYOUT)",
        "weights": [],
        "ops": ["Untilize×1", "SparseSDPA×1", "Tilize×1"],
    },
    {
        "id": "s9",
        "label": "V projection (wkv_b2)",
        "fn": "_apply_wkv_b2",
        "file": F,
        "lines": "908-914",
        "col": 5,
        "row": 2,
        "desc": "Project the latent-V attention output up to v_head_dim per head.",
        "snippet": "return ttnn.linear(t, self.wkv_b2_weight, ...**self._get_mm_kwargs('wkv_b2', seq_len_local))",
        "weights": [["wkv_b2", "[512, 128] ×128 heads", "batched, per-head"]],
        "ops": ["Matmul×1"],
    },
    {
        "id": "s10",
        "label": "o_proj epilogue",
        "fn": "_o_proj_epilogue",
        "file": F,
        "lines": "950-970",
        "col": 6,
        "row": 2,
        "desc": "Concat heads, output projection, TP reduce-scatter back to the layer output contract.",
        "snippet": "v_out = ttnn.experimental.nlp_concat_heads(attn_out, ...)\nv_out = ttnn.linear(v_out, self.o_proj_weight, ...)\nreturn ttnn.experimental.reduce_scatter_minimal_async(v_out, dim=3, cluster_axis=self.tp_axis)",
        "weights": [["o_proj", "[16384, 7168]", "TP-shard on input"]],
        "ops": ["NLPConcatHeads×1", "Matmul×1", "ReduceScatterMinimalAsync×1", "UntilizeWithUnpadding(resid)"],
    },
    {
        "id": "out",
        "label": "layer output",
        "kind": "io",
        "col": 7,
        "row": 2,
        "desc": "MLA output activation (chunk).",
        "inputs": [["out", "[1,1,1280,1792]", "bf16", "TILE", "SP dim2 · TP dim3"]],
    },
]

DH = "SP↕2 seq(dim2) · TP↕4 hidden(dim3)"  # hidden-size sharded on TP
DR = "SP↕2 seq(dim2) · TP replicated"  # replicated on TP
DHD = "SP↕2 seq(dim2) · TP↕4 heads(dim1)"  # head-sharded on TP
DBC = "SP↕2 block-cyclic(dim2) · TP replicated"  # block-cyclic cache
DG = "SP replicated (gathered) · block-cyclic order"
SPARSE_EDGES = [
    ["in", "s1", "hidden", "[1,1,1280,1792]", "bf16", "TILE", DH],
    ["in", "s2", "hidden", "[1,1,1280,1792]", "bf16", "TILE", DH],
    ["in", "s3", "hidden (weights_proj)", "[1,1,1280,1792]", "bf16", "TILE", DH],
    ["in", "s5", "hidden", "[1,1,1280,1792]", "bf16", "TILE", DH],
    ["s1", "s3", "qr", "[1,1,1280,1536]", "bf16", "TILE", DR],
    ["s1", "s4", "qr", "[1,1,1280,1536]", "bf16", "TILE", DR],
    ["s2", "s3", "index_kv_cache", "[B,1,T/sp,128]", "bf8", "ROW_MAJOR", DBC],
    ["s3", "s8", "indices (top-k)", "[1,1,1280,2048]", "uint32", "ROW_MAJOR", DR],
    ["s4", "s8", "tt_q (absorbed)", "[1,32,1280,576]", "bf16", "TILE", DHD],
    ["s5", "s6", "tt_kvpe", "[1,1,1280,576]", "bf16", "ROW_MAJOR", DR],
    ["s6", "s7", "kvpe_cache", "[B,1,T/sp,576]", "bf16", "ROW_MAJOR", DBC],
    ["s7", "s8", "kvpe_dev (prefix)", "[B,1,T,576]", "bf16", "ROW_MAJOR", DG],
    ["s8", "s9", "attn_out (latent-V)", "[1,32,1280,512]", "bf16", "TILE", DHD],
    ["s9", "s10", "attn_out", "[1,32,1280,128]", "bf16", "TILE", DHD],
    ["s10", "out", "out", "[1,1,1280,1792]", "bf16", "TILE", DH],
]

DENSE_BLOCKS = [
    {
        "id": "in",
        "label": "hidden_states",
        "kind": "io",
        "col": 0,
        "row": 1,
        "desc": "Layer input activation (this chunk).",
        "outputs": [["hidden_states", "[1,1,1280,1792]", "bf16", "TILE", "SP dim2 · TP dim3"]],
    },
    {
        "id": "d1",
        "label": "q_a latent",
        "fn": "_q_a_latent",
        "file": F,
        "lines": "731-775",
        "col": 1,
        "row": 0,
        "desc": "Down-project to the q latent, TP all-reduce, q_a RMSNorm. (Identical to the sparse q_a stem.)",
        "snippet": "qr = ttnn.linear(hidden_states, self.q_a_proj_weight, ...)\nqr = ttnn.experimental.reduce_scatter_minimal_async(qr, dim=3, cluster_axis=self.tp_axis)\nqr = ttnn.experimental.all_gather_async(qr, dim=3, cluster_axis=self.tp_axis)\nreturn ttnn.rms_norm(qr, weight=self.q_a_layernorm_weight, ...)",
        "weights": [["q_a_proj", "[7168, 1536]", "TP-shard on hidden"], ["q_a_layernorm", "[1536]", "replicated"]],
        "ops": ["Matmul×1", "ReduceScatterMinimalAsync×1", "AllGatherAsync×1", "LayerNorm×1"],
    },
    {
        "id": "dnull",
        "label": "indexer (NullIndexer)",
        "fn": "NullIndexer.forward",
        "file": FI,
        "lines": "631-637",
        "col": 1,
        "row": 2,
        "kind": "noop",
        "desc": "Dense v3.1 baseline: has_indexer=False. forward() returns None with no device ops — no indexer, no top-k.",
        "snippet": "class NullIndexer:\n    def forward(self, *args, **kwargs):\n        return None   # dense path: no indexer, no top-k",
        "weights": [],
        "ops": ["(0 device ops)"],
    },
    {
        "id": "d3",
        "label": "kv stem",
        "fn": "_kv_stem",
        "file": F,
        "lines": "827-906",
        "col": 1,
        "row": 3,
        "desc": "kv_a projection with MQA, TP all-reduce, split, kv-norm, RoPE, concat, cache format (bf8 TILE for the dense ring cache).",
        "snippet": "tt_kv = ttnn.linear(hidden_states, self.kv_a_proj_with_mqa_weight, ...)\ntt_kv = ttnn.experimental.all_gather_async(tt_kv, dim=1, cluster_axis=self.tp_axis)\ntt_kv = ttnn.experimental.fast_reduce_nc(tt_kv, dims=[1], ...)\ntt_kv_nope = ttnn.rms_norm(tt_kv_nope, weight=self.kv_a_layernorm_weight, ...)\ntt_kvpe = self._to_cache_format(ttnn.concat([tt_kv_nope, tt_kv_rope], dim=-1), kvpe_cache)",
        "weights": [
            ["kv_a_proj_with_mqa", "[7168, 576]", "TP-shard on hidden"],
            ["kv_a_layernorm", "[512]", "replicated"],
        ],
        "ops": [
            "Matmul×1",
            "AllGatherAsync×1",
            "FastReduceNC×1",
            "Slice×2",
            "LayerNorm×1",
            "RotaryEmbeddingIndexed×1",
            "Concat×1",
            "Typecast×1",
        ],
    },
    {
        "id": "d2",
        "label": "q stem (absorbed)",
        "fn": "_q_stem",
        "file": F,
        "lines": "777-825",
        "col": 2,
        "row": 0,
        "desc": "q_b projection → heads → split → absorb nope into latent (wkv_b1) → RoPE → concat.",
        "snippet": "tt_q = ttnn.linear(qr, self.q_b_proj_weight, ...)\ntt_q,_,_ = ttnn.experimental.nlp_create_qkv_heads(tt_q, num_heads=num_heads_local, ...)\ntt_q_nope = ttnn.linear(tt_q_nope, self.wkv_b1_weight, ...)\ntt_q = ttnn.concat([tt_q_nope, tt_q_rope], dim=-1)",
        "weights": [
            ["q_b_proj", "[1536, 24576]", "TP col-shard (heads)"],
            ["wkv_b1", "[128, 512] ×128 heads", "batched, per-head"],
        ],
        "ops": ["Matmul×2", "NlpCreateHeads×1", "Slice×2", "RotaryEmbeddingIndexed×1", "Concat×1"],
    },
    {
        "id": "d4",
        "label": "KVPE cache write",
        "fn": "_chunked_attn",
        "file": F,
        "lines": "683-691",
        "col": 2,
        "row": 3,
        "desc": "Write this chunk's KVPE into its cache slot (bf8 TILE).",
        "snippet": "ttnn.experimental.deepseek_prefill.update_padded_kv_cache(\n    kvpe_cache, tt_kvpe, slot_idx=cache_user_id, layer_idx=cache_layer_idx,\n    num_layers=self.layer_num, kv_actual_global=kv_actual_isl, cluster_axis=self.sp_axis)",
        "weights": [],
        "ops": ["UpdatePaddedKvCache×1"],
    },
    {
        "id": "d5",
        "label": "ring MLA (full prefix)",
        "fn": "_chunked_attn",
        "file": F,
        "lines": "696-716",
        "col": 3,
        "row": 1,
        "desc": "ttnn.transformer.ring_mla → RingJointSDPA over the ENTIRE prefix (logical_n = full cache+chunk), causal, V materialized in-op from the latent cache. Cost scales with prefix length.",
        "snippet": "attn_out, _ = ttnn.transformer.ring_mla(\n    tt_q, kvpe_cache, persistent_output_buffer_kv=self._chunked_kv_buf,\n    head_dim_v=self.kv_lora_rank, logical_n=kv_actual_isl + chunk_size_global,\n    cluster_axis=self.sp_axis, kv_cache_batch_idx=cache_batch_idx, kv_actual_isl=kv_actual_isl)",
        "weights": [],
        "ops": ["RingJointSDPA×1"],
    },
    {
        "id": "d6",
        "label": "V projection (wkv_b2)",
        "fn": "_chunked_attn",
        "file": F,
        "lines": "723-728",
        "col": 4,
        "row": 1,
        "desc": "Project the latent-V attention output up to v_head_dim per head.",
        "snippet": "attn_out = ttnn.linear(attn_out, self.wkv_b2_weight, ...**self._get_mm_kwargs('wkv_b2', seq_len_local))",
        "weights": [["wkv_b2", "[512, 128] ×128 heads", "batched, per-head"]],
        "ops": ["Matmul×1"],
    },
    {
        "id": "d7",
        "label": "o_proj epilogue",
        "fn": "_o_proj_epilogue",
        "file": F,
        "lines": "950-970",
        "col": 5,
        "row": 1,
        "desc": "Concat heads, output projection, TP reduce-scatter.",
        "snippet": "v_out = ttnn.experimental.nlp_concat_heads(attn_out, ...)\nv_out = ttnn.linear(v_out, self.o_proj_weight, ...)\nreturn ttnn.experimental.reduce_scatter_minimal_async(v_out, dim=3, cluster_axis=self.tp_axis)",
        "weights": [["o_proj", "[16384, 7168]", "TP-shard on input"]],
        "ops": ["NLPConcatHeads×1", "Matmul×1", "ReduceScatterMinimalAsync×1"],
    },
    {
        "id": "out",
        "label": "layer output",
        "kind": "io",
        "col": 6,
        "row": 1,
        "desc": "MLA output activation (chunk).",
        "inputs": [["out", "[1,1,1280,1792]", "bf16", "TILE", "SP dim2 · TP dim3"]],
    },
]

DENSE_EDGES = [
    ["in", "d1", "hidden", "[1,1,1280,1792]", "bf16", "TILE", DH],
    ["in", "d3", "hidden", "[1,1,1280,1792]", "bf16", "TILE", DH],
    ["d1", "d2", "qr", "[1,1,1280,1536]", "bf16", "TILE", DR],
    ["d1", "dnull", "qr", "[1,1,1280,1536]", "bf16", "TILE", DR],
    ["d2", "d5", "tt_q (absorbed)", "[1,32,1280,576]", "bf16", "TILE", DHD],
    ["d3", "d4", "tt_kvpe", "[1,1,1280,576]", "bf8", "TILE", DR],
    ["d4", "d5", "kvpe_cache", "[B,1,T/sp,576]", "bf8", "TILE", DBC],
    ["d5", "d6", "attn_out (latent-V)", "[1,32,1280,512]", "bf16", "TILE", DHD],
    ["d6", "d7", "attn_out", "[1,32,1280,128]", "bf8", "TILE", DHD],
    ["d7", "out", "out", "[1,1,1280,1792]", "bf16", "TILE", DH],
]

BLOCKS = {"sparse": SPARSE_BLOCKS, "dense": DENSE_BLOCKS}
EDGES = {"sparse": SPARSE_EDGES, "dense": DENSE_EDGES}

META = {
    "branch": "main",
    "commit": "099251b681c",
    "commit_subject": "[Feature] #0 - Exabox system health check wrapper script added. (#48593)",
    "hardware": "LoudBox — 8× Blackhole p150b",
    "mesh": "SP=2 × TP=4 (line / FABRIC_1D)",
    "proxy": "1/4-Galaxy proxy (per-chip compute = Galaxy; sequence length ×1/4)",
    "test": "models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py",
    "local": {"chunk": 1280, "warm_cold_cache": 12800, "long_cache": 128000, "per_chip_seq": 640},
    "galaxy": {"chunk": 5120, "warm_cold_cache": 51200, "long_cache": 512000, "mesh": "SP=8 × TP=4"},
    "key_changes": [
        "Add sparse MLA (DSA) support into the deepseek_v3_d_p stack (#47832)",
        "block-cyclic per-user lightning-indexer key cache for sparse chunked prefill (#48938)",
        "sparse_sdpa multi-user read via cache_batch_idx; in-kernel block-cyclic remap, drop host reorder (#48888, #48733)",
        "Unify DSA MLA into a single ttMLA (has_indexer switches sparse vs dense v3.1 baseline)",
        "Add warm / cold / long perf scenarios to the MLA tracy harness (#49100)",
    ],
    "config": {
        "hidden": 7168,
        "heads": 128,
        "q_lora": 1536,
        "kv_lora": 512,
        "qk_rope": 64,
        "qk_nope": 128,
        "v_head": 128,
        "kvpe": 576,
        "index_heads": 64,
        "index_head_dim": 128,
        "index_topk": 2048,
    },
}

CAVEATS = [
    {
        "sev": "resolved",
        "title": "Sparse cold recovered from the un-clobbered raw report",
        "body": "The saved sparse cold summary CSV had been overwritten by a non-default 41-iteration / 25,600-token-chunk (~1.024M cache) stress run. The matched-parameter sparse cold (11 iterations, 1,280-token chunk — identical to dense cold) still existed in reports/2026_07_07_23_25_57 and was re-derived here with the driver's own merge_device_rows post-processing. All sparse-vs-dense cold numbers below use this recovered, matched run.",
    },
    {
        "sev": "pending",
        "title": "Dense long is not yet measured",
        "body": "No dense long-scenario tracy dump exists. The long comparison is sparse-only; dense long is marked N/A and will be generated after this report.",
    },
    {
        "sev": "info",
        "title": "Node durations are measured per call, assigned by execution order",
        "body": "Every device op call is device-collapsed across the 8 chips (compute=max=critical path, collectives=avg) and assigned to a semantic block by walking the execution-ordered call stream against the code-verified op template; once-per-forward structural ops (SparseSDPA, RingJointSDPA, IndexerScore, Topk, FastReduceNC, …) are pinned to their node. Block and op durations are thus REAL per-call Tracy times (not code-average estimates), and sum exactly to the scenario total. Caveat: a few ttnn ops are relabels/composites (the prefix gather surfaces as AllBroadcast; MeshPartition/Copy/padding come from to_memory_config/CCL internals) — these are placed in their issuing block with real time but flagged “composite” in the expanded view, as their internal wiring is inferred, not a literal Python call.",
    },
]

# output-tensor annotation per authored internal op node (feeds the expanded intra-block edges).
# shape · dtype · layout · dist — omitted for composite/relabel nodes (marked separately).
TENSOR = {
    # sparse
    "s1.qa": "[1,1,1280,1536] bf16 TILE · " + DR,
    "s1.norm": "[1,1,1280,1536] bf16 TILE · " + DR,
    "s2.wk": "[1,1,1280,128] bf16 TILE · " + DR,
    "s2.rope": "[1,1,1280,128] bf16 TILE · " + DR,
    "s2.wr": "index_kv_cache [B,1,T/sp,128] bf8 ROW_MAJOR · " + DBC,
    "s3.wqb": "[1,1,1280,8192] bf16 TILE · " + DHD,
    "s3.heads": "[1,16,1280,128] bf16 TILE · " + DHD,
    "s3.wproj": "weights [1,1,1280,64] bf16 · " + DR,
    "s3.score": "logits [1,16,1280,T] bf16 · SP↕2 seq · TP↕4 heads",
    "s3.topk": "indices [1,1,1280,2048] uint32 ROW_MAJOR · " + DR,
    "s4.qb": "[1,1,1280,24576] bf16 TILE · " + DHD,
    "s4.heads": "[1,32,1280,192] bf16 TILE · " + DHD,
    "s4.absorb": "q_nope→latent [1,32,1280,512] bf16 TILE · " + DHD,
    "s4.cat": "tt_q [1,32,1280,576] bf16 TILE · " + DHD,
    "s5.kva": "[1,1,1280,576] bf16 TILE · " + DR,
    "s5.cat": "[1,1,1280,576] bf16 TILE · " + DR,
    "s5.ut": "tt_kvpe [1,1,1280,576] bf16 ROW_MAJOR · " + DR,
    "s6.wr": "kvpe_cache [B,1,T/sp,576] bf16 ROW_MAJOR · " + DBC,
    "s7.ag": "kvpe_dev [B,1,T,576] bf16 ROW_MAJOR · " + DG,
    "s8.q2rm": "q [1,32,1280,576] bf16 ROW_MAJOR · " + DHD,
    "s8.sdpa": "attn_out [1,32,1280,512] bf16 ROW_MAJOR · " + DHD,
    "s8.o2tile": "attn_out [1,32,1280,512] bf16 TILE · " + DHD,
    "s9.wkvb2": "attn_out [1,32,1280,128] bf16 TILE · " + DHD,
    "s10.cat": "[1,1,1280,4096] bf16 TILE · " + DR,
    "s10.o": "[1,1,1280,7168] bf16 · " + DH,
    "s10.rs": "out [1,1,1280,1792] bf16 TILE · " + DH,
    # dense
    "d1.qa": "[1,1,1280,1536] bf16 TILE · " + DR,
    "d1.norm": "qr [1,1,1280,1536] bf16 TILE · " + DR,
    "d2.qb": "[1,1,1280,24576] bf16 TILE · " + DHD,
    "d2.absorb": "[1,32,1280,512] bf16 TILE · " + DHD,
    "d2.cat": "tt_q [1,32,1280,576] bf16 TILE · " + DHD,
    "d3.kva": "[1,1,1280,576] bf16 TILE · " + DR,
    "d3.tc": "tt_kvpe [1,1,1280,576] bf8 TILE · " + DR,
    "d4.wr": "kvpe_cache [B,1,T/sp,576] bf8 TILE · " + DBC,
    "d5.ring": "attn_out [1,32,1280,512] bf16 TILE · " + DHD,
    "d6.wkvb2": "attn_out [1,32,1280,128] bf8 TILE · " + DHD,
    "d7.cat": "[1,1,1280,4096] bf16 TILE · " + DR,
    "d7.o": "[1,1,1280,7168] bf16 · " + DH,
    "d7.rs": "out [1,1,1280,1792] bf16 TILE · " + DH,
}

# Node placement (col = level, top→down; row = lane, left→right) chosen to keep dataflow edges
# crossing-free: index / query / kv columns kept straight, branches converge on the attention node.
LAYOUT = {
    "sparse": {
        "in": (0, 1),
        "s2": (1, 0),
        "s1": (1, 1),
        "s5": (1, 2),
        "s3": (2, 0),
        "s4": (2, 1),
        "s6": (2, 2),
        "s7": (3, 2),
        "s8": (4, 1),
        "s9": (5, 1),
        "s10": (6, 1),
        "out": (7, 1),
    },
    "dense": {
        "in": (0, 1),
        "dnull": (1, 0),
        "d1": (1, 1),
        "d3": (1, 2),
        "d2": (2, 1),
        "d4": (2, 2),
        "d5": (3, 1),
        "d6": (4, 1),
        "d7": (5, 1),
        "out": (6, 1),
    },
}
for _m, _mp in LAYOUT.items():
    for _b in BLOCKS[_m]:
        if _b["id"] in _mp:
            _b["col"], _b["row"] = _mp[_b["id"]]

payload = {"data": DATA, "blocks": BLOCKS, "edges": EDGES, "meta": META, "caveats": CAVEATS, "tensor": TENSOR}
PAYLOAD_JSON = json.dumps(payload, separators=(",", ":"))

# ------------------------------------------------------------------ HTML
TITLE = "DeepSeek V3.2 · Sparse MLA Performance"
HTML = r"""<title>__TITLE__</title>
<style>
:root{
  --bg:#f7f8fa; --panel:#ffffff; --panel-2:#f0f2f6; --ink:#1a2230; --ink-soft:#55607a;
  --line:#e2e6ee; --line-strong:#cfd5e2; --accent:#0e7c86; --sparse:#0e7c86; --dense:#b7791f;
  --good:#1f9d57; --warn:#c9821a; --crit:#c0392b; --mono:"SFMono-Regular",ui-monospace,Menlo,Consolas,monospace;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0d1017; --panel:#151a24; --panel-2:#1b212d; --ink:#e6ebf4; --ink-soft:#94a0b8;
  --line:#232c3a; --line-strong:#33404f; --accent:#3fb8c4; --sparse:#3fb8c4; --dense:#e0a83a;
  --good:#3fce7f; --warn:#e6a13a; --crit:#e8695a;
}}
:root[data-theme="light"]{--bg:#f7f8fa;--panel:#ffffff;--panel-2:#f0f2f6;--ink:#1a2230;--ink-soft:#55607a;--line:#e2e6ee;--line-strong:#cfd5e2;--accent:#0e7c86;--sparse:#0e7c86;--dense:#b7791f;--good:#1f9d57;--warn:#c9821a;--crit:#c0392b;}
:root[data-theme="dark"]{--bg:#0d1017;--panel:#151a24;--panel-2:#1b212d;--ink:#e6ebf4;--ink-soft:#94a0b8;--line:#232c3a;--line-strong:#33404f;--accent:#3fb8c4;--sparse:#3fb8c4;--dense:#e0a83a;--good:#3fce7f;--warn:#e6a13a;--crit:#e8695a;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5;-webkit-font-smoothing:antialiased;}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px 80px;}
h1,h2,h3{text-wrap:balance;margin:0;}
a{color:var(--accent);}
code,.mono{font-family:var(--mono);font-variant-numeric:tabular-nums;}
header.top{position:sticky;top:0;z-index:40;background:color-mix(in srgb,var(--bg) 88%,transparent);backdrop-filter:blur(10px);border-bottom:1px solid var(--line);}
.top-inner{max-width:1180px;margin:0 auto;padding:12px 20px;display:flex;flex-wrap:wrap;gap:14px 20px;align-items:center;}
.brand{display:flex;flex-direction:column;gap:2px;margin-right:auto;}
.brand .k{font:600 12px/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--accent);}
.brand .t{font:650 16px/1.2 var(--sans);}
.seg{display:inline-flex;background:var(--panel-2);border:1px solid var(--line);border-radius:9px;padding:3px;gap:2px;}
.seg button{font:600 12.5px/1 var(--sans);color:var(--ink-soft);background:none;border:0;padding:7px 12px;border-radius:6px;cursor:pointer;letter-spacing:.01em;}
.seg button[aria-pressed="true"]{background:var(--panel);color:var(--ink);box-shadow:0 1px 2px rgba(0,0,0,.14);}
.seg.mode button[aria-pressed="true"][data-m="sparse"]{color:var(--sparse);}
.seg.mode button[aria-pressed="true"][data-m="dense"]{color:var(--dense);}
.seg-label{font:600 10px/1 var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--ink-soft);margin-right:2px;align-self:center;}
.ctl{display:flex;align-items:center;gap:7px;}
.theme-btn{margin-left:4px;background:var(--panel-2);border:1px solid var(--line);color:var(--ink-soft);border-radius:8px;width:34px;height:34px;cursor:pointer;font-size:15px;}
.hero{padding:34px 0 10px;}
.hero h1{font:700 30px/1.15 var(--sans);letter-spacing:-.01em;}
.hero p.lede{margin:12px 0 0;max-width:70ch;color:var(--ink-soft);font-size:15.5px;}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px;}
.chip{font:600 12px/1 var(--sans);padding:7px 11px;border-radius:20px;background:var(--panel);border:1px solid var(--line);color:var(--ink-soft);}
.chip b{color:var(--ink);font-weight:650;}
section{margin-top:34px;}
.sec-head{display:flex;align-items:baseline;gap:12px;margin-bottom:14px;border-bottom:1px solid var(--line);padding-bottom:8px;}
.sec-head .n{font:700 11px/1 var(--mono);color:var(--accent);letter-spacing:.1em;}
.sec-head h2{font:650 19px/1.2 var(--sans);}
.sec-head .sub{margin-left:auto;color:var(--ink-soft);font-size:12.5px;}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:18px;}
.grid-sum{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;}
.stat .l{font:600 10.5px/1 var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--ink-soft);}
.stat .v{font:700 25px/1.1 var(--sans);margin-top:8px;font-variant-numeric:tabular-nums;}
.stat .d{font-size:12px;margin-top:4px;color:var(--ink-soft);}
.delta.pos{color:var(--good);} .delta.neg{color:var(--crit);}
.two{display:grid;grid-template-columns:1fr;gap:18px;}
@media(min-width:900px){.two{grid-template-columns:1.15fr .85fr;}}
table{width:100%;border-collapse:collapse;font-size:13px;}
th,td{text-align:right;padding:6px 9px;border-bottom:1px solid var(--line);white-space:nowrap;}
th:first-child,td:first-child{text-align:left;}
th{font:600 10.5px/1.3 var(--mono);letter-spacing:.05em;text-transform:uppercase;color:var(--ink-soft);cursor:pointer;user-select:none;position:sticky;top:0;background:var(--panel);}
td.op{font-family:var(--mono);font-size:12px;}
td .bar{position:relative;}
.tblwrap{max-height:520px;overflow:auto;border:1px solid var(--line);border-radius:11px;}
.dbar{height:7px;border-radius:4px;background:var(--accent);opacity:.85;}
.dbar-cell{min-width:120px;}
.dbar-track{background:var(--panel-2);border-radius:4px;overflow:hidden;}
.tag{display:inline-block;font:600 10px/1 var(--mono);padding:3px 6px;border-radius:5px;background:var(--panel-2);color:var(--ink-soft);border:1px solid var(--line);}
.legend{display:flex;flex-wrap:wrap;gap:14px;align-items:center;font-size:12px;color:var(--ink-soft);margin-bottom:10px;}
.heatbar{height:11px;width:150px;border-radius:6px;background:linear-gradient(90deg,#ffffff,#ffd9c9,#f08a6b,#d64a2f,#a5160b);border:1px solid var(--line-strong);}
.graphwrap{position:relative;height:600px;overflow:hidden;border:1px solid var(--line);border-radius:12px;background:
  radial-gradient(circle at 1px 1px, var(--line) 1px, transparent 0) 0 0/22px 22px, var(--panel);cursor:grab;}
.graphwrap.panning{cursor:grabbing;}
svg.graph{display:block;width:100%;height:100%;touch-action:none;}
.gzoom{position:absolute;top:10px;right:10px;z-index:5;display:flex;gap:4px;align-items:center;}
.gzoom button{background:var(--panel);border:1px solid var(--line-strong);color:var(--ink);border-radius:7px;height:28px;min-width:28px;padding:0 8px;cursor:pointer;font:700 13px/1 var(--mono);box-shadow:0 1px 3px rgba(0,0,0,.12);}
.gzoom button:hover{background:var(--panel-2);}
.gzoom .zlvl{font:600 11px/1 var(--mono);color:var(--ink-soft);padding:0 4px;min-width:38px;text-align:center;}
.node rect{stroke:var(--line-strong);stroke-width:1;rx:9;cursor:pointer;transition:filter .12s;}
.node:hover rect{filter:brightness(1.06);}
.node.sel rect{stroke:var(--accent);stroke-width:2.5;}
.node text{pointer-events:none;font-family:var(--sans);}
.node .nlab{font:650 12.5px/1 var(--sans);}
.node .ndur{font:700 12px/1 var(--mono);}
.node .npct{font:600 10px/1 var(--mono);opacity:.8;}
.node.io rect{fill:var(--panel-2);stroke-dasharray:4 3;}
.node.noop rect{fill:var(--panel-2);opacity:.6;stroke-dasharray:3 3;}
.edge{fill:none;stroke:var(--line-strong);stroke-width:1.6;marker-end:url(#arrow);}
.edge.hot{stroke:var(--accent);}
.elabel{font:600 9.5px/1 var(--mono);fill:var(--ink-soft);}
.elabel .bg{fill:var(--panel);}
.drawer{position:fixed;top:0;right:0;height:100%;width:min(440px,92vw);background:var(--panel);border-left:1px solid var(--line-strong);box-shadow:-8px 0 30px rgba(0,0,0,.22);transform:translateX(105%);transition:transform .22s ease;z-index:60;overflow:auto;}
.drawer.open{transform:none;}
.drawer .dh{position:sticky;top:0;background:var(--panel);border-bottom:1px solid var(--line);padding:16px 18px;display:flex;align-items:flex-start;gap:10px;}
.drawer .dh h3{font:650 16px/1.25 var(--sans);}
.drawer .db{padding:16px 18px;}
.drawer .close{margin-left:auto;background:var(--panel-2);border:1px solid var(--line);border-radius:8px;width:30px;height:30px;cursor:pointer;color:var(--ink);}
.kv{font:600 10.5px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink-soft);margin:16px 0 7px;}
pre.code{background:var(--panel-2);border:1px solid var(--line);border-radius:9px;padding:12px;overflow:auto;font:12px/1.5 var(--mono);margin:0;}
.iot{width:100%;font-size:11.5px;}
.iot td{padding:4px 6px;border-bottom:1px solid var(--line);white-space:normal;}
.pathref{font:11.5px/1.4 var(--mono);color:var(--ink-soft);word-break:break-all;}
.caveat{display:block;border-left:3px solid var(--line-strong);padding:10px 14px;border-radius:0 9px 9px 0;background:var(--panel-2);margin:0 0 10px;overflow-wrap:anywhere;}
#caveatBox{padding-top:14px;display:block;}
.caveat.resolved{border-color:var(--good);} .caveat.pending{border-color:var(--warn);} .caveat.info{border-color:var(--accent);}
.caveat h4{font:650 13.5px/1.3 var(--sans);margin:0 0 4px;display:block;}
.caveat .badge{font:700 9px/1 var(--mono);letter-spacing:.1em;text-transform:uppercase;padding:3px 6px;border-radius:5px;display:inline-block;vertical-align:middle;margin-right:8px;}
.caveat.resolved .badge{background:color-mix(in srgb,var(--good) 20%,transparent);color:var(--good);}
.caveat.pending .badge{background:color-mix(in srgb,var(--warn) 22%,transparent);color:var(--warn);}
.caveat.info .badge{background:color-mix(in srgb,var(--accent) 20%,transparent);color:var(--accent);}
.caveat p{margin:0;font-size:13px;color:var(--ink-soft);}
.cmp-bars{display:flex;flex-direction:column;gap:14px;}
.cmp-row{display:grid;grid-template-columns:64px 1fr;gap:10px;align-items:center;}
.cmp-row .sc{font:600 12px/1 var(--mono);text-transform:uppercase;color:var(--ink-soft);}
.cmp-track{display:flex;flex-direction:column;gap:5px;}
.cmp-bar{height:19px;border-radius:5px;display:flex;align-items:center;justify-content:flex-end;padding-right:8px;font:700 11px/1 var(--mono);color:#fff;min-width:2px;}
.cmp-bar.sparse{background:var(--sparse);} .cmp-bar.dense{background:var(--dense);}
.cmp-bar.na{background:var(--panel-2);color:var(--ink-soft);justify-content:center;padding:0;border:1px dashed var(--line-strong);}
.note{font-size:12.5px;color:var(--ink-soft);}
.hidden{display:none!important;}
.iterwrap{overflow-x:auto;}
.miss{color:var(--warn);font-weight:600;}
details.appx{border:1px solid var(--line);border-radius:11px;margin-bottom:10px;background:var(--panel);overflow:hidden;}
details.appx .ac{display:block;overflow-wrap:anywhere;}
details.appx summary{cursor:pointer;padding:13px 16px;font:650 14px/1.2 var(--sans);list-style:none;display:flex;align-items:center;gap:10px;}
details.appx summary::-webkit-details-marker{display:none;}
details.appx summary .tri{transition:transform .15s;color:var(--ink-soft);}
details.appx[open] summary .tri{transform:rotate(90deg);}
details.appx .ac{padding:0 16px 16px;}
.footer{margin-top:50px;padding-top:20px;border-top:1px solid var(--line);color:var(--ink-soft);font-size:12px;}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;}
@media (prefers-reduced-motion:reduce){*{transition:none!important;}}

/* --- mode pills (report-wide convention: teal=sparse, amber=dense) --- */
.mpill{display:inline-flex;align-items:center;gap:6px;font:700 11px/1 var(--sans);padding:5px 10px;border-radius:20px;border:1px solid transparent;}
.mpill .dot{width:9px;height:9px;border-radius:50%;}
.mpill.sparse{background:color-mix(in srgb,var(--sparse) 15%,transparent);color:var(--sparse);border-color:color-mix(in srgb,var(--sparse) 35%,transparent);}
.mpill.sparse .dot{background:var(--sparse);}
.mpill.dense{background:color-mix(in srgb,var(--dense) 15%,transparent);color:var(--dense);border-color:color-mix(in srgb,var(--dense) 35%,transparent);}
.mpill.dense .dot{background:var(--dense);}
.mpill.off{opacity:.4;}
.hdr-legend{margin-left:auto;display:flex;gap:8px;align-items:center;}
.cmp-bar.sparse{background:var(--sparse);} .cmp-bar.dense{background:var(--dense);}
.cmp-bar .blab{margin-right:auto;padding-left:8px;opacity:.9;font-weight:700;}
.cur-mode{font-weight:700;}
.cur-mode.sparse{color:var(--sparse);} .cur-mode.dense{color:var(--dense);}
/* --- info button + tooltip (class scoped to .infobtn so it never matches .caveat.info) --- */
.infobtn{position:relative;display:inline-flex;align-items:center;justify-content:center;width:19px;height:19px;border-radius:50%;border:1px solid var(--line-strong);background:var(--panel-2);color:var(--ink-soft);font:700 11px/1 var(--sans);cursor:help;margin-left:2px;}
.infobtn .pop{position:absolute;top:26px;left:50%;transform:translateX(-50%);width:300px;background:var(--panel);border:1px solid var(--line-strong);border-radius:9px;padding:11px 13px;box-shadow:0 8px 26px rgba(0,0,0,.22);font:400 12px/1.5 var(--sans);color:var(--ink-soft);z-index:20;opacity:0;visibility:hidden;transition:opacity .13s;text-align:left;pointer-events:none;}
.infobtn:hover .pop,.infobtn:focus .pop,.infobtn:focus-within .pop,.infobtn.open .pop{opacity:1;visibility:visible;}
.infobtn .pop b{color:var(--ink);}
/* --- graph expand controls + internal op nodes --- */
.gtoggle{display:inline-flex;gap:0;}
.node .obox{fill:var(--panel);stroke:var(--line);stroke-width:1;}
.onode rect{stroke:var(--line);stroke-width:.8;cursor:default;}
.onode text{font-family:var(--mono);}
.onode .ol{font:600 9.5px/1 var(--mono);}
.onode .od{font:700 9px/1 var(--mono);}
.onode.comp rect{stroke-dasharray:3 2;opacity:.92;}
.oedge{fill:none;stroke:var(--line-strong);stroke-width:1;marker-end:url(#arrowsm);}
.oelabel{font:500 8px/1 var(--mono);fill:var(--ink-soft);}
.node .expbtn{cursor:pointer;}
.node .expbtn rect{fill:var(--panel-2);stroke:var(--line-strong);}
.node .expbtn text{font:700 11px/1 var(--mono);fill:var(--ink-soft);}
.compbadge{font:700 7.5px/1 var(--mono);fill:var(--warn);}
</style>

<header class="top">
  <div class="top-inner">
    <div class="brand"><span class="k">DeepSeek V3.2 · DSA</span><span class="t">Sparse MLA Perf Report</span></div>
    <div class="ctl"><span class="seg-label">Scenario</span>
      <div class="seg" id="segScenario" role="group" aria-label="scenario">
        <button data-s="warm" aria-pressed="true">Warm</button>
        <button data-s="cold" aria-pressed="false">Cold</button>
        <button data-s="long" aria-pressed="false">Long</button>
      </div></div>
    <div class="ctl"><span class="seg-label">Mode</span>
      <div class="seg mode" id="segMode" role="group" aria-label="mode">
        <button data-m="sparse" aria-pressed="true">Sparse</button>
        <button data-m="dense" aria-pressed="false">Dense</button>
      </div></div>
    <button class="theme-btn" id="themeBtn" title="Toggle theme" aria-label="Toggle theme">◐</button>
  </div>
</header>

<div class="wrap">
  <div class="hero">
    <h1>DeepSeek V3.2 sparse attention in MLA — device performance</h1>
    <p class="lede">Critical-path device-kernel timing for the chunked-prefill MLA layer, measured with Tracy on
      <b id="hwLede"></b>. Sparse (v3.2 DSA: lightning indexer + top-k <span class="mono">sparse_sdpa</span>) is
      compared against the dense v3.1 ring-MLA baseline across three prefill scenarios. Every graph node is traced
      to source; every duration comes from a Tracy report.</p>
    <div class="chips" id="heroChips"></div>
  </div>

  <section id="summary">
    <div class="sec-head"><span class="n">01</span><h2>Scenario summary</h2><span class="sub" id="sumSub"></span></div>
    <div class="grid-sum" id="statGrid"></div>
  </section>

  <section id="compare">
    <div class="sec-head"><span class="n">02</span><h2>Sparse vs dense — total critical path</h2>
      <span class="hdr-legend">
        <span class="mpill sparse"><span class="dot"></span>Sparse (v3.2 DSA)</span>
        <span class="mpill dense"><span class="dot"></span>Dense (v3.1 ring)</span>
      </span></div>
    <p class="note" style="margin:-6px 0 12px">Device-collapsed device-kernel time · lower is better. Teal = sparse, amber = dense throughout this report.</p>
    <div class="panel"><div class="cmp-bars" id="cmpBars"></div>
      <p class="note" id="cmpNote" style="margin-top:14px"></p></div>
  </section>

  <section id="ops">
    <div class="sec-head"><span class="n">03</span><h2>Operations</h2>
      <span class="infobtn" tabindex="0" aria-label="How to read this table">i<span class="pop" id="opsInfo"></span></span>
      <span class="sub" id="opsSub"></span></div>
    <div class="tblwrap"><table id="opTable"><thead><tr>
      <th data-sort="ord">Op code</th><th data-sort="count">Calls</th>
      <th data-sort="total">Total ms</th><th data-sort="avg">Avg µs</th>
      <th data-sort="pct">% of trace</th><th class="dbar-cell">share</th>
    </tr></thead><tbody id="opBody"></tbody></table></div>
    <p class="note" style="margin-top:8px">Click a column to sort (default: by duration).</p>
  </section>

  <section id="cold" class="hidden">
    <div class="sec-head"><span class="n">04</span><h2>Cold prefill — growth as the cache fills</h2>
      <span class="sub" id="coldSub"></span></div>
    <div class="two">
      <div class="panel"><h3 style="font:650 14px/1 var(--sans);margin-bottom:4px">Per-iteration critical path</h3>
        <p class="note" style="margin:0 0 10px">Each iteration is one <span class="mono">chunk</span> forward; the last equals the warm step.</p>
        <div id="iterChart"></div></div>
      <div class="panel"><div style="display:flex;align-items:baseline;gap:10px;margin-bottom:6px">
        <h3 style="font:650 14px/1 var(--sans)">Top-N ops across iterations</h3>
        <span class="seg" style="margin-left:auto"><span class="seg-label" style="padding:6px 4px">Top</span>
          <input id="topN" type="number" min="3" max="20" value="10" style="width:52px;background:var(--panel-2);border:1px solid var(--line);color:var(--ink);border-radius:6px;padding:5px;font:600 12px var(--mono)"></span></div>
        <div id="topNChart"></div></div>
    </div>
  </section>

  <section id="graph">
    <div class="sec-head"><span class="n">05</span><h2>MLA dataflow graph</h2>
      <span class="mpill" id="graphModePill" style="margin-left:4px"><span class="dot"></span><span></span></span>
      <span class="hdr-legend">
        <span class="seg-label">View</span>
        <div class="seg gtoggle" id="segView" role="group" aria-label="graph detail">
          <button data-v="semantic" aria-pressed="true">Semantic</button>
          <button data-v="ops" aria-pressed="false">Ops</button>
        </div></span></div>
    <div class="legend">
      <span>Node heat (share of trace):</span><span>0%</span><span class="heatbar"></span><span>max</span>
      <span style="margin-left:12px">Flows top→down. Edges carry tensors — <span class="mono">shape · dtype · layout · dist</span>; hover for full detail.</span>
      <span>Click <b>＋</b> on a node to expand it into its constituent ops (or use the <b>Ops</b> view for all). Click a node label for source.</span>
    </div>
    <div class="graphwrap" id="graphwrap">
      <div class="gzoom">
        <button data-z="out" aria-label="Zoom out" title="Zoom out">−</button>
        <span class="zlvl" id="zlvl">100%</span>
        <button data-z="in" aria-label="Zoom in" title="Zoom in">＋</button>
        <button data-z="reset" title="Reset view">⟳ Reset</button>
      </div>
      <svg class="graph" id="svgGraph" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>
    <p class="note" id="graphNote" style="margin-top:10px"></p>
  </section>

  <section id="notes">
    <div class="sec-head"><span class="n">06</span><h2>Data integrity &amp; method</h2>
      <span class="sub">collapsed — expand for caveats</span></div>
    <details class="appx"><summary><span class="tri">▸</span> Data-integrity caveats &amp; measurement method</summary>
      <div class="ac" id="caveatBox"></div></details>
  </section>

  <section id="appendix">
    <div class="sec-head"><span class="n">07</span><h2>Appendix</h2><span class="sub">evidence · self-contained</span></div>
    <details class="appx"><summary><span class="tri">▸</span> A · Node reference — source &amp; snippets</summary>
      <div class="ac" id="appxNodes"></div></details>
    <details class="appx"><summary><span class="tri">▸</span> B · Full Tracy per-op reports (all scenarios × modes)</summary>
      <div class="ac" id="appxReports"></div></details>
    <details class="appx"><summary><span class="tri">▸</span> C · Experiment metadata — branch, commit, config, hardware</summary>
      <div class="ac" id="appxMeta"></div></details>
    <details class="appx"><summary><span class="tri">▸</span> D · Node-duration attribution model</summary>
      <div class="ac" id="appxMethod"></div></details>
  </section>

  <div class="footer" id="footer"></div>
</div>

<aside class="drawer" id="drawer" aria-hidden="true">
  <div class="dh"><h3 id="drTitle"></h3><button class="close" id="drClose" aria-label="Close">✕</button></div>
  <div class="db" id="drBody"></div>
</aside>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const P = JSON.parse(document.getElementById('payload').textContent);
const {data, blocks, edges, meta, caveats, tensor} = P;
let S = 'warm', M = 'sparse', sortKey='ord', sortDir=1;
let view='semantic';               // graph detail: 'semantic' | 'ops'
let expanded=new Set();            // per-node expand state (block ids)
let gView=null, gBase=null, gSig=null;  // graph pan/zoom viewBox state (+ layout signature)
const fmtMs = ns => (ns/1e6).toLocaleString(undefined,{minimumFractionDigits:3,maximumFractionDigits:3});
const fmtUs = ns => (ns/1e3).toLocaleString(undefined,{maximumFractionDigits:1});
const fmtInt = n => n.toLocaleString();
const esc = s => String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

/* ---------- theme ---------- */
const themeBtn=document.getElementById('themeBtn');
themeBtn.onclick=()=>{const cur=document.documentElement.getAttribute('data-theme')
  ||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');
  document.documentElement.setAttribute('data-theme',cur==='dark'?'light':'dark');draw();};

/* ---------- heat scale (white -> deep red) ---------- */
const HEAT=[[255,255,255],[255,217,201],[240,138,107],[214,74,47],[165,22,11]];
function heat(t){t=Math.max(0,Math.min(1,t));const x=t*(HEAT.length-1);const i=Math.floor(x);
  const f=x-i;const a=HEAT[i],b=HEAT[Math.min(i+1,HEAT.length-1)];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},${Math.round(a[2]+(b[2]-a[2])*f)})`;}
function inkOn(t){return t>0.55?'#fff':'#231a15';}

function ent(){return data.modes[M][S];}
function bt(){return data.block_timing[M][S];}

/* ---------- header chips + lede ---------- */
document.getElementById('hwLede').textContent=meta.hardware+', '+meta.mesh;
document.getElementById('heroChips').innerHTML=[
  ['Hardware',meta.hardware],['Mesh',meta.mesh],['Proxy',meta.proxy],
  ['Branch',meta.branch],['Commit',meta.commit],
  ['Local chunk',meta.local.chunk+' tok'],['Galaxy target','5120 tok · SP8×TP4']
].map(([k,v])=>`<span class="chip"><b>${esc(k)}</b> &nbsp;${esc(v)}</span>`).join('');

/* ---------- summary stats ---------- */
function drawSummary(){
  const e=ent(); const other=M==='sparse'?'dense':'sparse'; const oe=data.modes[other][S];
  const sub=document.getElementById('sumSub');
  const local = S==='long'?meta.local.long_cache:meta.local.warm_cold_cache;
  sub.innerHTML=`${M} · ${S} — LoudBox local: chunk ${meta.local.chunk}, cache ${fmtInt(local)} tok`;
  const g=document.getElementById('statGrid'); g.innerHTML='';
  const cards=[];
  cards.push(['Total critical path', fmtMs(e.total_ns)+' ms', `${fmtInt(e.total_calls)} op calls · ${e.ops.length} op codes`]);
  // vs other mode
  let d='';
  if(oe){const diff=(e.total_ns-oe.total_ns)/oe.total_ns*100;
    const faster=diff<0; d=`<span class="delta ${faster?'pos':'neg'}">${faster?'▼':'▲'} ${Math.abs(diff).toFixed(1)}% ${faster?'faster':'slower'}</span> vs ${other}`;
    cards.push([`vs ${other}`, fmtMs(oe.total_ns)+' ms', d]);
  } else cards.push([`vs ${other}`, 'N/A', `<span class="miss">${other} ${S} not measured</span>`]);
  // dominant op
  const top=[...e.ops].sort((a,b)=>b.total_ns-a.total_ns)[0];
  cards.push(['Dominant op', top.op.replace('DeviceOperation','').replace('Operation',''), `${top.pct.toFixed(1)}% · ${fmtMs(top.total_ns)} ms`]);
  // scenario descriptor
  const desc={warm:'1 chunk @ filled cache',cold:'full prefill, cache 0→max',long:'1 chunk @ 0.5M-class cache'}[S];
  cards.push(['Scenario', S, desc]);
  g.innerHTML=cards.map(([l,v,dd])=>`<div class="stat"><div class="l">${esc(l)}</div><div class="v">${v}</div><div class="d">${dd}</div></div>`).join('');
}

/* ---------- comparison bars ---------- */
function drawCompare(){
  const box=document.getElementById('cmpBars'); box.innerHTML='';
  let maxns=0; ['warm','cold','long'].forEach(sc=>['sparse','dense'].forEach(m=>{const e=data.modes[m][sc];if(e)maxns=Math.max(maxns,e.total_ns);}));
  ['warm','cold','long'].forEach(sc=>{
    const row=document.createElement('div'); row.className='cmp-row';
    let tracks='';
    ['sparse','dense'].forEach(m=>{
      const e=data.modes[m][sc];
      if(e){const w=Math.max(2,(e.total_ns/maxns*100)).toFixed(1);
        tracks+=`<div class="cmp-bar ${m}" style="width:${w}%"><span class="blab">${m}</span><span>${fmtMs(e.total_ns)} ms</span></div>`;}
      else tracks+=`<div class="cmp-bar na">${m} — N/A (not measured)</div>`;
    });
    row.innerHTML=`<div class="sc">${sc}${sc===S?' ◂':''}</div><div class="cmp-track">${tracks}</div>`;
    box.appendChild(row);
  });
  // note
  const w=data.modes.sparse.warm.total_ns, dw=data.modes.dense.warm.total_ns;
  const c=data.modes.sparse.cold.total_ns, dc=data.modes.dense.cold.total_ns;
  document.getElementById('cmpNote').innerHTML=
    `At the short warm prefix (12.8k tok) sparse is <b>${((1-w/dw)*100).toFixed(1)}% faster</b> than dense `+
    `(${fmtMs(w)} vs ${fmtMs(dw)} ms). Over the cold prefill sparse is <b>${((c/dc-1)*100).toFixed(1)}% slower</b> `+
    `(${fmtMs(c)} vs ${fmtMs(dc)} ms) — the indexer + top-k overhead is not yet amortised at this sequence length. `+
    `Sparse's advantage grows with prefix length: dense ring-MLA scales with the full prefix, while sparse attends a fixed top-k=2048. `+
    `<span class="miss">Dense long is not yet measured</span>, so the crossover point is not pinned here.`;
}

/* ---------- ops table ---------- */
function orderedOps(){
  // exec order proxy: the CSV is duration-sorted; keep a stable exec-order index by op code first-seen is not available,
  // so "ord" uses the code-verified block order mapping to sort; fallback to duration.
  const e=ent(); return e.ops.map((o,i)=>({...o,idx:i}));
}
const OPORDER={sparse:['q_a_proj','indexer','q_stem','kv_stem','write','gather','SparseSDPA','wkv_b2','o_proj'],dense:[]};
function drawOps(){
  const e=ent(); const body=document.getElementById('opBody');
  document.getElementById('opsSub').innerHTML=`${M} · ${S} — ${fmtInt(e.total_calls)} calls · total ${fmtMs(e.total_ns)} ms`;
  let rows=[...e.ops];
  const maxpct=Math.max(...rows.map(r=>r.pct));
  const keyf={ord:r=>r.total_ns,count:r=>r.count,total:r=>r.total_ns,avg:r=>r.avg_ns,pct:r=>r.pct};
  rows.sort((a,b)=>(keyf[sortKey](b)-keyf[sortKey](a))*sortDir);
  body.innerHTML=rows.map(r=>{
    const w=(r.pct/maxpct*100).toFixed(1);
    return `<tr><td class="op">${esc(r.op.replace('DeviceOperation','').replace('Operation',''))}</td>
      <td>${fmtInt(r.count)}</td><td>${fmtMs(r.total_ns)}</td><td>${fmtUs(r.avg_ns)}</td>
      <td>${r.pct.toFixed(2)}%</td>
      <td class="dbar-cell"><div class="dbar-track"><div class="dbar" style="width:${w}%;background:${heat(r.pct/maxpct)}"></div></div></td></tr>`;
  }).join('');
}
document.querySelectorAll('#opTable th[data-sort]').forEach(th=>th.onclick=()=>{
  const k=th.dataset.sort; if(sortKey===k)sortDir*=-1; else{sortKey=k;sortDir=1;} drawOps();});

/* ---------- cold charts ---------- */
function svgEl(t,a){const e=document.createElementNS('http://www.w3.org/2000/svg',t);for(const k in a)e.setAttribute(k,a[k]);return e;}
function drawCold(){
  const sec=document.getElementById('cold');
  if(S!=='cold'){sec.classList.add('hidden');return;} sec.classList.remove('hidden');
  const bi=data.cold_by_iter[M];
  document.getElementById('coldSub').textContent=`${M} · ${bi.length} iterations · step ${meta.local.chunk} tok`;
  // per-iteration total line
  drawLine('iterChart', bi.map(it=>it.iteration), bi.map(it=>it.total_ns/1e6), bi.map(it=>it.cache_depth_tokens),
    'iteration','ms', ['var(--'+M+')']);
  drawTopN();
}
document.getElementById('topN').oninput=drawTopN;
function drawTopN(){
  if(S!=='cold')return;
  const bi=data.cold_by_iter[M]; const N=Math.max(3,Math.min(20,+document.getElementById('topN').value||10));
  // aggregate each op across iters, take top N by summed total_ns
  const agg={}; bi.forEach(it=>it.ops.forEach(o=>{agg[o.op]=(agg[o.op]||0)+o.total_ns;}));
  const topOps=Object.entries(agg).sort((a,b)=>b[1]-a[1]).slice(0,N).map(x=>x[0]);
  const series=topOps.map(op=>bi.map(it=>{const f=it.ops.find(o=>o.op===op);return f?f.total_ns/1e6:0;}));
  drawMultiLine('topNChart', bi.map(it=>it.iteration), series, topOps.map(o=>o.replace('DeviceOperation','').replace('Operation','')));
}
function axis(g,W,H,pad,xs,ymax,xlab,ylab){
  g.appendChild(svgEl('line',{x1:pad.l,y1:H-pad.b,x2:W-pad.r,y2:H-pad.b,stroke:'var(--line-strong)'}));
  g.appendChild(svgEl('line',{x1:pad.l,y1:pad.t,x2:pad.l,y2:H-pad.b,stroke:'var(--line-strong)'}));
  for(let i=0;i<=4;i++){const yy=pad.t+(H-pad.t-pad.b)*i/4;const val=ymax*(1-i/4);
    g.appendChild(svgEl('line',{x1:pad.l,y1:yy,x2:W-pad.r,y2:yy,stroke:'var(--line)','stroke-dasharray':'2 3'}));
    const t=svgEl('text',{x:pad.l-6,y:yy+3,'text-anchor':'end'});t.setAttribute('style','font:10px var(--mono);fill:var(--ink-soft)');t.textContent=val.toFixed(val<10?1:0);g.appendChild(t);}
}
function drawLine(id,xs,ys,depths,xlab,ylab,colors){
  const host=document.getElementById(id);host.innerHTML='';
  const W=host.clientWidth||440,H=210,pad={l:44,r:14,t:12,b:34};
  const svg=svgEl('svg',{width:'100%',height:H,viewBox:`0 0 ${W} ${H}`});
  const ymax=Math.max(...ys)*1.12||1;axis(svg,W,H,pad,xs,ymax);
  const X=i=>pad.l+(W-pad.l-pad.r)*(i/(xs.length-1||1));const Y=v=>pad.t+(H-pad.t-pad.b)*(1-v/ymax);
  let dd=ys.map((v,i)=>`${i?'L':'M'}${X(i)},${Y(v)}`).join(' ');
  svg.appendChild(svgEl('path',{d:dd,fill:'none',stroke:colors[0],'stroke-width':2.4}));
  ys.forEach((v,i)=>{const c=svgEl('circle',{cx:X(i),cy:Y(v),r:3.2,fill:colors[0]});
    c.appendChild(svgEl('title',{})).textContent=`iter ${xs[i]} · depth ${depths[i].toLocaleString()} tok · ${v.toFixed(3)} ms`;svg.appendChild(c);});
  const xl=svgEl('text',{x:(W)/2,y:H-6,'text-anchor':'middle'});xl.setAttribute('style','font:10px var(--mono);fill:var(--ink-soft)');xl.textContent='iteration (cache depth →)';svg.appendChild(xl);
  host.appendChild(svg);
}
const PAL=['#3fb8c4','#e0a83a','#c0392b','#7e57c2','#1f9d57','#e8695a','#4f8ff0','#d081c4','#8ba63c','#e07b39','#5ac8b0','#b06cd6','#c99a2e','#6ac46a','#e05a8a','#4aa3a3','#9d7bd8','#c47a4a','#7ab648','#d64a90','#4f9fd0'];
function drawMultiLine(id,xs,series,labels){
  const host=document.getElementById(id);host.innerHTML='';
  const W=host.clientWidth||440,H=262,pad={l:44,r:14,t:12,b:64};
  const svg=svgEl('svg',{width:'100%',height:H,viewBox:`0 0 ${W} ${H}`});
  const ymax=Math.max(...series.flat())*1.12||1;axis(svg,W,H,pad,xs,ymax);
  const X=i=>pad.l+(W-pad.l-pad.r)*(i/(xs.length-1||1));const Y=v=>pad.t+(H-pad.t-pad.b)*(1-v/ymax);
  series.forEach((ys,si)=>{const col=PAL[si%PAL.length];
    let dd=ys.map((v,i)=>`${i?'L':'M'}${X(i)},${Y(v)}`).join(' ');
    svg.appendChild(svgEl('path',{d:dd,fill:'none',stroke:col,'stroke-width':1.9,opacity:.92}));
    ys.forEach((v,i)=>{const c=svgEl('circle',{cx:X(i),cy:Y(v),r:2.6,fill:col});
      c.appendChild(svgEl('title',{})).textContent=`${labels[si]} · iter ${xs[i]} · ${v.toFixed(3)} ms`; svg.appendChild(c);});});
  // x-axis iteration ticks
  xs.forEach((xv,i)=>{const tx=svgEl('text',{x:X(i),y:H-pad.b+12,'text-anchor':'middle'});tx.setAttribute('style','font:8.5px var(--mono);fill:var(--ink-soft)');tx.textContent=xv;svg.appendChild(tx);});
  // legend
  labels.forEach((lb,si)=>{const y=H-40+Math.floor(si/3)*12;const x=pad.l+(si%3)*((W-pad.l)/3);
    svg.appendChild(svgEl('rect',{x,y:y-7,width:9,height:9,rx:2,fill:PAL[si%PAL.length]}));
    const t=svgEl('text',{x:x+13,y:y+1});t.setAttribute('style','font:9px var(--mono);fill:var(--ink-soft)');t.textContent=lb.length>16?lb.slice(0,15)+'…':lb;svg.appendChild(t);});
  host.appendChild(svg);
}

/* ---------- graph (top-down, two-level: semantic blocks ↔ constituent ops) ---------- */
const NW=204, NH=50, LANE=236, VGAP=56, MX=22, MY=22, OROW=30, CARD_TOP=58, OTOP=66;
const trunc=(s,n)=>s.length>n?s.slice(0,n-1)+'…':s;
const opDur=ns=>ns/1e3<1000?(ns/1e3).toFixed(0)+'µs':(ns/1e6).toFixed(2)+'ms';
function isExp(id){return view==='ops'||expanded.has(id);}
function expOps(id){const e=data.expanded&&data.expanded[M]?data.expanded[M][S]:null;return e&&e[id]?e[id]:[];}
function nodeH(b){ if(b.kind==='io')return 40; if(b.kind==='noop')return NH;
  if(!isExp(b.id))return NH; return OTOP+Math.max(1,expOps(b.id).length)*OROW+8; }
function addExpBtn(g,b,sym){
  const bx=svgEl('g',{class:'expbtn'}); bx.setAttribute('transform',`translate(${NW-21},7)`);
  bx.appendChild(svgEl('rect',{width:15,height:15,rx:4}));
  const t=svgEl('text',{x:7.5,y:11.5,'text-anchor':'middle'}); t.textContent=sym; bx.appendChild(t);
  bx.appendChild(svgEl('title',{})).textContent=sym==='＋'?'expand into constituent ops':'collapse';
  bx.onclick=e=>{e.stopPropagation(); if(window._dragged||view==='ops')return; expanded.has(b.id)?expanded.delete(b.id):expanded.add(b.id); drawGraph();};
  g.appendChild(bx);
}
// header identical to the collapsed node: heat box + centred label + "abs ms · rel%" + expand toggle
function renderHeader(g,b,ns,t,total,sym){
  g.appendChild(svgEl('rect',{width:NW,height:NH,rx:9,fill:heat(t)}));
  const ink=inkOn(t);
  const lab=svgEl('text',{class:'nlab',x:NW/2,y:18,'text-anchor':'middle',fill:ink}); lab.textContent=trunc(b.label,26); g.appendChild(lab);
  const dur=svgEl('text',{class:'ndur',x:NW/2,y:35,'text-anchor':'middle',fill:ink}); dur.textContent=`${fmtMs(ns)} ms · ${(ns/total*100).toFixed(1)}%`; g.appendChild(dur);
  addExpBtn(g,b,sym);
}
/* ---------- pan / zoom (viewBox-based; handlers attached once) ---------- */
function applyView(){ if(!gView)return; const svg=document.getElementById('svgGraph');
  svg.setAttribute('viewBox',`${gView.x} ${gView.y} ${gView.w} ${gView.h}`);
  const zl=document.getElementById('zlvl'); if(zl&&gBase)zl.textContent=Math.round(gBase.w/gView.w*100)+'%'; }
function zoomAt(px,py,k){ if(!gView)return;
  const mx=gView.x+px*gView.w, my=gView.y+py*gView.h;
  let nw=gView.w*k; nw=Math.max(gBase.w*0.15,Math.min(gBase.w*6,nw)); const nh=nw*(gBase.h/gBase.w);
  gView.w=nw; gView.h=nh; gView.x=mx-px*nw; gView.y=my-py*nh; applyView(); }
function setupPanZoom(){
  const svg=document.getElementById('svgGraph'), wrap=document.getElementById('graphwrap');
  let drag=false,sx,sy,ox,oy;
  svg.addEventListener('wheel',e=>{e.preventDefault(); const r=svg.getBoundingClientRect();
    zoomAt((e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height, e.deltaY<0?0.88:1.135);},{passive:false});
  wrap.addEventListener('mousedown',e=>{ window._dragged=false;   // clear stale drag flag on EVERY press
    if(e.target.closest&&(e.target.closest('.gzoom')||e.target.closest('.expbtn')))return;
    drag=true; sx=e.clientX; sy=e.clientY; ox=gView.x; oy=gView.y; wrap.classList.add('panning'); });
  window.addEventListener('mousemove',e=>{ if(!drag)return; const r=svg.getBoundingClientRect();
    gView.x=ox-(e.clientX-sx)/r.width*gView.w; gView.y=oy-(e.clientY-sy)/r.height*gView.h;
    if(Math.abs(e.clientX-sx)+Math.abs(e.clientY-sy)>4)window._dragged=true; applyView(); });
  window.addEventListener('mouseup',()=>{drag=false; wrap.classList.remove('panning');});
  document.querySelectorAll('.gzoom button').forEach(b=>b.onclick=()=>{
    const z=b.dataset.z; if(z==='reset'){gView={...gBase}; applyView();} else zoomAt(0.5,0.5, z==='in'?0.83:1.2); });
}
function drawGraph(){
  const svg=document.getElementById('svgGraph'); svg.innerHTML='';
  const bl=blocks[M], eg=edges[M], tim=bt();
  const total=tim?tim.total_ns:1;
  const maxNode=Math.max(...bl.filter(b=>!b.kind).map(b=>tim?tim.nodes[b.id]||0:0),1);
  let maxOp=1; const ex=data.expanded&&data.expanded[M]?data.expanded[M][S]:null;
  if(ex) for(const k in ex) for(const o of ex[k]) maxOp=Math.max(maxOp,o.total_ns);
  const levels=[...new Set(bl.map(b=>b.col))].sort((a,b)=>a-b);
  const hById={}; bl.forEach(b=>hById[b.id]=nodeH(b));
  const levelH={}; levels.forEach(l=>levelH[l]=Math.max(...bl.filter(b=>b.col===l).map(b=>hById[b.id])));
  const levelY={}; let cursor=MY; levels.forEach(l=>{levelY[l]=cursor; cursor+=levelH[l]+VGAP;});
  const maxLane=Math.max(...bl.map(b=>b.row));
  const W=MX*2+(maxLane+1)*LANE, H=cursor-VGAP+MY;
  svg.setAttribute('preserveAspectRatio','xMidYMid meet');
  // reset pan/zoom to fit only when the LAYOUT changed (mode/scenario/view/expand); preserve it on a
  // same-layout redraw (e.g. selecting a node opens the drawer without snapping the view back).
  const sig=`${M}|${S}|${view}|${[...expanded].sort().join(',')}`;
  gBase={x:0,y:0,w:W,h:H}; if(sig!==gSig||!gView){gView={...gBase};} gSig=sig; applyView();
  const defs=svgEl('defs',{});
  [['arrow',7],['arrowsm',5]].forEach(([id,s])=>{const mk=svgEl('marker',{id,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:s,markerHeight:s,orient:'auto-start-reverse'});mk.appendChild(svgEl('path',{d:'M0,0 L10,5 L0,10 z',fill:'var(--line-strong)'}));defs.appendChild(mk);});
  svg.appendChild(defs);
  const pos={}; bl.forEach(b=>{const x=MX+b.row*LANE, y=levelY[b.col]; pos[b.id]={x,y,h:hById[b.id],cx:x+NW/2};});
  // block edges (top→down)
  eg.forEach(([f,t,name,shape,dtype,layout,dist])=>{
    const a=pos[f],b=pos[t]; if(!a||!b)return;
    const x1=a.cx,y1=a.y+a.h,x2=b.cx,y2=b.y,my=(y1+y2)/2;
    const p=svgEl('path',{d:`M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`,class:'edge'});
    p.appendChild(svgEl('title',{})).textContent=`${name}\n${shape} · ${dtype} · ${layout}\n${dist}`;
    svg.appendChild(p);
    const g=svgEl('g',{class:'elabel'}); const txt=`${shape} ${dtype}`; const bw=txt.length*5+8;
    const lx=(x1+x2)/2, ly=my;
    g.appendChild(svgEl('rect',{class:'bg',x:lx-bw/2,y:ly-8,width:bw,height:13,rx:3}));
    const te=svgEl('text',{x:lx,y:ly+2,'text-anchor':'middle'}); te.textContent=txt; g.appendChild(te);
    g.style.pointerEvents='none'; svg.appendChild(g);
  });
  // nodes
  bl.forEach(b=>{
    const p=pos[b.id]; const ns=b.kind?0:(tim?tim.nodes[b.id]||0:0); const t=ns/maxNode;
    const g=svgEl('g',{class:'node'+(b.kind==='io'?' io':'')+(b.kind==='noop'?' noop':'')+(b.id===window._selNode?' sel':''),transform:`translate(${p.x},${p.y})`});
    if(b.kind){
      g.appendChild(svgEl('rect',{width:NW,height:p.h,rx:9,fill:'var(--panel-2)'}));
      const lab=svgEl('text',{class:'nlab',x:NW/2,y:p.h/2+4,'text-anchor':'middle',fill:'var(--ink)'}); lab.textContent=trunc(b.label,26); g.appendChild(lab);
      g.appendChild(svgEl('title',{})).textContent=b.label;
    } else if(isExp(b.id)){
      // card below the (unchanged) header, holding the stacked constituent ops
      g.appendChild(svgEl('rect',{class:'obox',y:CARD_TOP-2,width:NW,height:p.h-(CARD_TOP-2),rx:9}));
      const ops=expOps(b.id);
      for(let i=0;i<ops.length-1;i++){ const yA=OTOP+i*OROW+20, yB=OTOP+(i+1)*OROW;
        g.appendChild(svgEl('path',{class:'oedge',d:`M28,${yA} L28,${yB}`}));
        const tt=tensor[ops[i].id]; if(tt){const el=svgEl('text',{class:'oelabel',x:34,y:(yA+yB)/2+3}); el.setAttribute('text-anchor','start'); el.textContent=trunc(tt.split(' · ')[0],20); g.appendChild(el);}
      }
      let oy=OTOP;
      ops.forEach(o=>{
        const ot=o.total_ns/maxOp; const og=svgEl('g',{class:'onode'+(o.misc?' comp':''),transform:`translate(8,${oy})`});
        og.appendChild(svgEl('rect',{width:NW-16,height:20,rx:5,fill:heat(ot)}));
        const oink=inkOn(ot);
        const ol=svgEl('text',{class:'ol',x:6,y:13,fill:oink}); ol.setAttribute('text-anchor','start'); ol.textContent=trunc(o.label.replace(' (composite)',''),15); og.appendChild(ol);
        const od=svgEl('text',{class:'od',x:NW-22,y:13,fill:oink}); od.setAttribute('text-anchor','end'); od.textContent=`${opDur(o.total_ns)} · ${(o.total_ns/total*100).toFixed(1)}%`; og.appendChild(od);
        const tt=tensor[o.id];
        og.appendChild(svgEl('title',{})).textContent=`${o.label} · ${o.count} call${o.count>1?'s':''}\n${fmtMs(o.total_ns)} ms · ${(o.total_ns/total*100).toFixed(2)}% of trace`+(tt?`\n→ ${tt}`:(o.misc?'\n(composite / relabelled — real time, inferred wiring)':''));
        g.appendChild(og); oy+=OROW;
      });
      renderHeader(g,b,ns,t,total,'−');
    } else {
      renderHeader(g,b,ns,t,total,'＋');
    }
    if(!b.kind){ g.style.cursor='pointer'; g.onclick=()=>{if(window._dragged)return; openDrawer(b.id);};
      g.appendChild(svgEl('title',{})).textContent=`${b.label}\n${fmtMs(ns)} ms · ${(ns/total*100).toFixed(1)}% of trace\n(click for source; ＋ to expand)`;
    }
    svg.appendChild(g);
  });
  const pill=document.getElementById('graphModePill'); pill.className='mpill '+M;
  pill.querySelector('.dot').style.background=`var(--${M})`; pill.querySelector('span:last-child').textContent=M==='sparse'?'Sparse path':'Dense path';
  document.getElementById('graphNote').innerHTML=
    `${M==='sparse'?'Sparse v3.2 DSA: the indexer builds a top-k index that <span class="mono">sparse_sdpa</span> attends — bounded by k=2048, not prefix length.':'Dense v3.1 baseline: NullIndexer (no-op); <span class="mono">ring_mla</span> attends the ENTIRE prefix, so it scales with sequence length.'} `+
    `Flows top→down. Node colour = share of the ${S} trace (white→red); expanded op colour = share within the ${M}/${S} run. Durations are real per-call Tracy times (Appendix D). Composite/relabelled ops are dashed.`;
}

/* ---------- drawer ---------- */
const drawer=document.getElementById('drawer');
document.getElementById('drClose').onclick=()=>{drawer.classList.remove('open');drawer.setAttribute('aria-hidden','true');window._selNode=null;drawGraph();};
function openDrawer(id){
  const b=blocks[M].find(x=>x.id===id);const tim=bt();const ns=tim?tim.nodes[id]||0:0;const total=tim?tim.total_ns:1;
  window._selNode=id; drawGraph();
  document.getElementById('drTitle').textContent=b.label;
  const ops=expOps(id);
  let maxOpAll=1; const exAll=data.expanded&&data.expanded[M]?data.expanded[M][S]:null;
  if(exAll) for(const k in exAll) for(const o of exAll[k]) maxOpAll=Math.max(maxOpAll,o.total_ns);
  const swatch=v=>`display:inline-block;width:11px;height:11px;border-radius:3px;background:${heat(v)};border:1px solid var(--line);vertical-align:-1px;margin-right:6px`;
  const opsTbl=ops.length?`<div class="kv">Constituent ops (measured, ${S})</div><table class="iot"><thead><tr><td>op</td><td>calls</td><td>ms</td><td>%overall</td></tr></thead><tbody>`+
    ops.map(o=>{const hv=o.total_ns/maxOpAll;return `<tr><td class="mono"><span style="${swatch(hv)}"></span>${esc(o.label.replace(' (composite)',''))}${o.misc?' <span class="tag" style="color:var(--warn)">comp</span>':''}</td><td>${o.count}</td><td>${fmtMs(o.total_ns)}</td><td><span style="background:${heat(hv)};color:${inkOn(hv)};border-radius:4px;padding:2px 6px;font-weight:700;display:inline-block">${(o.total_ns/total*100).toFixed(1)}%</span></td></tr>`;}).join('')+`</tbody></table>`:'';
  const io=[];
  edges[M].filter(e=>e[1]===id).forEach(e=>io.push(['in',e]));
  edges[M].filter(e=>e[0]===id).forEach(e=>io.push(['out',e]));
  let ioRows=io.map(([dir,e])=>`<tr><td><span class="tag">${dir}</span></td><td class="mono">${esc(e[2])}</td><td class="mono">${esc(e[3])}</td><td class="mono">${esc(e[4])}·${esc(e[5])}</td><td class="mono">${esc(e[6])}</td></tr>`).join('');
  const wts=(b.weights&&b.weights.length)?`<div class="kv">Weights</div><table class="iot"><tbody>${b.weights.map(w=>`<tr><td class="mono">${esc(w[0])}</td><td class="mono">${esc(w[1])}</td><td>${esc(w[2])}</td></tr>`).join('')}</tbody></table>`:'';
  document.getElementById('drBody').innerHTML=`
    <div style="display:flex;gap:16px;margin-bottom:6px">
      <div><div class="l" style="font:600 10px/1 var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--ink-soft)">Duration (${S})</div>
        <div style="font:700 22px/1.1 var(--sans)">${fmtMs(ns)} ms</div></div>
      <div><div class="l" style="font:600 10px/1 var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--ink-soft)">Share</div>
        <div style="font:700 22px/1.1 var(--sans)">${(ns/total*100).toFixed(1)}%</div></div>
    </div>
    <p class="note" style="margin:6px 0 0">${esc(b.desc||'')}</p>
    ${b.file?`<div class="kv">Source</div><div class="pathref">${esc(b.file)}:${esc(b.lines)}<br>${esc(b.fn||'')}()</div>`:''}
    ${b.snippet?`<div class="kv">Snippet</div><pre class="code">${esc(b.snippet)}</pre>`:''}
    ${wts}
    ${opsTbl}
    ${ioRows?`<div class="kv">Tensor edges</div><table class="iot"><thead><tr><td></td><td>tensor</td><td>shape</td><td>dtype·layout</td><td>distribution</td></tr></thead><tbody>${ioRows}</tbody></table>`:''}
  `;
  drawer.classList.add('open');drawer.setAttribute('aria-hidden','false');
}

/* ---------- caveats + appendix ---------- */
function drawStatic(){
  document.getElementById('opsInfo').innerHTML=`<b>Reading the table.</b> Rows are device-collapsed per logical op call: compute ops take the <b>max</b> across the 8 chips (the critical path), collectives the <b>avg</b>. Counts are per-forward for warm/long, and summed over all 11 iterations for cold.`;
  document.getElementById('caveatBox').innerHTML=caveats.map(c=>`
    <div class="caveat ${c.sev}"><h4><span class="badge">${c.sev}</span>${esc(c.title)}</h4><p>${esc(c.body)}</p></div>`).join('');
  // A: node reference
  let nodesHtml='';
  ['sparse','dense'].forEach(m=>{
    nodesHtml+=`<h3 style="font:650 14px/1 var(--sans);margin:14px 0 8px;color:var(--${m})">${m} path</h3>`;
    blocks[m].filter(b=>!b.kind||b.kind==='noop').forEach(b=>{
      nodesHtml+=`<div style="border:1px solid var(--line);border-radius:9px;padding:11px 13px;margin-bottom:8px">
        <div style="font:650 13px/1.3 var(--sans)">${esc(b.label)} <span class="tag">${esc(b.fn||'')}</span></div>
        <div class="pathref" style="margin:3px 0 6px">${esc(b.file||'')}:${esc(b.lines||'')}</div>
        <p class="note" style="margin:0 0 7px">${esc(b.desc||'')}</p>
        ${b.snippet?`<pre class="code">${esc(b.snippet)}</pre>`:''}
      </div>`;
    });
  });
  document.getElementById('appxNodes').innerHTML=nodesHtml;
  // B: full reports
  let rep='';
  ['sparse','dense'].forEach(m=>['warm','cold','long'].forEach(sc=>{
    const e=data.modes[m][sc];
    rep+=`<h3 style="font:650 13px/1 var(--sans);margin:14px 0 6px">${m} · ${sc}</h3>`;
    if(!e){rep+=`<p class="note miss">Not measured (N/A).</p>`;return;}
    rep+=`<p class="note">source: <span class="mono">${esc(e.csv_path||'')}</span> · total ${fmtMs(e.total_ns)} ms · ${fmtInt(e.total_calls)} calls</p>`;
    rep+=`<div class="tblwrap" style="max-height:300px"><table><thead><tr><th>OP CODE</th><th>count</th><th>total ms</th><th>avg µs</th><th>%</th></tr></thead><tbody>`+
      [...e.ops].sort((a,b)=>b.total_ns-a.total_ns).map(o=>`<tr><td class="op">${esc(o.op)}</td><td>${fmtInt(o.count)}</td><td>${fmtMs(o.total_ns)}</td><td>${fmtUs(o.avg_ns)}</td><td>${o.pct.toFixed(2)}%</td></tr>`).join('')+
      `</tbody></table></div>`;
  }));
  document.getElementById('appxReports').innerHTML=rep;
  // C: meta
  const L=meta.local,G=meta.galaxy,C=meta.config;
  document.getElementById('appxMeta').innerHTML=`
    <div class="kv">Branch / commit</div>
    <p class="note"><span class="mono">${esc(meta.branch)}</span> @ <span class="mono">${esc(meta.commit)}</span><br>${esc(meta.commit_subject)}</p>
    <div class="kv">Key changes in this line of work</div>
    <ul class="note" style="margin:0;padding-left:18px">${meta.key_changes.map(k=>`<li>${esc(k)}</li>`).join('')}</ul>
    <div class="kv">Hardware</div>
    <p class="note">${esc(meta.hardware)} · mesh ${esc(meta.mesh)} · ${esc(meta.proxy)}.<br>
    Detected 8× Blackhole p150b (UMD chips 0–7). Test: <span class="mono">${esc(meta.test)}</span></p>
    <div class="kv">Sequence sizes</div>
    <table class="iot"><thead><tr><td></td><td>chunk</td><td>warm/cold cache</td><td>long cache</td><td>mesh</td></tr></thead><tbody>
      <tr><td>LoudBox (measured)</td><td class="mono">${fmtInt(L.chunk)}</td><td class="mono">${fmtInt(L.warm_cold_cache)}</td><td class="mono">${fmtInt(L.long_cache)}</td><td class="mono">SP2×TP4</td></tr>
      <tr><td>Galaxy (target)</td><td class="mono">${fmtInt(G.chunk)}</td><td class="mono">${fmtInt(G.warm_cold_cache)}</td><td class="mono">${fmtInt(G.long_cache)}</td><td class="mono">SP8×TP4</td></tr>
    </tbody></table>
    <div class="kv">Model config (per forward)</div>
    <p class="note mono">hidden ${C.hidden} · heads ${C.heads} · q_lora ${C.q_lora} · kv_lora ${C.kv_lora} · qk_rope ${C.qk_rope} · qk_nope ${C.qk_nope} · v_head ${C.v_head} · kvpe ${C.kvpe} · index_heads ${C.index_heads} · index_head_dim ${C.index_head_dim} · topk ${C.index_topk}</p>`;
  // D: method
  document.getElementById('appxMethod').innerHTML=`
    <p class="note">Tracy reports device-kernel time per <b>op code</b>, device-collapsed across the 8 chips (compute = max = critical path; collectives = avg), sliced to the <span class="mono">signpost("start")…signpost("stop")</span> region. Semantic-block time is not directly measured, so each op-code total is distributed across the blocks that emit it, weighted by the number of calls each block issues, per the code-verified block→op mapping shown on every node (Appendix A / node drawer).</p>
    <p class="note">Consequences: (1) block times sum <b>exactly</b> to the scenario total; (2) uniquely-placed ops — <span class="mono">SparseSDPA</span>, <span class="mono">RingJointSDPA</span>, <span class="mono">IndexerScore</span>, <span class="mono">TopkLargeIndices</span> — are exact; (3) a block's share of a multi-call op code (e.g. the 11 Matmuls) assumes equal cost per call, an approximation; (4) some collective ops are relabelled by ttnn (AllGather→AllBroadcast, non-minimal ReduceScatter) and composite ops (MeshPartition, Copy, Untilize/Tilize padding) decompose internally — these are attributed to their issuing block (gather / reshard / cache-format) from the source trace. The dominant <span class="mono">AllBroadcast</span> (the full-prefix KVPE gather) is placed on the prefix-gather node, which is why that node grows with sequence length in the long scenario.</p>`;
  // footer
  document.getElementById('footer').innerHTML=`Generated from Tracy dumps under <span class="mono">generated/profiler/deepseek_v32_{sparse,dense}_mla_perf/</span> · ${esc(meta.hardware)} · ${esc(meta.branch)}@${esc(meta.commit)}. Node graph verified against <span class="mono">tt/mla/mla.py</span> &amp; <span class="mono">tt/mla/indexer.py</span>.`;
}

/* ---------- wire controls ---------- */
function syncPressed(){
  document.querySelectorAll('#segScenario button').forEach(x=>x.setAttribute('aria-pressed',x.dataset.s===S));
  document.querySelectorAll('#segMode button').forEach(x=>x.setAttribute('aria-pressed',x.dataset.m===M));
}
function updateAvail(){ // grey out unmeasured (mode,scenario) combos (dense·long)
  document.querySelectorAll('#segScenario button').forEach(x=>{
    const na=!data.modes[M][x.dataset.s];
    x.disabled=na; x.style.opacity=na?'.4':''; x.style.cursor=na?'not-allowed':'pointer';
    x.title=na?`${M} · ${x.dataset.s} not measured`:'';
  });
}
document.querySelectorAll('#segScenario button').forEach(b=>b.onclick=()=>{
  if(!data.modes[M][b.dataset.s])return; S=b.dataset.s; syncPressed(); draw();});
document.querySelectorAll('#segMode button').forEach(b=>b.onclick=()=>{
  M=b.dataset.m; expanded.clear(); window._selNode=null; if(!ent())S='warm'; syncPressed(); draw();});
document.querySelectorAll('#segView button').forEach(b=>b.onclick=()=>{
  view=b.dataset.v; document.querySelectorAll('#segView button').forEach(x=>x.setAttribute('aria-pressed',x===b)); drawGraph();});
function draw(){ if(!ent())S='warm'; syncPressed(); updateAvail();
  drawSummary();drawCompare();drawOps();drawCold();drawGraph();}
drawStatic();draw();setupPanZoom();
addEventListener('resize',()=>{if(S==='cold')drawCold();});
</script>
"""

out = HTML.replace("__TITLE__", TITLE).replace("__PAYLOAD__", PAYLOAD_JSON)
open(SP + "/report.html", "w").write(out)
print("wrote", SP + "/report.html", len(out), "bytes")
