# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Every switch of mistral_large_4 in one table. Environment variable = ``MISTRAL_LARGE_4_<NAME>``; no other module of
the model reads the environment. MAX_PRECISION names the full-precision value of each precision switch.
"""

from models.demos.deepseek_v3_d_p.tt.mistral_large_4._settings_base import Setting, Settings  # noqa: F401

TABLE = {
    "NORM_MATH_FIDELITY": Setting("HiFi4", ("HiFi4", "HiFi2", "LoFi"), "math fidelity of the distributed RMSNorm"),
    "NORM_FP32_ACC": Setting(True, None, "fp32 dest accumulation in the distributed RMSNorm"),
    "NORM_STATS_DTYPE": Setting("float32", ("float32", "bfloat16"), "dtype of the sum(x^2) stats gathered over TP"),
    "ATTN_NORM_DTYPE": Setting(
        "float32",
        ("float32", "bfloat16"),
        "attn_norm's output dtype, which the indexer reads as is; float32 keeps the index keys and queries off a "
        "bf16 rounding of the normed input",
    ),
    "INDEXER_MATH_FIDELITY": Setting(
        "HiFi4", ("HiFi4", "HiFi2", "LoFi"), "math fidelity of the indexer projections, LayerNorms and RoPE"
    ),
    "INDEXER_FP32_ACC": Setting(True, None, "fp32 dest accumulation in the indexer projections and norms"),
    "INDEXER_SCORE_FIDELITY": Setting(
        "HiFi4", ("HiFi4", "HiFi2", "LoFi"), "ring_indexer_score_dsa math fidelity (its DEST is bf16)"
    ),
    "INDEXER_K_CHUNK": Setting(
        32, (32, 64, 128, 256), "score op k_chunk_size; 32 takes the path that honours the math fidelity"
    ),
    "NUM_LINKS": Setting(2, (2, 1), "fabric links used by every collective, dispatch and combine"),
    "INDEXER_TOPOLOGY": Setting("Ring", ("Linear", "Ring"), "topology of the indexer's gathers"),
    "INDEXER_SCORE": Setting(
        "ring",
        ("composed", "ring"),
        "composed: keys gathered over SP, per-head linear(relu) in fp32 and an fp32 weighted head sum; ring: "
        "ring_indexer_score_dsa (sums the heads in bf16 DEST)",
    ),
    "INDEXER_OWN_KEYS": Setting(
        "bfloat16",
        ("bfloat16", "cache"),
        "the chunk's own index keys scored as computed (all_gather over SP, before the served cache's bfp8 rounding) "
        "or as read back from the cache",
    ),
    "INDEXER_SCORE_SHIFT": Setting(
        True,
        None,
        "subtract each row's bf16 top-k cut value from the fp32 scores before the bf16 cast topk_large_indices "
        "needs, so scores near the cut keep their precision",
    ),
    "INDEXER_HEAD_SUM": Setting(
        "batched",
        ("batched", "loop"),
        "composed score's weighted head sum: batched = one fp32 matmul over all heads per block of keys, then the "
        "head weights and a sum over heads; loop = slice, matmul, slice, multiply, add per head",
    ),
    "INDEXER_HEAD_SUM_BLOCK": Setting(
        40960,
        (2560, 5120, 10240, 20480, 40960, 0),
        "keys per batched head-sum matmul, 0 = all keys in one ([1, 32, 160, block] fp32 per chip; its memory grows "
        "with the context)",
    ),
    "INDEXER_NORM": Setting(
        "fused",
        ("composed", "fused"),
        "q / k LayerNorms: composed = mean, centre, variance, rsqrt, scale, shift as fp32 ttnn ops; fused = "
        "ttnn.layer_norm",
    ),
    "INDEXER_WK_DTYPE": Setting(
        "bfloat16",
        ("bfloat16", "bfloat8_b"),
        "idx_wk [6144, 128] dtype; bf16 keeps the index keys off the bfp8 weight error",
    ),
    "INDEXER_HEAD_WEIGHT_DTYPE": Setting(
        "float32", ("float32", "bfloat16"), "dtype of the head weights x the softmax-scaling table f[p, h]"
    ),
    "INDEXER_Q_DTYPE": Setting(
        "bfloat16",
        ("float32", "bfloat16"),
        "indexer queries: float32 = wq_b output, head split and RoPE (fp32 per-position tables) in fp32; bfloat16 = "
        "nlp_create_qkv_heads + rotary_embedding_indexed",
    ),
    "INDEXER_HADAMARD": Setting(
        True,
        None,
        "rotate index keys and queries by the orthonormal Walsh-Hadamard H128 / sqrt(128) after RoPE (scores "
        "unchanged), so the served index cache holds the keys the decode's DsaIndexerKBranch writes; False = "
        "unrotated keys",
    ),
    "ATTN_MATH_FIDELITY": Setting(
        "HiFi4", ("HiFi4", "HiFi2", "LoFi"), "math fidelity of the attention projections, kv norm and RoPE"
    ),
    "ATTN_FP32_ACC": Setting(True, None, "fp32 dest accumulation in the attention matmuls and sparse_sdpa"),
    "ATTN_SDPA_FIDELITY": Setting("HiFi2", ("HiFi4", "HiFi2", "LoFi"), "sparse_sdpa math fidelity"),
    "ATTN_SDPA_K_CHUNK": Setting(
        256, (32, 64, 128, 256), "sparse_sdpa k_chunk_size (divides top-k 2048; 512 overflows L1 at K 576)"
    ),
    "ATTN_TOPOLOGY": Setting("Ring", ("Linear", "Ring"), "topology of the attention's TP collectives"),
    "ATTN_WKV_A_DTYPE": Setting(
        "bfloat16", ("bfloat16", "bfloat8_b"), "wkv_a [576, 6144] dtype (replicated); bfp8 biases the attention output"
    ),
    "ATTN_WKV_B_DTYPE": Setting("bfloat8_b", ("bfloat8_b", "bfloat16"), "absorbed wkv_b (W_uk, W_uv) dtype"),
    "ATTN_SHARED_KV_WEIGHTS": Setting(
        "bfloat16",
        ("bfloat16", "bfloat8_b"),
        "dtype of wq, W_uk, W_uv on a shared-KV layer (no wkv_a; it attends its source layer's bfp8 prefix)",
    ),
    "ATTN_CHUNK_KV": Setting(
        "bfloat16",
        ("bfloat16", "bfloat8_b"),
        "the chunk's own kvpe rows attended in bf16 (before the served cache's bfp8 rounding) or as read back",
    ),
    "ATTN_PARTIAL_DTYPE": Setting("float32", ("float32", "bfloat16"), "wo partials summed over TP in this dtype"),
    "ATTN_SDPA": Setting(
        "fused",
        ("auto", "composed", "fused"),
        "sparse attention core. fused: ttnn.transformer.sparse_sdpa (bf16 running state, approximate exp); composed: "
        "embedding gather of the top-k keys, fp32 scores and exp, normalized by the bf16 weights' own sum; auto: "
        "composed on NoPE layers, fused on RoPE layers",
    ),
    "ATTN_COMPOSED_ROWS": Setting(
        32, (8, 16, 32), "query rows per block of the composed sparse attention (160 overflows L1)"
    ),
    "ATTN_RESHARD": Setting(
        "all_to_all",
        ("all_gather", "all_to_all"),
        "heads -> sequence over TP around sparse_sdpa: all_gather + mesh_partition, or all_to_all_async_generic",
    ),
    "ATTN_PASTE": Setting(
        "slab",
        ("runs", "slab"),
        "the chunk's bf16 keys into the gathered prefix. runs: one slice + slice_write per block of rows; slab: one "
        "strided slice_write through [1, 8, max_seq / 8, 576] views (chunk-aligned chunks only)",
    ),
    "ATTN_KV_UNTILIZE": Setting(
        "fused",
        ("fused", "typecast"),
        "gathered bfp8 prefix -> bf16 ROW_MAJOR. fused: one untilize (unpacks bfp8 exactly); typecast: typecast then "
        "untilize",
    ),
    "ATTN_PASTE_GATHER": Setting(
        "row_major",
        ("tile", "row_major"),
        "the chunk's bf16 kvpe rows for the paste. tile: all_gather over SP, then untilize the gathered [chunk, "
        "576]; row_major: untilize this row's [chunk / 8, 576], then all_gather",
    ),
    "RESIDUAL_FP32_IN": Setting(
        True,
        None,
        "the attention / MLP / experts fp32 TP sums go to the residual add as fp32 (the add rounds once to bf16) "
        "instead of a typecast to bf16 first",
    ),
    "MOE_REDUCE": Setting(
        "fused",
        ("separate", "fused"),
        "separate = the routed experts and the shared expert each reduce_scatter their TP partials and moe_combine "
        "adds them; fused = the experts' partial is added into the shared expert's fp32 partial and one "
        "reduce_scatter sums both",
    ),
    "ROUTER_OUT": Setting(
        "topk",
        ("dense", "topk"),
        "dense = the router's [rows, 128] weights (4 nonzero), which the experts top-k again; topk = softmax over "
        "the router's top-k logits [rows, 4] and its expert ids handed to the experts",
    ),
    "EXPERTS_DISPATCH_IN": Setting(
        "tile",
        ("row_major", "tile"),
        "the gathered rows into dispatch: row_major = untilized first (to_layout); tile = as they are, the dispatch "
        "kernel untilizes them",
    ),
    "TOPK_SHARED": Setting(
        "alias",
        ("clone", "alias"),
        "a shared-index layer's top-k: clone = a copy of its source full layer's (ttnn.clone); alias = that tensor "
        "itself",
    ),
    "NORM_GATHER": Setting(
        "stats",
        ("stats", "input", "rms_norm"),
        "attn_norm / ffn_norm. stats = the distributed norm (local sum(x^2), all_gather of the stats, post op on the "
        "TP slice; readers all_gather the output); input = all_gather the bf16 input once and normalize the whole "
        "row on every chip (its pre / post ops overflow L1 at 6144 wide); rms_norm = the same through ttnn.rms_norm",
    ),
    "NORM_GATHER_TOPOLOGY": Setting("Ring", ("Linear", "Ring"), "topology of NORM_GATHER=input's all_gather"),
    "MM_L1_ACC": Setting(
        True, None, "packer L1 accumulation in the attention / MLP / shared-expert matmuls (not bit-identical)"
    ),
    "INDEXER_MM": Setting(
        "auto",
        ("auto", "tuned"),
        "the indexer's wk / wq_b / wproj matmuls with mm_configs.py's 2D rule (get_indexer) or ttnn's choice",
    ),
    "MOE_OVERLAP": Setting(
        "shared_dispatch",
        ("off", "shared_dispatch"),
        "shared_dispatch = the shared expert's matmuls run on grid rows 1.. while the routed experts' dispatch runs "
        "on row 0 (two sub-devices, sub_device.py; needs MOE_REDUCE=fused); a trace capture splits at the sub-device "
        "load / clear",
    ),
    "MM_CONFIG": Setting(
        "tuned",
        ("auto", "tuned", "tuned_l1"),
        "auto = ttnn picks the matmul program configs; tuned = the attention / shared-expert / MLP matmuls use "
        "mm_configs.py's 2D multicast program configs; tuned_l1 = the same with the fp32 TP partials (wo, w2) in L1",
    ),
    "MLP_MATH_FIDELITY": Setting("HiFi4", ("HiFi4", "HiFi2", "LoFi"), "math fidelity of the dense MLP matmuls"),
    "MLP_FP32_ACC": Setting(True, None, "fp32 dest accumulation in the dense MLP matmuls"),
    "MLP_INTERMEDIATE_DTYPE": Setting(
        "bfloat16", ("bfloat16", "float32"), "gate, up and silu(gate) * up of the dense MLP"
    ),
    "MLP_PARTIAL_DTYPE": Setting(
        "float32", ("float32", "bfloat16"), "w2 partials summed over TP in this dtype (a bf16 TP reduce biases the sum)"
    ),
    "MLP_TOPOLOGY": Setting("Ring", ("Linear", "Ring"), "topology of the MLP's TP collectives"),
    "ROUTER_MATH_FIDELITY": Setting("HiFi4", ("HiFi4", "HiFi2", "LoFi"), "math fidelity of the router logits"),
    "ROUTER_FP32_ACC": Setting(True, None, "fp32 dest accumulation in the router logits and row sums"),
    "ROUTER_WEIGHT_DTYPE": Setting(
        "float32", ("float32", "bfloat16", "bfloat8_b"), "dtype of the replicated gate [128, 6144]"
    ),
    "ROUTER_TOPOLOGY": Setting("Ring", ("Linear", "Ring"), "topology of the router's x_full gather over TP"),
    "FFN_GATHER": Setting(
        "shared",
        ("each", "shared"),
        "each = the router, routed experts and shared expert each all_gather the ffn_norm output over TP; shared = "
        "one gather per block, reused (tp_gather.py)",
    ),
    "ROUTER_SELECT": Setting(
        "threshold",
        ("eq", "threshold"),
        "the router's 0 / 1 top-k selection. eq: each top-k id compared with every expert id; threshold: logits >= "
        "the k-th largest top-k value (an exact fp32 tie at the k-th value selects one more expert)",
    ),
    "EXPERTS_MATH_FIDELITY": Setting("HiFi4", ("HiFi4", "HiFi2", "LoFi"), "math fidelity of the routed expert matmuls"),
    "EXPERTS_FP32_ACC": Setting(True, None, "fp32 dest accumulation in the routed expert matmuls"),
    "EXPERTS_INTERMEDIATE_DTYPE": Setting(
        "float32",
        ("float32", "bfloat16"),
        "gate, up and silu(gate) * up of each routed expert (bf16 intermediates widen the norm ratio)",
    ),
    "EXPERTS_PARTIAL_DTYPE": Setting(
        "float32", ("float32", "bfloat16"), "the 4 dispatch groups' partial outputs summed over TP in this dtype"
    ),
    "EXPERTS_CAPACITY_FACTOR": Setting(
        4,
        (1, 2, 3, 4),
        "dispatch buffer = factor x the group's tokens (4 = top-4 all on one chip, never drops a token)",
    ),
    "EXPERTS_WEIGHT_DTYPE": Setting(
        "bfloat4_b",
        ("bfloat4_b", "bfloat8_b"),
        "routed expert weights; bfloat4_b fits all 72 layers on one Galaxy, bfloat8_b does not",
    ),
    "EXPERTS_IMPL": Setting(
        "unified",
        ("unified", "loop"),
        "unified_routed_expert_moe (each expert on its own token count) or a per-expert loop over the whole 5120-row "
        "region of every expert (not trace-safe)",
    ),
    "SERVE_CONTEXT": Setting(
        "default",
        ("default", "long"),
        "slot geometry. default: the runner's (2 slots x 56320); long: one slot of 1044480 tokens "
        "(max_position_embeddings in whole chunks), SEQ_BUFFERS=shared, SSMAX_TABLE=shared_log, ATTN_KV_ROWS=extent, "
        "a 128 MB trace region and 4352-byte fabric packets (PREFILL_* / ML4_FABRIC_PAYLOAD / switches set by the "
        "caller win)",
    ),
    "SEQ_BUFFERS": Setting(
        "per_layer",
        ("per_layer", "shared"),
        "the max_seq-sized device tensors that are the same on every layer (attention / indexer gather scratches, "
        "RoPE and position tables, the scaling tables' zero sin): one copy per layer or one per mesh (seqbuf.py). "
        "Read when a module is built",
    ),
    "SSMAX_TABLE": Setting(
        "per_layer",
        ("per_layer", "shared_log"),
        "the softmax scaling f[p, h] = s_h log(p + 1) + b_h of attention and indexer. per_layer: a host fp32 table "
        "per layer of every position; shared_log: one fp32 log(p + 1) table per mesh, f = s x log + b in fp32 on the "
        "device per chunk. Read when a module is built",
    ),
    "ATTN_KV_ROWS": Setting(
        "max_seq",
        ("max_seq", "gathered", "extent"),
        "the gathered bfp8 prefix -> bf16 ROW_MAJOR for sparse_sdpa. max_seq: the whole [max_seq, 576] scratch; "
        "gathered: only the gathered slabs (untilize_with_unpadding), eager path only; extent: the traced path "
        "untilizes only the gathered slabs, the extent read on the device from the chunk start, full-size output "
        "(eager as max_seq)",
    ),
    "KV_CACHE_FORMAT": Setting(
        "bf16_rm",
        ("bf16_rm", "bfp8_tile"),
        "served kvpe cache encoding. bf16_rm: bfloat16 ROW_MAJOR, one 32 x 576 record (36,864 B) per ND shard, the "
        "decode ring's format, so records copy unchanged; bfp8_tile: bfloat8_b TILE records (19,584 B), half the "
        "DRAM, not migratable to the ring",
    ),
    "KV_MIRRORS": Setting(
        True,
        None,
        "every layer gets a kvpe region; a shared-KV layer writes its source layer's rows into its own region, as "
        "the decode ring's table expects one KV entry per layer. False: shared-KV layers own no region",
    ),
    "EXPERTS_TOPOLOGY": Setting("Ring", ("Linear", "Ring"), "topology of the experts' dispatch / combine (axis 0)"),
    "EXPERTS_TP_TOPOLOGY": Setting(
        "Ring", ("Linear", "Ring"), "topology of the experts' TP (axis 1) all_gather and reduce_scatter"
    ),
}

MAX_PRECISION = {
    "NORM_MATH_FIDELITY": "HiFi4",
    "NORM_FP32_ACC": True,
    "NORM_STATS_DTYPE": "float32",
    "ATTN_NORM_DTYPE": "float32",
    "INDEXER_MATH_FIDELITY": "HiFi4",
    "INDEXER_FP32_ACC": True,
    "INDEXER_SCORE_FIDELITY": "HiFi4",
    "INDEXER_K_CHUNK": 32,
    "INDEXER_SCORE": "composed",
    "INDEXER_OWN_KEYS": "bfloat16",
    "INDEXER_SCORE_SHIFT": True,
    "INDEXER_NORM": "composed",
    "INDEXER_WK_DTYPE": "bfloat16",
    "INDEXER_HEAD_WEIGHT_DTYPE": "float32",
    "INDEXER_Q_DTYPE": "float32",
    "ATTN_MATH_FIDELITY": "HiFi4",
    "ATTN_FP32_ACC": True,
    "ATTN_SDPA_FIDELITY": "HiFi4",
    "ATTN_PARTIAL_DTYPE": "float32",
    "ATTN_WKV_A_DTYPE": "bfloat16",
    "ATTN_SHARED_KV_WEIGHTS": "bfloat16",
    "ATTN_CHUNK_KV": "bfloat16",
    "MLP_MATH_FIDELITY": "HiFi4",
    "MLP_FP32_ACC": True,
    "MLP_PARTIAL_DTYPE": "float32",
    "ROUTER_MATH_FIDELITY": "HiFi4",
    "ROUTER_FP32_ACC": True,
    "ROUTER_WEIGHT_DTYPE": "float32",
    "EXPERTS_MATH_FIDELITY": "HiFi4",
    "EXPERTS_FP32_ACC": True,
    "EXPERTS_INTERMEDIATE_DTYPE": "float32",
    "EXPERTS_PARTIAL_DTYPE": "float32",
    "EXPERTS_WEIGHT_DTYPE": "bfloat8_b",
}

settings = Settings("MISTRAL_LARGE_4_", TABLE, max_precision=MAX_PRECISION)
