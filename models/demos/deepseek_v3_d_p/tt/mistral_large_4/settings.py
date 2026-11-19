# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Every switch of mistral_large4_1t in one table. Environment variable = ``MISTRAL_LARGE4_1T_<NAME>``. Nothing else in tt/ or
bringup/hooks.py reads the environment (the framework's settings lint fails it).

Each precision switch (math fidelity, fp32 accumulation) also names its full-precision value in MAX_PRECISION:
component and swap tests run with it; the shipped defaults are judged end to end by the ladder. Weights and the KV
cache are bfloat8_b from the start (the prefill spec's dataformats), not switches lowered later.
"""

from models.demos.deepseek_v3_d_p.tt.mistral_large_4._settings_base import Setting, Settings  # noqa: F401  (Setting: the entries agents add to TABLE

TABLE = {
    "NORM_MATH_FIDELITY": Setting("HiFi4", ("HiFi4", "HiFi2", "LoFi"), "rule 7: HiFi4 (C.attn_dense.attn_norm)"),
    "NORM_FP32_ACC": Setting(True, None, "rule 7: fp32 dest accumulation in the distributed RMSNorm"),
    "NORM_STATS_DTYPE": Setting(
        "float32", ("float32", "bfloat16"), "plan: fp32 sum(x^2) stats gathered over TP (C.attn_dense.attn_norm)"
    ),
    "ATTN_NORM_DTYPE": Setting(
        "float32",
        ("float32", "bfloat16"),
        "attn_norm's output dtype, which the indexer reads as is: fp32 keeps the index keys and queries off a bf16 "
        "rounding of the normed input (layer 39 swap overlap 0.99468 with bf16, gate 0.995; S.attn_nope_shared_kv_moe.02)",
    ),
    "INDEXER_MATH_FIDELITY": Setting(
        "HiFi4", ("HiFi4", "HiFi2", "LoFi"), "rule 7: indexer projections, LayerNorms, RoPE (C.attn_dense.indexer)"
    ),
    "INDEXER_FP32_ACC": Setting(True, None, "rule 7: fp32 dest accumulation in the indexer projections and norms"),
    "INDEXER_SCORE_FIDELITY": Setting(
        "HiFi4", ("HiFi4", "HiFi2", "LoFi"), "ring_indexer_score_dsa math fidelity (its DEST is bf16 by the op)"
    ),
    "INDEXER_K_CHUNK": Setting(
        32, (32, 64, 128, 256), "known_issues: k_chunk_size 32 takes the score op's fidelity-honouring path"
    ),
    "NUM_LINKS": Setting(
        2, (2, 1), "fabric links every collective, dispatch and combine uses; 2: bit-identical, -19% chunk (acc-03, fast-02)"
    ),
    "INDEXER_TOPOLOGY": Setting("Ring", ("Linear", "Ring"), "ring topology of the indexer's gathers"),
    "INDEXER_SCORE": Setting(
        "ring",
        ("composed", "ring"),
        "composed: keys gathered over SP, per-head linear(relu) in fp32, fp32 weighted head sum (C.attn_moe.indexer: "
        "layer 1 top-k overlap 0.989 with ring); ring: ring_indexer_score_dsa (sums the heads in bf16 DEST). perf 6: "
        "ring by default, the indexer step 4.5 -> 1.6 s per chunk, logits PCC 0.9660 -> 0.9670 (PERF.md)",
    ),
    "INDEXER_OWN_KEYS": Setting(
        "bfloat16",
        ("bfloat16", "cache"),
        "the chunk's own index keys scored as computed (all_gather over SP, before the served cache's bfp8 rounding; fp32 "
        "with an fp32 ATTN_NORM_DTYPE) or as read back from the cache (C.attn_nope_shared_kv_moe.indexer)",
    ),
    "INDEXER_SCORE_SHIFT": Setting(
        True,
        None,
        "subtract each row's bf16 top-k cut value from the fp32 scores before the bf16 cast topk_large_indices needs, so "
        "the scores near the cut keep their precision (layer 39 overlap 0.9892 without; C.attn_nope_shared_kv_moe.indexer)",
    ),
    "INDEXER_HEAD_SUM": Setting(
        "batched",
        ("batched", "loop"),
        "perf 4: the composed score's weighted head sum: batched = one fp32 matmul over all heads per block of keys, "
        "then the head weights and a sum over heads; loop = the bring-up's 5 ops per head (slice, matmul, slice, "
        "multiply, add)",
    ),
    "INDEXER_HEAD_SUM_BLOCK": Setting(
        40960,
        (2560, 5120, 10240, 20480, 40960, 0),
        "keys per batched head-sum matmul, 0 = all keys in one ([1, 32, 160, block] fp32 per chip: 840 MB at 40960). "
        "Warm chunk at position 51200 with fused SDPA: 10.1 s at 10240, 9.4 s at 20480, 9.1 s at 40960, 9.0 s at 0 "
        "(all 56320 keys; its memory grows with the context, so 40960 is the default)",
    ),
    "INDEXER_NORM": Setting(
        "fused",
        ("composed", "fused"),
        "q / k LayerNorms: composed = mean, centre, variance, rsqrt, scale, shift as fp32 ttnn ops; fused = ttnn.layer_norm "
        "(layer 39 index keys rel 0.0013 vs fp32 from fp32 input, composed 0.0003; S.attn_nope_shared_kv_moe.02). perf 6: "
        "fused by default, -0.3 to -0.6 s per chunk, logits PCC unchanged (PERF.md)",
    ),
    "INDEXER_WK_DTYPE": Setting(
        "bfloat16",
        ("bfloat16", "bfloat8_b"),
        "idx_wk [6144, 128] (0.8 MB): bf16 keeps the index keys off the bfp8 weight error (layer 1 swap overlap 0.9947 "
        "with bfp8, gate 0.995; S.attn_moe.02)",
    ),
    "INDEXER_HEAD_WEIGHT_DTYPE": Setting(
        "float32",
        ("float32", "bfloat16"),
        "head weights x the softmax-scaling table f[p, h] computed and kept in this dtype (S.attn_moe.02)",
    ),
    "INDEXER_Q_DTYPE": Setting(
        "bfloat16",
        ("float32", "bfloat16"),
        "indexer queries: float32 = wq_b output, head split and RoPE (composed, fp32 per-position tables) in fp32; "
        "bfloat16 = nlp_create_qkv_heads + rotary_embedding_indexed (bf16 only) (S.attn_moe.02). perf 6: bfloat16 by "
        "default, -0.2 to -0.5 s per chunk, logits PCC -0.001 (PERF.md)",
    ),
    "INDEXER_HADAMARD": Setting(
        True,
        None,
        "index keys and queries rotated by the orthonormal Walsh-Hadamard H128 / sqrt(128) after RoPE (the scores are "
        "unchanged): the served index cache then holds the keys the Reload decode's DsaIndexerKBranch writes, so the KV "
        "manager can copy them into its cache (the P/D deployment sets it); False = unrotated keys, as the bring-up",
    ),
    "ATTN_MATH_FIDELITY": Setting(
        "HiFi4", ("HiFi4", "HiFi2", "LoFi"), "rule 7: attention projections, kv norm, RoPE (C.attn_dense.attention)"
    ),
    "ATTN_FP32_ACC": Setting(True, None, "rule 7: fp32 dest accumulation in the attention matmuls and sparse_sdpa"),
    "ATTN_SDPA_FIDELITY": Setting("HiFi2", ("HiFi4", "HiFi2", "LoFi"), "sparse_sdpa math fidelity"),
    "ATTN_SDPA_K_CHUNK": Setting(
        256, (32, 64, 128, 256), "sparse_sdpa k_chunk_size (divides top-k 2048; 512 overflows L1 at K 576)"
    ),
    "ATTN_TOPOLOGY": Setting("Ring", ("Linear", "Ring"), "topology of the attention's TP collectives"),
    "ATTN_WKV_A_DTYPE": Setting(
        "bfloat16",
        ("bfloat16", "bfloat8_b"),
        "wkv_a [576, 6144] (7 MB per chip, replicated): bfp8 alone puts layer 1's attention at rel 0.0056, median row "
        "norm -0.41 %, row ratio min 0.986 (CPU emulation; wq / wkv_b / wo together < 0.001); S.attn_moe.03",
    ),
    "ATTN_WKV_B_DTYPE": Setting("bfloat8_b", ("bfloat8_b", "bfloat16"), "absorbed wkv_b (W_uk, W_uv) dtype"),
    "ATTN_SHARED_KV_WEIGHTS": Setting(
        "bfloat16",
        ("bfloat16", "bfloat8_b"),
        "wq, W_uk, W_uv of a shared-KV layer (no wkv_a; it attends its source layer's bfp8 prefix): layer 39 attention "
        "vs golden worst row 0.0550 with bfp8 (limit 0.0515). bf16 W_uk + W_uv everywhere fails layer 0 (column 1327 "
        "rel 0.081, limit 0.024); S.attn_nope_shared_kv_moe.03",
    ),
    "ATTN_CHUNK_KV": Setting(
        "bfloat16",
        ("bfloat16", "bfloat8_b"),
        "the chunk's own kvpe rows attended in bf16 (before the served cache's bfp8 rounding) or as read back",
    ),
    "ATTN_PARTIAL_DTYPE": Setting(
        "float32", ("float32", "bfloat16"), "wo partials summed over TP in this dtype (layer 0 column 5022 cancels)"
    ),
    "ATTN_SDPA": Setting(
        "fused",
        ("auto", "composed", "fused"),
        "perf 5: fused by default (warm chunk 16.3 s -> 10.1 s, logits / final norm PCC unchanged at 0.966 / 0.969 against "
        "the golden; the bring-up's per-op gate on layer 3 is exceeded, the end-to-end one is not). sparse attention core. fused: ttnn.transformer.sparse_sdpa (bf16 running state, approximate exp; op rel 0.007 "
        "at layer 3, 0.019 at layer 1). composed: embedding gather of the top-k keys, fp32 scores and exp, normalized "
        "by the bf16 weights' own sum (op rel 0.003 at layers 1 and 3). auto: composed on NoPE layers (layer 3 fused: "
        "worst column 1.7 x its limit), fused on RoPE layers (layer 1 composed: median row norm -0.48 %, limit 0.49 %, "
        "as with an exact core; the fused op's +0.1 % offsets it). That -0.48 % was bfp8 wkv_a's (ATTN_WKV_A_DTYPE): with "
        "wkv_a bf16 the composed core is unbiased at layer 1 and fused fails the swap row norm ratio, so composed is the "
        "default on every layer (S.attn_moe.03)",
    ),
    "ATTN_COMPOSED_ROWS": Setting(
        32, (8, 16, 32), "query rows per block of the composed sparse attention (160 overflows L1: perf 4 sweep)"
    ),
    "ATTN_RESHARD": Setting(
        "all_to_all",
        ("all_gather", "all_to_all"),
        "heads -> sequence over TP around sparse_sdpa: all_gather + mesh_partition, or all_to_all_async_generic (perf 5 "
        "default: 0.3-0.6 s faster per warm chunk, bit-identical output)",
    ),
    "ATTN_PASTE": Setting(
        "slab",
        ("runs", "slab"),
        "perf 8: the chunk's bf16 keys into the gathered prefix. runs: one slice + slice_write per block of rows (2 x 8"
        " ops); slab: one strided slice_write through [1, 8, max_seq / 8, 576] views (chunk-aligned chunks only)",
    ),
    "ATTN_KV_UNTILIZE": Setting(
        "fused",
        ("fused", "typecast"),
        "perf 8: gathered bfp8 prefix -> bf16 ROW_MAJOR. fused: one untilize (unpacks bfp8 exactly); typecast: typecast then"
        " untilize",
    ),
    "ATTN_PASTE_GATHER": Setting(
        "row_major",
        ("tile", "row_major"),
        "perf 12: the chunk's bf16 kvpe rows for the paste. tile: all_gather over SP, then untilize the gathered "
        "[chunk, 576]; row_major: untilize this row's [chunk / 8, 576], then all_gather (8x less untilize work)",
    ),
    "RESIDUAL_FP32_IN": Setting(
        True,
        None,
        "perf 12: the attention / MLP / experts fp32 TP sums go to the residual add as fp32 (the add rounds once to "
        "bf16) instead of a typecast to bf16 first: one op fewer per sum",
    ),
    "MOE_REDUCE": Setting(
        "fused",
        ("separate", "fused"),
        "perf 12: separate = the routed experts and the shared expert each reduce_scatter their TP partials and "
        "moe_combine adds them; fused = the experts' partial is added into the shared expert's fp32 partial and one "
        "reduce_scatter sums both (one collective and two ops fewer per MoE layer)",
    ),
    "ROUTER_OUT": Setting(
        "topk",
        ("dense", "topk"),
        "perf 13: dense = the router's [rows, 128] weights (4 nonzero), which the experts top-k again; topk = softmax over "
        "the router's own top-k logits [rows, 4] and its expert ids handed to the experts (5 ops fewer per MoE layer)",
    ),
    "EXPERTS_DISPATCH_IN": Setting(
        "tile",
        ("row_major", "tile"),
        "perf 14: the gathered rows into dispatch: row_major = untilized first (to_layout); tile = as they are, the "
        "dispatch kernel untilizes them (one op fewer per MoE layer, same bf16 rows)",
    ),
    "TOPK_SHARED": Setting(
        "alias",
        ("clone", "alias"),
        "perf 13: a shared-index layer's top-k: clone = a copy of its source full layer's (ttnn.clone); alias = that "
        "tensor itself (one op fewer on each of the 28 shared-index layers)",
    ),
    "NORM_GATHER": Setting(
        "stats",
        ("stats", "input", "rms_norm"),
        "perf 13: attn_norm / ffn_norm. stats = the distributed norm (local sum(x^2), all_gather of the stats, post op on "
        "the TP slice; every reader then all_gathers the output); input = all_gather the bf16 input once, normalize the "
        "whole row on every chip and hand it on full width (the readers' gathers go; its pre / post ops overflow L1 at "
        "6144 wide, fast-07 / fast-08); rms_norm = the same through ttnn.rms_norm (large-row kernels)",
    ),
    "NORM_GATHER_TOPOLOGY": Setting("Ring", ("Linear", "Ring"), "topology of NORM_GATHER=input's all_gather"),
    "MM_L1_ACC": Setting(
        True,
        None,
        "perf 14: packer L1 accumulation in the attention / MLP / shared-expert matmuls (acc-07-mm: wq 0.154 -> 0.139 ms, "
        "not bit-identical). On by default: -2.7 ms per chunk on the accurate settings, -1.5 ms on the fast ones, logits "
        "PCC +0.001 / +0.003 (fast-10)",
    ),
    "INDEXER_MM": Setting(
        "auto",
        ("auto", "tuned"),
        "perf 16: the indexer's wk / wq_b / wproj matmuls with tt/mm_configs.py's 2D rule (get_indexer) or ttnn's choice",
    ),
    "MOE_OVERLAP": Setting(
        "shared_dispatch",
        ("off", "shared_dispatch"),
        "perf 16: shared_dispatch = the shared expert's matmuls run on grid rows 1.. while the routed experts' dispatch "
        "runs on row 0 (two sub-devices, tt/sub_device.py; with MOE_REDUCE=fused); a trace capture splits at the "
        "sub-device load / clear. Default (perf 16): 727.5 -> 708 ms per chunk, bit-identical (p2-03)",
    ),
    "MM_CONFIG": Setting(
        "tuned",
        ("auto", "tuned", "tuned_l1"),
        "perf 12: auto = ttnn picks the matmul program configs; tuned = the attention / shared-expert / MLP matmuls "
        "run tt/mm_configs.py's 2D multicast program configs (per-chip microbenchmark acc-07-mm: wo 0.39 -> 0.18 ms, "
        "shared w2 0.32 -> 0.15 ms, bit-identical); tuned_l1 = the same with the fp32 TP partials (wo, w2) in L1",
    ),
    "MLP_MATH_FIDELITY": Setting("HiFi4", ("HiFi4", "HiFi2", "LoFi"), "rule 7: dense MLP matmuls (C.attn_dense.mlp)"),
    "MLP_FP32_ACC": Setting(True, None, "rule 7: fp32 dest accumulation in the dense MLP matmuls"),
    "MLP_INTERMEDIATE_DTYPE": Setting(
        "bfloat16", ("bfloat16", "float32"), "gate, up and silu(gate) * up of the dense MLP"
    ),
    "MLP_PARTIAL_DTYPE": Setting(
        "float32", ("float32", "bfloat16"), "w2 partials summed over TP in this dtype (a bf16 TP reduce biases the sum)"
    ),
    "MLP_TOPOLOGY": Setting("Ring", ("Linear", "Ring"), "topology of the MLP's TP collectives"),
    "ROUTER_MATH_FIDELITY": Setting("HiFi4", ("HiFi4", "HiFi2", "LoFi"), "rule 7: router logits (C.attn_moe.router)"),
    "ROUTER_FP32_ACC": Setting(True, None, "rule 7: fp32 dest accumulation in the router logits and row sums"),
    "ROUTER_WEIGHT_DTYPE": Setting(
        "float32",
        ("float32", "bfloat16", "bfloat8_b"),
        "plan: the gate [128, 6144] replicated in fp32 (3.1 MB per chip)",
    ),
    "ROUTER_TOPOLOGY": Setting("Ring", ("Linear", "Ring"), "topology of the router's x_full gather over TP"),
    "FFN_GATHER": Setting(
        "shared",
        ("each", "shared"),
        "perf 7: each = the router, routed experts and shared expert each all_gather the ffn_norm output over TP; "
        "shared = one gather per block, reused (tt/tp_gather.py). Sweep 5 eager: 7176 -> 6999 ms, outputs identical",
    ),
    "ROUTER_SELECT": Setting(
        "threshold",
        ("eq", "threshold"),
        "perf 7: the router's 0 / 1 top-k selection. eq: each of the k top-k ids compared with every expert id (2k + 1"
        " ops); threshold: logits >= the k-th largest top-k value, and the max taken from the top-k values (3 ops; an"
        " exact fp32 tie at the k-th value would select one more expert). Sweep 5 eager: 7176 -> 6341 ms, logits PCC"
        " 0.9649 -> 0.9666",
    ),
    "EXPERTS_MATH_FIDELITY": Setting(
        "HiFi4", ("HiFi4", "HiFi2", "LoFi"), "rule 7: routed expert matmuls (C.attn_moe.experts)"
    ),
    "EXPERTS_FP32_ACC": Setting(True, None, "rule 7: fp32 dest accumulation in the routed expert matmuls"),
    "EXPERTS_INTERMEDIATE_DTYPE": Setting(
        "float32",
        ("float32", "bfloat16"),
        "gate, up and silu(gate) * up of each routed expert (known_issues: bf16 intermediates widen the norm ratio)",
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
        "routed expert weights: bfloat4_b fits all 72 layers on one Galaxy (about 18.1 GB per chip; bfloat8_b would "
        "need about 34.2 GB against a 27.2 GB budget); the bring-up's layer subset ran bfloat8_b",
    ),
    "EXPERTS_IMPL": Setting(
        "unified",
        ("unified", "loop"),
        "perf 3: unified_routed_expert_moe (each expert on its own token count) or the bring-up's per-expert loop over "
        "the whole 5120-row region of every expert. loop is not trace-safe: sweep 5 traced it to final_norm PCC 0.0",
    ),
    "SERVE_CONTEXT": Setting(
        "default",
        ("default", "long"),
        "long context: model/serve.py's slot geometry. default: the runner's (2 slots x 56320); long: one slot of "
        "1044480 tokens (204 chunks, max_position_embeddings in whole chunks), SEQ_BUFFERS=shared, "
        "SSMAX_TABLE=shared_log, ATTN_KV_ROWS=extent, a 128 MB trace region, 4352-byte fabric packets (12288 hung "
        "the 1M build at long prefixes; PREFILL_* / ML4_FABRIC_PAYLOAD / switches set by the caller win)",
    ),
    "SEQ_BUFFERS": Setting(
        "per_layer",
        ("per_layer", "shared"),
        "long ISL 1: the max_seq-sized device tensors that are the same on every layer (attention / indexer gather "
        "scratches, RoPE and position tables, the scaling tables' zero sin), one copy per layer or one per mesh "
        "(tt/seqbuf.py; -56 KB per token per chip, bit-identical). Read when a module is built",
    ),
    "SSMAX_TABLE": Setting(
        "per_layer",
        ("per_layer", "shared_log"),
        "long ISL 3: the softmax scaling f[p, h] = s_h log(p + 1) + b_h of attention and indexer. per_layer: a table "
        "per layer of every position (host fp32); shared_log: one fp32 log(p + 1) table per mesh, f = s x log + b in "
        "fp32 on the device per chunk (eager: RowTables slices; traced: one SplitTables selection per chunk). Read "
        "when a module is built",
    ),
    "ATTN_KV_ROWS": Setting(
        "max_seq",
        ("max_seq", "gathered", "extent"),
        "long ISL 3: the gathered bfp8 prefix -> bf16 ROW_MAJOR for sparse_sdpa. max_seq: the whole [max_seq, 576] "
        "scratch; gathered: only the gathered slabs (untilize_with_unpadding), eager path only (its shape follows the "
        "chunk, so the one-capture trace keeps max_seq); extent (long context fix 1, needs the tt-metal untilize with "
        "gathered_prefix_tensor): the metadata path untilizes only the gathered slabs, the extent read on the device "
        "from the chunk start (one capture for every chunk), full-size output; the eager path as max_seq",
    ),
    "KV_CACHE_FORMAT": Setting(
        "bf16_rm",
        ("bf16_rm", "bfp8_tile"),
        "served kvpe cache encoding. bf16_rm: bfloat16 ROW_MAJOR, one 32 x 576 record (36,864 B) per ND shard, the "
        "tt-blaze decode ring's format, so the KV manager copies records unchanged; bfp8_tile: bfloat8_b TILE records "
        "(19,584 B), half the DRAM, not migratable to the ring",
    ),
    "KV_MIRRORS": Setting(
        True,
        None,
        "every layer gets a kvpe region; a shared-KV layer writes its source layer's rows into its own region, as the "
        "decode ring's table expects one KV entry per layer. False: shared-KV layers own no region",
    ),
    "EXPERTS_TOPOLOGY": Setting(
        "Ring", ("Linear", "Ring"), "experts' dispatch / combine (axis 0) topology; Ring: bit-identical, -5% (fast-01)"
    ),
    "EXPERTS_TP_TOPOLOGY": Setting(
        "Ring",
        ("Linear", "Ring"),
        "perf 8: topology of the experts' TP (axis 1) all_gather and reduce_scatter (since perf 13 the fused MoE "
        "reduce_scatter). perf 17: Ring (fp32 [640, 6144] reduce_scatter 0.31 -> 0.17 ms, p2-01; 56k ISL 7.14 -> 7.05 s "
        "on one node, logits PCC 0.9679 -> 0.9646, p2-07)",
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

settings = Settings("MISTRAL_LARGE4_1T_", TABLE, max_precision=MAX_PRECISION)
