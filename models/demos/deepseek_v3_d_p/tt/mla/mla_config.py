# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Optimal matmul and SDPA configurations for the MLA module, keyed by local sequence length
(per-device after SP sharding). Configs sourced from op_unit_tests/test_mla_matmuls.py
and op_unit_tests/test_ring_joint_mla.py.

Production local seq_len values:
  - 128k total / 8 SP devices = 16384 per device
  - 100k total / 8 SP devices = 12800 per device
  - 128k total / 32 SP devices = 4096 per device
  - 100k total / 32 SP devices = 3200 per device
"""

import ttnn

# Available core grid is 12x10, but due to di/dt and throttling problems, use 11x10 temporarily
COMPUTE_GRID = (11, 10)

# GLM-5.1/5.2 share num_heads=64 and q_lora_rank=2048 (DeepSeek v3.2 is 128/1536, Kimi is 64/1536),
# so a single tagged candidate below disambiguates both GLM variants from Kimi/DeepSeek at the same
# seq_len_local=640 slot. Sourced from op_unit_tests/test_mla_matmuls_glm_chunked.py's BEST dict
# (2x4 Blackhole loudbox proxy for the production 8x4 chunk-5120/sp8 shape); see
# GLM52_MLA_MATMUL_TUNING.md for the tuning process/results.
_GLM_TAGS = {"num_heads": 64, "q_lora_rank": 2048, "chunked_only": True}
# Indexer linears have no separate single-shot shape (see the indexer.* entries below), so they
# must NOT carry chunked_only -- only the disambiguating num_heads/q_lora_rank tags.
_GLM_INDEXER_TAGS = {"num_heads": 64, "q_lora_rank": 2048}

MLA_MATMUL_CONFIG = {
    # hidden_states @ q_a_proj_weight
    "q_a_proj": {
        640: [
            {
                "num_heads": 64,
                "q_lora_rank": 1536,
                "chunked_only": True,
                "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    per_core_M=2,
                    per_core_N=5,
                    transpose_mcast=False,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
                "out_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_dtype": ttnn.bfloat16,
            },
            {
                **_GLM_TAGS,
                "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=6,
                    per_core_M=2,
                    per_core_N=6,
                    transpose_mcast=False,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
                "out_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_dtype": ttnn.bfloat16,
            },
        ],
        4096: {
            "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=14,
                out_subblock_h=1,
                out_subblock_w=5,
                per_core_M=13,
                per_core_N=5,
                transpose_mcast=False,
                fuse_batch=False,
                fused_activation=None,
            ),
            "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        },
        3200: {
            "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=14,
                out_subblock_h=5,
                out_subblock_w=1,
                per_core_M=10,
                per_core_N=5,
                transpose_mcast=False,
                fuse_batch=False,
                fused_activation=None,
            ),
            "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        },
    },
    # tt_q @ q_b_proj_weight (after layernorm)
    "q_b_proj": {
        640: [
            {
                "num_heads": 64,
                "q_lora_rank": 1536,
                "chunked_only": True,
                "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=3,
                    per_core_M=2,
                    per_core_N=9,
                    transpose_mcast=False,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                "act_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_dtype": ttnn.bfloat16,
            },
            {
                # Also serves indexer.wq_b (identical per-chip shape) via the qr latent's shared
                # act_mem_config (see mla.py::_q_a_latent's norm_memory_config).
                **_GLM_TAGS,
                "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=6,
                    per_core_M=2,
                    per_core_N=12,
                    transpose_mcast=False,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                "act_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_dtype": ttnn.bfloat16,
            },
        ],
        4096: {
            "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=6,
                per_core_M=13,
                per_core_N=18,
                transpose_mcast=False,
                fuse_batch=False,
                fused_activation=None,
            ),
            "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        },
        3200: {
            "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=6,
                per_core_M=10,
                per_core_N=18,
                transpose_mcast=False,
                fuse_batch=False,
                fused_activation=None,
            ),
            "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        },
    },
    # tt_q_nope @ wkv_b1_weight
    "wkv_b1": {
        640: [
            {
                "num_heads": 64,
                "q_lora_rank": 1536,
                "chunked_only": True,
                "program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=2,
                    out_subblock_h=2,
                    out_subblock_w=4,
                    per_core_M=4,
                    per_core_N=16,
                ),
                "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
                "out_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_dtype": ttnn.bfloat16,
            },
            {
                # Batched (Z=heads/tp) matmul -- DM-bound, config-invariant per the tuning sweep
                # (in0_block_w/subblocks/act-out-mem all measured ~50us regardless).
                **_GLM_TAGS,
                "program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=6,
                    out_subblock_h=2,
                    out_subblock_w=4,
                    per_core_M=4,
                    per_core_N=16,
                ),
                "act_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_dtype": ttnn.bfloat16,
            },
        ],
        4096: {
            "program_config": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=8,
                per_core_M=2,
                per_core_N=16,
                fuse_batch=False,
                mcast_in0=False,
            ),
            "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        },
        3200: {
            "program_config": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=8,
                per_core_M=1,
                per_core_N=16,
                fuse_batch=False,
                mcast_in0=False,
            ),
            "act_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        },
    },
    # hidden_states @ kv_a_proj_with_mqa_weight
    "kv_a_proj_with_mqa": {
        640: [
            {
                "num_heads": 64,
                "q_lora_rank": 1536,
                "chunked_only": True,
                "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=14,
                    out_subblock_h=2,
                    out_subblock_w=1,
                    per_core_M=2,
                    per_core_N=2,
                    transpose_mcast=False,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                "act_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_dtype": ttnn.bfloat16,
            },
            {
                **_GLM_TAGS,
                "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=2,
                    per_core_M=2,
                    per_core_N=2,
                    transpose_mcast=False,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                "act_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_dtype": ttnn.bfloat16,
            },
        ],
        4096: {
            "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=13,
                per_core_N=2,
                transpose_mcast=False,
                fuse_batch=False,
                fused_activation=None,
            ),
            "act_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        },
        3200: {
            "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=8,
                out_subblock_h=2,
                out_subblock_w=2,
                per_core_M=10,
                per_core_N=2,
                transpose_mcast=False,
                fuse_batch=False,
                fused_activation=None,
            ),
            "act_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        },
    },
    # tt_v_latent_post_repeat @ wkv_b2_weight
    "wkv_b2": {
        640: [
            {
                "num_heads": 64,
                "q_lora_rank": 1536,
                "chunked_only": True,
                "program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=2,
                    out_subblock_h=4,
                    out_subblock_w=1,
                    per_core_M=4,
                    per_core_N=4,
                ),
                "act_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_dtype": ttnn.bfloat8_b,
            },
            {
                # Batched (Z=heads/tp) matmul -- DM-bound, config-invariant per the tuning sweep.
                **_GLM_TAGS,
                "program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=2,
                    out_subblock_h=4,
                    out_subblock_w=2,
                    per_core_M=4,
                    per_core_N=8,
                ),
                "act_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_dtype": ttnn.bfloat8_b,
            },
        ],
        4096: {
            "program_config": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=8,
                out_subblock_h=2,
                out_subblock_w=4,
                per_core_M=2,
                per_core_N=4,
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=False,
            ),
            "act_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat8_b,
        },
        3200: {
            "program_config": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=16,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=1,
                per_core_N=4,
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=False,
            ),
            "act_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat8_b,
        },
    },
    # v_out @ o_proj_weight
    "o_proj": {
        640: [
            {
                "num_heads": 64,
                "q_lora_rank": 1536,
                "chunked_only": True,
                "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=7,
                    per_core_M=2,
                    per_core_N=21,
                    transpose_mcast=False,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
                "out_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_dtype": ttnn.bfloat16,
            },
            {
                **_GLM_TAGS,
                "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=16,
                    out_subblock_h=1,
                    out_subblock_w=6,
                    per_core_M=2,
                    per_core_N=18,
                    transpose_mcast=False,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
                "out_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_dtype": ttnn.bfloat16,
            },
        ],
        4096: {
            "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=7,
                per_core_M=13,
                per_core_N=21,
                transpose_mcast=False,
                fuse_batch=False,
                fused_activation=None,
            ),
            "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        },
        3200: {
            "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=7,
                per_core_M=10,
                per_core_N=21,
                transpose_mcast=False,
                fuse_batch=False,
                fused_activation=None,
            ),
            "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        },
    },
    # GLM-5.1/5.2 DSA indexer linears (indexer.py). No dense/Kimi analogue -- only sparse models call
    # these. DeepSeek v3.2 is also sparse (num_heads=128, q_lora_rank=1536) so these still need the
    # num_heads/q_lora_rank gate to avoid misapplying a GLM-shaped config to it. Not tagged
    # chunked_only: the indexer's write_k/forward are always block-cyclic (single-shot is folded onto
    # the same shape as one full-seq chunk), so there is no separate single-shot shape to exclude.
    # qr @ indexer.wq_b_weight -- identical per-chip shape to q_b_proj (see indexer.py forward()).
    "indexer.wq_b": {
        640: {
            **_GLM_INDEXER_TAGS,
            "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=6,
                per_core_M=2,
                per_core_N=12,
                transpose_mcast=False,
                fuse_batch=False,
                fused_activation=None,
            ),
            "act_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        },
    },
    # hidden_states @ indexer.wk_weight
    "indexer.wk": {
        640: {
            **_GLM_INDEXER_TAGS,
            "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=2,
                per_core_N=1,
                transpose_mcast=False,
                fuse_batch=False,
                fused_activation=None,
            ),
            "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        },
    },
    # hidden_states @ indexer.weights_proj_weight -- N_t=1 core-floors at 10 cores; re-tuned for its
    # #51005 BF16 weight (was BF8 when this config was first tuned): moving act to L1 was the win.
    "indexer.weights_proj": {
        640: {
            **_GLM_INDEXER_TAGS,
            "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=24,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=2,
                per_core_N=1,
                transpose_mcast=False,
                fuse_batch=False,
                fused_activation=None,
            ),
            "act_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        },
    },
}


MLA_SDPA_CONFIG = {
    # Tuned for the Galaxy balanced MLA PCC path. The 8x4 max-sl cases hit the issue #45521 fallback.
    # 128k total seq_len → 16384 per device on 8x4
    16384: {
        "q_chunk_size": 128,
        "k_chunk_size": 320,
    },
    # 100k total seq_len → 12800 per device on 8x4
    12800: {
        "q_chunk_size": 160,
        "k_chunk_size": 320,
    },
    # 128k total seq_len → 4096 per device on 32x4, or scaled 2x4
    4096: {
        "q_chunk_size": 128,
        "k_chunk_size": 320,
    },
    # 100k total seq_len → 3200 per device on 32x4, or scaled 2x4
    3200: {
        "q_chunk_size": 160,
        "k_chunk_size": 320,
    },
    # 5k total seq_len → 640 per device on 8x4
    640: {
        "q_chunk_size": 32,
        "k_chunk_size": 640,
        "num_heads": None,
        "dense_head_cap_non_dsa": 64,
        "chunked_only": True,
    },
}


def get_matmul_config(weight_name: str, seq_len_local: int) -> dict | None:
    """Get optimal matmul config for a given weight and local sequence length (per-device).

    Returns None if no config is found for the given weight_name/seq_len_local combination.
    """
    return MLA_MATMUL_CONFIG.get(weight_name, {}).get(seq_len_local)


def resolve_gated_matmul_config(
    weight_name: str,
    seq_len_local: int,
    *,
    num_heads: int | None = None,
    q_lora_rank: int | None = None,
    is_chunked: bool | None = None,
) -> dict | None:
    """Resolve a matmul config entry, disambiguating a seq_len slot shared by multiple model
    variants' candidates (a list) via the num_heads/q_lora_rank/chunked_only tags declared on each
    candidate. A slot holding a single untagged dict (no variant conflict) is returned as-is.
    Returns None when nothing matches (caller falls back to defaults). Shared by
    ttMLA._resolve_mm_cfg and TtIndexer, since both consume the same MLA_MATMUL_CONFIG table.
    """
    entry = get_matmul_config(weight_name, seq_len_local)
    if entry is None:
        return None
    candidates = entry if isinstance(entry, list) else [entry]
    for cfg in candidates:
        if cfg.get("num_heads") not in (None, num_heads):
            continue
        if cfg.get("q_lora_rank") not in (None, q_lora_rank):
            continue
        if cfg.get("chunked_only") and not is_chunked:
            continue
        return cfg
    return None


def get_sdpa_config(seq_len_local: int) -> dict | None:
    """Get optimal SDPA chunk sizes for a given local sequence length (per-device).

    Returns None if no config is found for the given seq_len_local.
    """
    return MLA_SDPA_CONFIG.get(seq_len_local)


# DSA lightning-indexer scoring config, keyed by resident index-head count (index_n_heads). The
# indexer runs indexer_score with head_group_size=0, so ALL index heads stay on-chip and the key
# chunk is L1-bound, scaling ~1/heads. Values are the measured per-model optima (k_chunk sweep on
# LoudBox / Blackhole at Sq=640, T=56320): a larger k_chunk OOMs L1 (DeepSeek@64h fits <=96,
# GLM@32h fits <=256). DeepSeek is flat so 64 is optimal and L1-safe; GLM is ~8% faster at 224.
DSA_INDEXER_CONFIG: dict[int, dict[str, int]] = {
    64: {"k_chunk_size": 64},  # DeepSeek V3.2
    32: {"k_chunk_size": 224},  # GLM 5.1 / 5.2
}


def get_indexer_key_chunk(index_n_heads: int) -> int:
    """Indexer_score k_chunk_size for a resident index-head count. Raises on an unmapped head count:
    k_chunk is L1-bound and a too-large value OOMs, so a new model must be swept (largest L1-safe
    k_chunk) and added to DSA_INDEXER_CONFIG rather than silently defaulted."""
    cfg = DSA_INDEXER_CONFIG.get(index_n_heads)
    if cfg is None:
        raise KeyError(
            f"No DSA indexer k_chunk_size tuned for index_n_heads={index_n_heads}; sweep the largest "
            f"L1-safe k_chunk and add it to DSA_INDEXER_CONFIG (tuned: {sorted(DSA_INDEXER_CONFIG)})."
        )
    return cfg["k_chunk_size"]
