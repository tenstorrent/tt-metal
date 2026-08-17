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
            # Kimi-K3 (96 heads): per-device shape is identical (K = hidden/tp = 1792 either way),
            # so the K2.6 tiling above transfers unchanged.
            {
                "num_heads": 96,
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
            # Kimi-K3 (96 heads): N widens 3072 -> 4608 per device (N_t 96 -> 144), so per_core_N
            # must cover 144 over 11 columns -> 14. K_t = 48, in0_block_w=8 divides it;
            # out_subblock_w=7 divides 14.
            {
                "num_heads": 96,
                "q_lora_rank": 1536,
                "chunked_only": True,
                "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=7,
                    per_core_M=2,
                    per_core_N=14,
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
            # Kimi-K3 (96 heads): batch = H_loc goes 16 -> 24. MatmulMultiCoreReuse spreads
            # batch * (M_t/per_core_M) * (N_t/per_core_N) blocks over cores, so K2.6's per_core_M=4
            # would ask for 24 * (20/4) = 120 blocks on a 110-core grid; per_core_M=5 gives 96.
            {
                "num_heads": 96,
                "q_lora_rank": 1536,
                "chunked_only": True,
                "program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=2,
                    out_subblock_h=1,
                    out_subblock_w=8,
                    per_core_M=5,
                    per_core_N=16,
                ),
                "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
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
            # Kimi-K3 (96 heads). in0_block_w PINNED AT 1 FOR ACCURACY, not perf: this is the only
            # tuned matmul on the KV-cache path, so its rounding compounds with KV depth instead of
            # staying local to one chunk. Every faster divisor of K_t=56 fails the 0.98 chunked PCC at
            # depth -- including in0_block_w=2, which has the BEST single-chunk PCC of any value. The
            # per-op PCC ranking is INVERTED against depth behaviour here, so no op-level measurement
            # can justify raising it. Ladder and guard: test_kimi_k3_mla_reference.py::
            # test_k3_accuracy_pinned_blocking. ibw=1 also matches what the untuned default picks, so
            # only out_mem_config is reclaimed here (placement cannot change numerics).
            # K2.6 still runs ibw=14 and shows the same degraded cache PCC; it passes, but thinner
            # than it needs. Giving it this entry would likely buy ~0.0005 at depth.
            {
                "num_heads": 96,
                "q_lora_rank": 1536,
                "chunked_only": True,
                "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=1,
                    out_subblock_h=2,
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
            # Kimi-K3 (96 heads): same batch increase as wkv_b1.
            {
                "num_heads": 96,
                "q_lora_rank": 1536,
                "chunked_only": True,
                "program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=2,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    per_core_M=5,
                    per_core_N=4,
                ),
                "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
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
            # Kimi-K3 (96 heads): K widens 2048 -> 3072 but N is the full 7168 either way, and
            # in0_block_w=8 divides K_t 64 and 96 alike, so the K2.6 tiling transfers unchanged.
            {
                "num_heads": 96,
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
    # Kimi-K3 output gate: all-gathered hidden @ g_proj_weight, sigmoid fused. K = 7168 (K_t 224),
    # N = 12288/tp = 3072 (N_t 96 -> per_core_N=9 over 11 columns). ttMLA keys off this fused_activation
    # (_gate_sigmoid_fused) to decide whether to apply a standalone sigmoid, so removing it is safe but
    # adding a second sigmoid elsewhere would double-apply. SIGMOID needs both params: VecMode::RC (=4)
    # and approx (0 = accurate). L1 out: identical tile count and core fill to o_proj, which measured
    # 118.0 us against g_proj's 136.6 with DRAM out -- placement only, so free accuracy-wise.
    "g_proj": {
        640: {
            "num_heads": 96,
            "q_lora_rank": 1536,
            "chunked_only": True,
            "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=8,
                out_subblock_h=2,
                out_subblock_w=3,
                per_core_M=2,
                per_core_N=9,
                transpose_mcast=False,
                fuse_batch=False,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID, 4.0, 0.0),
            ),
            "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
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
    # 5k total seq_len → 640 per device on 8x4.
    # Two candidates (first tag match wins, see ttMLA._select_cfg):
    #   1. the original catch-all, kept exactly as-is, with its empirical non-DSA head cap;
    #   2. Kimi-K3 at 96 heads, which the cap would otherwise send to the k=32 default.
    # K3 carries NO cap because it was measured, not assumed: test_ring_mla_chunked_accuracy
    # [kimi_k3-q32-k{32,128,256,512,640}] all pass at 24 heads/device with zero L1 OOM, and the
    # final-chunk PCC goes 0.99590 (k=32) → 0.99919 (128) → 0.99936 (256) → 0.99937 (512) →
    # 0.99938 (640). So the fallback to k=32 was costing ~0.0035 PCC and protecting nothing here.
    # (The cap's stated "L1 scales with head count" rationale does not hold — see
    # ttMLA._get_sdpa_program_config.) Accuracy saturates by k=256; 640 matches K2.6's validated
    # tiling.
    #
    # Also confirmed through the model path at 8 SP, which is the stronger check: the chunked-prefill
    # test resolves this K3 candidate (the catch-all is rejected by its cap at 96 heads) and runs
    # k_chunk=640 at 24 heads/device over 56320 tokens with no L1 OOM. Note the sweep test itself
    # cannot corroborate that on a wider mesh -- it forces FABRIC_1D_RING, which does not map beyond
    # 2 SP for any variant (K2.6's case fails identically), so the sweep is small-mesh-only.
    640: [
        {
            "q_chunk_size": 32,
            "k_chunk_size": 640,
            "num_heads": None,
            "dense_head_cap_non_dsa": 64,
            "chunked_only": True,
        },
        {
            "q_chunk_size": 32,
            "k_chunk_size": 640,
            "num_heads": 96,
            "chunked_only": True,
        },
    ],
}


def get_matmul_config(weight_name: str, seq_len_local: int) -> dict | list | None:
    """Raw matmul entry for a given weight and local sequence length (per-device).

    Returns None if there is no entry. **A slot may hold a LIST of candidates** (one per model
    flavour sharing this seq_len — e.g. Kimi-K2.6 and Kimi-K3 both at 640), and this accessor does
    NOT apply the gating tags. ``ttMLA`` deliberately reads the dicts directly and resolves through
    ``_select_cfg`` / ``_cfg_matches``, which is the only place that knows the live model's head
    count, q_lora_rank and chunked mode. Any new caller should do the same rather than assume the
    return value is a single usable config.
    """
    return MLA_MATMUL_CONFIG.get(weight_name, {}).get(seq_len_local)


def get_sdpa_config(seq_len_local: int) -> dict | list | None:
    """Raw SDPA entry for a given local sequence length (per-device).

    Returns None if there is no entry. May be a list of candidates and does not apply the gating
    tags — see ``get_matmul_config``.
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
