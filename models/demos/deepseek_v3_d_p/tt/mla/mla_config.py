# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Optimal matmul and SDPA configurations for the MLA module, keyed by local sequence length
(per-device after SP sharding). SDPA configs sourced from op_unit_tests/test_ring_joint_mla.py.

utils/test_mla_matmuls.py is a hand-run harness for local development and perf measurement of
these matmuls; its program configs are copied from the 4096 entries below with per_core_M adjusted
for the 5k chunk, so retuning either side means updating the other.

Production local seq_len values:
  - 128k total / 8 SP devices = 16384 per device
  - 100k total / 8 SP devices = 12800 per device
  - 128k total / 32 SP devices = 4096 per device
  - 100k total / 32 SP devices = 3200 per device

A slot may hold a LIST of candidates when variants share a seq_len; ttMLA takes the first whose
gating tags match, so order is priority order, most specific first. The tags and their semantics
live with the checks in ``ttMLA._cfg_matches`` -- one of them is the per-device K, so an entry whose
tiling cannot divide a given model's K is skipped rather than applied to it.
"""

import ttnn

# Available core grid is 12x10, but due to di/dt and throttling problems, use 11x10 temporarily
COMPUTE_GRID = (11, 10)

# GLM-5.2 has the 64-head, q_lora_rank=2048 geometry. These tags keep
# its 640-token configs separate from Kimi and DeepSeek variants sharing the same slot.
_GLM_TAGS = {"num_heads": 64, "q_lora_rank": 2048, "chunked_only": True}
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
            {
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
            {
                **_GLM_TAGS,
                "program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=6,
                    out_subblock_h=1,
                    out_subblock_w=8,
                    per_core_M=5,
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
            {
                **_GLM_TAGS,
                "program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=2,
                    out_subblock_h=1,
                    out_subblock_w=8,
                    per_core_M=5,
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
            {
                **_GLM_TAGS,
                "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=8,
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
    # GLM DSA indexer: Q is TP-sequence-sharded before projection (160 rows at SP8/TP4, chunk5120).
    # The 640-row entries remain useful for larger chunks / smaller TP and the unchanged K/gate stems.
    "indexer.wq_b": {
        160: {
            **_GLM_INDEXER_TAGS,
            # M=5 tiles, N=128 tiles: multicast the whole M slab across 64 N-parallel cores.
            # QB traced sweep: ~31 us vs ~60 us auto and ~48 us for the best tested 2D config.
            "program_config": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=16,
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=5,
                per_core_N=2,
                fuse_batch=True,
                mcast_in0=True,
            ),
            "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        },
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
    "indexer.weights_proj": {
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
            # Main's TP all-reduce path feeds this directly to high_bw_all_gather, which requires DRAM.
            "out_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        },
    },
    "indexer.k_hadamard": {
        640: {
            **_GLM_INDEXER_TAGS,
            "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=4,
                out_subblock_h=2,
                out_subblock_w=1,
                per_core_M=2,
                per_core_N=1,
                transpose_mcast=False,
                fuse_batch=False,
                fused_activation=None,
            ),
        },
    },
    "indexer.q_hadamard": {
        160: {
            **_GLM_INDEXER_TAGS,
            # Fuse 32 heads × 5 query tiles over 80 cores; broadcast the common H128 transform.
            # QB traced sweep: ~9 us vs ~94 us auto for BF16 input / BFP8 output.
            "program_config": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=2,
                out_subblock_h=2,
                out_subblock_w=4,
                per_core_M=2,
                per_core_N=4,
                fuse_batch=True,
                mcast_in0=False,
            ),
        },
        640: {
            **_GLM_INDEXER_TAGS,
            "program_config": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=2,
                out_subblock_h=2,
                out_subblock_w=4,
                per_core_M=8,
                per_core_N=4,
                fuse_batch=True,
                mcast_in0=False,
            ),
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


def _bax_mc2d(in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=COMPUTE_GRID,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fuse_batch=False,
        fused_activation=None,
    )


def _bax_reuse(in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N):
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=COMPUTE_GRID,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
    )


# Batch-axis attention (ttMLA(batch_axis=...), one user per mesh column, TP=1): every chip runs the
# WHOLE GLM-5.3 projection of its user, so the per-chip shapes are the TP=4 ones with the TP-split dim
# 4x wider, which the table above cannot serve. Kept as a separate table (selected only in batch-axis
# mode) so the TP candidates stay untouched. Tuned on one Blackhole chip by
# tests/op_unit_tests/test_mla_matmuls_glm_batch_axis.py (HiFi2, warm, best of 3); the comment on each
# entry is that measurement vs the TTNN-default fallback batch-axis used before.
_GLM_BATCH_AXIS_TAGS = {"num_heads": 64, "q_lora_rank": 2048}
MLA_BATCH_AXIS_MATMUL_CONFIG = {
    # 640x6144x2048: 74.1 us, 110 cores, 71% (default 81.1 us). Act stays DRAM: it is the block's
    # attn-norm output; in L1 it measured 68.8 us.
    "q_a_proj": {
        640: {
            **_GLM_BATCH_AXIS_TAGS,
            "program_config": _bax_mc2d(16, 1, 6, 2, 6),
            "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        }
    },
    # 640x2048x16384: 236.1 us, 110 cores, 60% (default 304.8 us on 103 cores). The 21 MB output does
    # not fit L1 beside the CBs, so it stays in DRAM; act_mem_config places the q_a_layernorm output.
    "q_b_proj": {
        640: {
            **_GLM_BATCH_AXIS_TAGS,
            "program_config": _bax_mc2d(8, 1, 8, 2, 48),
            "act_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        }
    },
    # 640x6144x576: 43.1 us, 90 cores (N_t=18 floor) (default 44.5 us). DM-bound: 42.6-45.1 us across
    # every blocking; only an L1 activation moves it (32.9 us).
    "kv_a_proj_with_mqa": {
        640: {
            **_GLM_BATCH_AXIS_TAGS,
            "program_config": _bax_mc2d(16, 1, 2, 2, 2),
            "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        }
    },
    # Z=64 x 640x192x512: 202.8 us, 64 cores (default 1D mcast 810.1 us on 20 cores). Reuse needs
    # per_core_M | M_t and per_core_N == N, so 1280/20 = 64 cores is the grid ceiling; per_core_M=10
    # (110 cores, 128 blocks) measured 206.4 us and is NOT used -- the same over-subscription returned
    # garbage on wkv_b2. The 42 MB output does not fit L1.
    "wkv_b1": {
        640: {
            **_GLM_BATCH_AXIS_TAGS,
            "program_config": _bax_reuse(6, 2, 4, 20, 16),
            "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        }
    },
    # Z=64 x 640x512x256: 191.6 us, 64 cores (default 1D mcast 758.2 us on 20 cores). DM-bound.
    "wkv_b2": {
        640: {
            **_GLM_BATCH_AXIS_TAGS,
            "program_config": _bax_reuse(4, 1, 8, 20, 8),
            "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat8_b,
        }
    },
    # 640x16384x6144: 493.5 us, 110 cores, 86% (default 849.1 us on 96 cores). in0_block_w=32 overflows L1.
    "o_proj": {
        640: {
            **_GLM_BATCH_AXIS_TAGS,
            "program_config": _bax_mc2d(16, 1, 6, 2, 18),
            "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        }
    },
    # 640x2048x4096 (all 640 query rows; no TP query split): 48.7 us with qr in L1, 110 cores, 73%
    # (default 93.0 us). qr lands in L1 through q_b_proj's act_mem_config above.
    "indexer.wq_b": {
        640: {
            **_GLM_BATCH_AXIS_TAGS,
            "program_config": _bax_mc2d(8, 1, 6, 2, 12),
            "act_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        }
    },
    # 640x6144x128: 41.3 us, 40 cores (N_t=4 floor) (default 42.2 us). DM-bound on the activation
    # (27.9 us with it in L1).
    "indexer.wk": {
        640: {
            **_GLM_BATCH_AXIS_TAGS,
            "program_config": _bax_mc2d(24, 1, 1, 2, 1),
            "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        }
    },
    # 640x6144x32 (BF16 weight): 33.6 us, 10 cores (N_t=1 floor) (default 74.9 us on 20 cores). 18.5 us
    # with the activation in L1.
    "indexer.weights_proj": {
        640: {
            **_GLM_BATCH_AXIS_TAGS,
            "program_config": _bax_mc2d(8, 1, 1, 2, 1),
            "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        }
    },
}


def _bax_mc1d(in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch=True):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=COMPUTE_GRID,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=fuse_batch,
        mcast_in0=False,
    )


def _bax(program_config, act, out, dtype=ttnn.bfloat16):
    return {
        **_GLM_BATCH_AXIS_TAGS,
        "program_config": program_config,
        "act_mem_config": act,
        "out_mem_config": out,
        "out_dtype": dtype,
    }


_DRAM, _L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG

# 256 rows per chip (2048-token chunks over SP=8). Same tuning test, M_t = 8: per_core_M = 1 puts the 8 block
# rows on the 10-row grid; the batched pair's Reuse needs per_core_M | 8 with <= 110 blocks -> per_core_M = 8.
# Comment = measured vs the TTNN-default fallback at 256 rows.
_BATCH_AXIS_256 = {
    "q_a_proj": _bax(_bax_mc2d(24, 1, 6, 1, 6), _DRAM, _L1),  # 63.0 us, 88 cores (default 65.5)
    "q_b_proj": _bax(_bax_mc2d(8, 1, 8, 1, 48), _L1, _L1),  # 146.7 us (default 145.4); qr in L1 for wq_b
    "kv_a_proj_with_mqa": _bax(_bax_mc2d(32, 1, 2, 1, 2), _DRAM, _L1),  # 25.8 us, 72 cores (default 28.8)
    "wkv_b1": _bax(_bax_reuse(6, 2, 4, 8, 16), _DRAM, _L1),  # 62.6 us, 64 cores (default 1D 676.8)
    "wkv_b2": _bax(_bax_reuse(4, 1, 8, 8, 8), _DRAM, _L1, ttnn.bfloat8_b),  # 78.5 us (default 1D 732.7)
    "o_proj": _bax(_bax_mc2d(16, 1, 6, 1, 18), _DRAM, _L1),  # 423.5 us, 88 cores (default 471.0)
    "indexer.wq_b": _bax(_bax_mc2d(8, 1, 6, 1, 12), _L1, _L1),  # 42.5 us, 88 cores (default 77.1)
    "indexer.wk": _bax(_bax_mc2d(48, 1, 1, 1, 1), _DRAM, _DRAM),  # 22.7 us, 32 cores (default 23.8)
    "indexer.weights_proj": _bax(_bax_mc2d(24, 1, 1, 1, 1), _DRAM, _DRAM),  # 17.9 us, 8 cores (default 68.1)
    # Hadamards: the indexer reads only program_config (fused 32-head batch for Q).
    "indexer.q_hadamard": _bax(_bax_mc1d(4, 1, 4, 3, 4), _DRAM, _DRAM, ttnn.bfloat8_b),  # 15.8 us (default 98.3)
    "indexer.k_hadamard": _bax(_bax_mc2d(4, 1, 1, 1, 1), _DRAM, _DRAM),  # 2.1 us (default 4.0)
}
for _name, _cfg in _BATCH_AXIS_256.items():
    MLA_BATCH_AXIS_MATMUL_CONFIG.setdefault(_name, {})[256] = _cfg

# 640 rows: the two Hadamards, missing from the first batch-axis tuning pass.
MLA_BATCH_AXIS_MATMUL_CONFIG.setdefault("indexer.q_hadamard", {})[640] = _bax(
    _bax_mc1d(4, 2, 4, 6, 4), _DRAM, _DRAM, ttnn.bfloat8_b
)  # 31.6 us, 107 cores (default 106.9)
MLA_BATCH_AXIS_MATMUL_CONFIG.setdefault("indexer.k_hadamard", {})[640] = _bax(
    _bax_mc2d(4, 2, 1, 2, 1), _DRAM, _DRAM
)  # 2.9 us (default 4.7) -- the TP table's 640 entry


def get_batch_axis_matmul_config(weight_name: str, seq_len_local: int) -> dict | None:
    """Batch-axis (TP=1, one user per mesh column) matmul entry, or None. Gating tags are not applied
    here -- callers check them, as with get_matmul_config."""
    return MLA_BATCH_AXIS_MATMUL_CONFIG.get(weight_name, {}).get(seq_len_local)


def get_matmul_config(weight_name: str, seq_len_local: int) -> dict | list | None:
    """Raw matmul entry for a given weight and local sequence length (per-device).

    Returns None if there is no entry. **A slot may hold a LIST of candidates** (one per model
    flavour sharing this seq_len — e.g. Kimi-K2.7 and Kimi-K3 both at 640), and this accessor does
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
# chunk is L1-bound, scaling ~1/heads. Values are measured per-model optima on Blackhole.
# DeepSeek is flat at 64. GLM's fused ring path uses five-tile block-cyclic runs, so KC=10 avoids
# splitting a run across work units while remaining fast at both 55K and 512K prefixes. The 320 value
# is L1-validated for the fused 32-head Ring Indexer; revalidate L1 before reusing it in a classic path.
DSA_INDEXER_CONFIG: dict[int, dict[str, int]] = {
    64: {"k_chunk_size": 64},  # DeepSeek V3.2
    32: {"k_chunk_size": 320},  # GLM 5.2
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


def _merge_chunk_sweep_configs():
    """Swept 2.5k..4.5k-chunk entries (glm_chunk_matmul_configs.TUNED) for row counts the hand-tuned tables
    above do not cover. A hand-tuned entry always wins: only absent (weight, rows) slots are filled."""
    from models.demos.deepseek_v3_d_p.tt.glm_chunk_matmul_configs import (
        TUNED,
        dtype,
        mem_config,
        program_config_from_desc,
    )

    for (table, name, rows), e in TUNED.items():
        if table == "tp":
            target, tags = MLA_MATMUL_CONFIG, (_GLM_INDEXER_TAGS if name.startswith("indexer.") else _GLM_TAGS)
        elif table == "bax":
            target, tags = MLA_BATCH_AXIS_MATMUL_CONFIG, _GLM_BATCH_AXIS_TAGS
        else:
            continue
        # program_config None = TTNN's own tiling won the sweep. The entry still pins the swept output memory
        # and dtype; an absent slot would instead take ttMLA's fallback (DRAM output, and for wkv_b1 / wkv_b2
        # a 1D batched config several times slower than either).
        target.setdefault(name, {}).setdefault(
            rows,
            {
                **tags,
                "program_config": program_config_from_desc(e["desc"]),
                "act_mem_config": mem_config(e["act_mem"]),
                "out_mem_config": mem_config(e["out_mem"]),
                "out_dtype": dtype(e["out_dtype"]),
            },
        )


_merge_chunk_sweep_configs()
