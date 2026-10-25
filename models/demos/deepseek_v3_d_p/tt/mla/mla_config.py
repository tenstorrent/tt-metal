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
# Mistral-Small-4 at 8x4. tp_factor keeps these off its PP=4 [8,1] stages, which hit the same 640-row
# slot at TP=1 where these tilings do not fit. fp32_dest_acc (set per entry, not a tag) selects ttMLA's
# HiFi2 + fp32-dest compute config for that matmul.
_MISTRAL4_TAGS = {"num_heads": 32, "q_lora_rank": 1024, "tp_factor": 4, "chunked_only": True}

MLA_MATMUL_CONFIG = {
    # hidden_states @ [q_a_proj | kv_a_proj_with_mqa | pad] (ttMLA._merge_qkv_a). GLM-5.3: [640,1536] x
    # [1536,2688] per device, N = 84 tiles over 11 columns (8 per core), M = 20 tiles over 10 rows. L1 out:
    # the TP reduce-scatter reads it directly (same as the q_a_proj entry it replaces).
    "qkv_a_proj": {
        640: [
            {
                **_GLM_TAGS,
                "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=8,
                    per_core_M=2,
                    per_core_N=8,
                    transpose_mcast=False,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
                "out_mem_config": ttnn.L1_MEMORY_CONFIG,
                "out_dtype": ttnn.bfloat16,
            },
        ],
    },
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
            # Mistral-Small-4 (32 heads, q_lora_rank 1024, TP=4): K = N = 1024 per device (Mt 20, Kt 32,
            # Nt 32). Traced single-chip, HiFi2 both sides: 16.8 us vs 33.4 default, rel. err 0.44% vs 0.86%.
            {
                **_MISTRAL4_TAGS,
                "fp32_dest_acc": True,
                "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=3,
                    per_core_M=2,
                    per_core_N=3,
                    transpose_mcast=False,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
                "out_mem_config": ttnn.DRAM_MEMORY_CONFIG,
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
        # GLM-5.3 sequence-sharded Q (ttMLA._q_stem_seq): this chip's 160 TP-quarter rows x all 64 heads against the
        # TP-replicated weight, [160, 2048] x [2048, 16384]. M = 5 tiles: multicast the whole M slab to N-parallel
        # cores (the indexer's wq_b shape, 4x wider); weight-read bound (35.6 MB bfp8).
        160: {
            **_GLM_TAGS,
            "program_config": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=COMPUTE_GRID,
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=5,
                per_core_M=5,
                per_core_N=5,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            ),
            "act_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_mem_config": ttnn.L1_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat16,
        },
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
            # Mistral-Small-4: same [640,1024]x[1024,1024] per-device shape as its q_a_proj. L1 both sides:
            # in0 is the q_a_layernorm output (act_mem_config), out feeds nlp_create_q_heads_split, which
            # pins DRAM itself. 12.1 us L1->L1 vs 16.8 DRAM->DRAM.
            {
                **_MISTRAL4_TAGS,
                "fp32_dest_acc": True,
                "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=3,
                    per_core_M=2,
                    per_core_N=3,
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
        # GLM-5.3 sequence-sharded Q: 64 heads x [160, 192] x [192, 512]. Same per-core block as the 640 entry
        # (64 blocks of 5 x 16 tiles: heads x M-blocks 64 x 1 instead of 16 x 4).
        160: {
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
            # Mistral-Small-4: batch 8 heads, Mt 20, Kt 2, Nt 8 (kv_lora_rank 256) -> 80 blocks. The 1D
            # batched fallback leaves heads unsplit on 20 cores: 16.8 us vs 41.3, bit-identical error.
            # DRAM out: the concat after it inherits in0's placement and feeds ring_mla, which needs DRAM.
            {
                **_MISTRAL4_TAGS,
                "program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=2,
                    out_subblock_h=2,
                    out_subblock_w=4,
                    per_core_M=2,
                    per_core_N=8,
                ),
                "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
                "out_mem_config": ttnn.DRAM_MEMORY_CONFIG,
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
            # Mistral-Small-4: [640,1024]x[1024,320] per device, one N tile per core on a 10x10 grid.
            # 12.5 us vs 29.3 default, rel. err 0.44% vs 0.86%. Output is forced to DRAM by _kv_stem.
            {
                **_MISTRAL4_TAGS,
                "fp32_dest_acc": True,
                "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 10),
                    in0_block_w=8,
                    out_subblock_h=2,
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
        # GLM-5.3 sequence-sharded Q: 64 heads x [160, 512] x [512, 256], same 5 x 8-tile block as the 640 entry.
        # Output feeds the sequence -> heads all-to-all (ttMLA._seq_to_heads).
        160: {
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
            "out_mem_config": ttnn.DRAM_MEMORY_CONFIG,
            "out_dtype": ttnn.bfloat8_b,
        },
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
            # Mistral-Small-4: batch 8, Mt 20, Kt 8, Nt 4 -> 80 blocks (fallback: 20 cores).
            # 17.7 us vs 38.9, rel. err 0.81% vs 1.67% (bfp8 out).
            {
                **_MISTRAL4_TAGS,
                "fp32_dest_acc": True,
                "program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    per_core_M=2,
                    per_core_N=4,
                ),
                "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
                "out_mem_config": ttnn.DRAM_MEMORY_CONFIG,
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
            # Mistral-Small-4: [640,1024]x[1024,4096] per device, bfp8 in0. rel. err 0.17% vs 0.83% default;
            # 31.5 us to L1 vs 44.0 to DRAM vs 58.5 default. L1 out as for Kimi: the TP reduce-scatter reads it
            # and writes its own DRAM output.
            {
                **_MISTRAL4_TAGS,
                "fp32_dest_acc": True,
                "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=COMPUTE_GRID,
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    per_core_M=2,
                    per_core_N=12,
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


# ---- small native chunks (smallchunk agent, 2026-10-25) ----------------------------------------------------------
# Native GLM chunks of 1024/2048 tokens give 128/256 rows per chip (32/64 TP-sequence rows for seqq) where the table
# has no entry, so every MLA/indexer matmul falls back to the TTNN auto config with a DRAM output (auto is ~2x slower
# than tuned on these shapes, e.g. indexer wq_b 60 vs 31 us), and the paths keyed on a tuned config (in-place q RoPE,
# L1 activations) turn off. GLM_SMALL_MM_CFG=1 derives the missing entry from the smallest larger tuned key by
# scaling only the M blocking (per_core_M / out_subblock_h); K/N blocking, grids, memory configs and dtypes are kept.
# GLM_SMALL_MM_CFG (DEFAULT ON = "1"): every weight; "wkv": only the batched per-head wkv_b1 / wkv_b2 (their auto
# fallback, ttMLA._make_batched_mm_kwargs, puts all 64 heads of a 1-tile-row M on ONE core: 425 / 500 us per layer at
# chunk 1024); "0": off. Measured L78 @97k (smallchunk agent): native 1024 254.6 -> 184.2 ms, 2048 314.7 -> 221.5 ms.
# Only ever applied when the caller passes the per-chip rows of a chunk SMALLER than the tuned one
# (chunk_rows < SMALL_CHUNK_TUNED_ROWS = 640 rows = chunk 5120 on 8 SP chips): the 5120 path is never touched, even for
# a call whose row count happens to have no entry. Callers pass chunk_rows only for GLM-5.3 (64 heads, q_lora 2048).
SMALL_CHUNK_TUNED_ROWS = 640


def small_mm_cfg_enabled(weight_name: str | None = None) -> bool:
    import os

    mode = os.environ.get("GLM_SMALL_MM_CFG", "1")
    if mode in ("", "0"):
        return False
    if mode == "wkv":
        return weight_name in ("wkv_b1", "wkv_b2")
    return True


def _divisor_at_least(n: int, lo: int) -> int:
    d = max(1, lo)
    while n % d:
        d += 1
    return d


def _largest_divisor_at_most(n: int, hi: int) -> int:
    d = max(1, min(n, hi))
    while n % d:
        d -= 1
    return d


def derive_small_rows_cfg(cfg: dict, seq_len_local: int) -> dict | None:
    """A copy of tuned entry ``cfg`` re-blocked for ``seq_len_local`` (< the entry's key) rows, or None if its program
    config type is not handled (caller then keeps the auto fallback)."""
    if seq_len_local % ttnn.TILE_SIZE:
        return None
    m_t = seq_len_local // ttnn.TILE_SIZE
    pc = cfg.get("program_config")
    if isinstance(pc, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig):
        grid = pc.compute_with_storage_grid_size
        m_cores = grid.x if pc.transpose_mcast else grid.y
        pm = _divisor_at_least(m_t, -(-m_t // m_cores))
        new = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=pc.in0_block_w,
            out_subblock_h=_largest_divisor_at_most(pm, max(1, 8 // pc.out_subblock_w)),
            out_subblock_w=pc.out_subblock_w,
            per_core_M=pm,
            per_core_N=pc.per_core_N,
            transpose_mcast=pc.transpose_mcast,
            fuse_batch=pc.fuse_batch,
            fused_activation=pc.fused_activation,
        )
    elif isinstance(pc, ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig):
        if not pc.mcast_in0:
            return None
        new = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y),
            in0_block_w=pc.in0_block_w,
            out_subblock_h=_largest_divisor_at_most(m_t, max(1, 8 // pc.out_subblock_w)),
            out_subblock_w=pc.out_subblock_w,
            per_core_M=m_t,
            per_core_N=pc.per_core_N,
            fuse_batch=pc.fuse_batch,
            fused_activation=pc.fused_activation,
            mcast_in0=True,
        )
    elif isinstance(pc, ttnn.MatmulMultiCoreReuseProgramConfig):
        pm = _largest_divisor_at_most(m_t, pc.per_core_M)
        new = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y),
            in0_block_w=pc.in0_block_w,
            out_subblock_h=_largest_divisor_at_most(pm, max(1, 8 // pc.out_subblock_w)),
            out_subblock_w=pc.out_subblock_w,
            per_core_M=pm,
            per_core_N=pc.per_core_N,
        )
    else:
        return None
    return {**cfg, "program_config": new}


def small_rows_entry(table: dict, seq_len_local: int, weight_name: str | None = None, chunk_rows: int | None = None):
    """Derived candidates for a missing ``seq_len_local`` slot: the smallest larger tuned key's candidates, each
    re-blocked (see derive_small_rows_cfg). None when disabled, present, nothing to derive from, or the chunk is not
    smaller than the tuned one (chunk_rows None or >= SMALL_CHUNK_TUNED_ROWS)."""
    if chunk_rows is None or chunk_rows >= SMALL_CHUNK_TUNED_ROWS:
        return None
    if not small_mm_cfg_enabled(weight_name) or seq_len_local in table or seq_len_local % ttnn.TILE_SIZE:
        return None
    larger = sorted(k for k in table if k > seq_len_local)
    if not larger:
        return None
    entry = table[larger[0]]
    cands = entry if isinstance(entry, (list, tuple)) else [entry]
    out = [d for d in (derive_small_rows_cfg(c, seq_len_local) for c in cands) if d is not None]
    return out or None


def get_matmul_config(weight_name: str, seq_len_local: int, chunk_rows: int | None = None) -> dict | list | None:
    """Raw matmul entry for a given weight and local sequence length (per-device).

    Returns None if there is no entry. **A slot may hold a LIST of candidates** (one per model
    flavour sharing this seq_len — e.g. Kimi-K2.7 and Kimi-K3 both at 640), and this accessor does
    NOT apply the gating tags. ``ttMLA`` deliberately reads the dicts directly and resolves through
    ``_select_cfg`` / ``_cfg_matches``, which is the only place that knows the live model's head
    count, q_lora_rank and chunked mode. Any new caller should do the same rather than assume the
    return value is a single usable config.
    """
    table = MLA_MATMUL_CONFIG.get(weight_name, {})
    entry = table.get(seq_len_local)
    return entry if entry is not None else small_rows_entry(table, seq_len_local, weight_name, chunk_rows)


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
