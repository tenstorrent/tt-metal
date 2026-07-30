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
        640: {
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
        640: {
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
        640: {
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
        640: {
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
        640: {
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
        640: {
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
}


# ---------------------------------------------------------------------------------------------------
# Kimi-K3 chunked-prefill (seq_len_local = 640) matmul configs.
#
# K3 is 96 heads (K2.6 is 64), so every K2.6 `640` entry above is rejected for it by the `num_heads`
# tag. A slot can hold a LIST of candidates (see ttMLA._select_cfg) and the first tag match wins, so
# each `640` entry below becomes [K2.6 candidate, K3 candidate]. Order is priority order; both are
# fully tagged and mutually exclusive, so it does not matter here.
#
# Verified per-op at the exact per-device shapes (S_loc=640, tp=4, H_loc=24) by
# tests/op_unit_tests/test_kimi_k3_mla_matmuls.py -- all PCC 0.9999 on a 2x4 Blackhole box.
# See models/demos/deepseek_v3_d_p/docs/KIMI_K3_MLA.md §3 for the shape audit.
_K3_HEADS = 96


def _retag_for_k3(cfg: dict) -> dict:
    """Reuse a K2.6 program config for K3 verbatim, changing only the head-count tag.

    Valid only where the per-device shape is unchanged. Deriving instead of copy-pasting keeps the
    two candidates from drifting apart, and makes "this config transfers unchanged" explicit.
    """
    return {**cfg, "num_heads": _K3_HEADS}


# q_a_proj: per-device shape is identical (K = hidden/tp = 1792 either way).
# o_proj: K widens 2048 -> 3072, but N is the full 7168 for both (K-sharded via mapper_tp0) and
# in0_block_w=8 divides K_t=64 and K_t=96 alike -- so the tiling is valid unchanged.
#
# kv_a_proj_with_mqa is NOT retagged: K2.6's tiling (in0_block_w=14) is dimensionally fine for K3 but
# measurably less accurate, and this is the ONE tuned matmul on the KV-cache path
# (kv_a_proj -> slice -> rms_norm -> kvpe.pack). Every later chunk re-reads that cache, so the loss
# compounds with KV depth instead of staying local to one chunk. Measured on 2x4 at S_loc=640 over 44
# chunks to 56320 tokens (test_mla_chunked_prefill[k3-depth56k-1u]):
#
#   in0_block_w=14 : KV cache k_nope 0.999810 -> output PCC FAILS 0.98 at kv_actual=3840
#   in0_block_w=1  : KV cache k_nope 0.999877 -> passes all 44, 0.98550 at kv_actual=55040
#
# A bisect isolated it: dropping this one candidate and keeping the other six (plus the k_chunk=640
# SDPA entry) passes; keeping only this one and dropping the other six still fails.
#
# K3 instead gets its own entry below that keeps the ACCURATE blocking (in0_block_w=1, the tiling the
# untuned default picks) and only reclaims the output placement, which cannot change numerics -- same
# dtype and same values, just a different buffer. Falling back to the default had silently also given
# up K2.6's `out_mem_config: L1`, which cost 46.1 us vs K2.6's 17.6 us for the same shape.
#
# NOTE for the K2.6 side: K2.6 still runs in0_block_w=14 and shows the same degraded KV-cache PCC
# (0.999810), asymptoting to 0.98589 against a 0.98 threshold. It passes, but the margin is thinner
# than it needs to be; giving it this same in0_block_w=1 entry would likely buy back ~0.0005 at depth.
for _name in ("q_a_proj", "o_proj"):
    _k26 = MLA_MATMUL_CONFIG[_name][640]
    MLA_MATMUL_CONFIG[_name][640] = [_k26, _retag_for_k3(_k26)]

MLA_MATMUL_CONFIG["kv_a_proj_with_mqa"][640] = [
    MLA_MATMUL_CONFIG["kv_a_proj_with_mqa"][640],
    {
        "num_heads": _K3_HEADS,
        "q_lora_rank": 1536,
        "chunked_only": True,
        # in0_block_w=1 / sub2x2 / pc2x2 replicates exactly what the untuned matmul picks (confirmed
        # from the tracy ATTRIBUTES of an untuned run: ibw1 sub2x2 pc2x2, 90 cores), so the arithmetic
        # -- and therefore the KV-cache PCC -- is unchanged from the accurate default.
        #
        # in0_block_w=1 IS THE ONLY VALUE THAT REACHES PRODUCTION DEPTH. Swept every divisor of
        # K_t=56 and depth-tested each (test_mla_chunked_prefill[k3-depth56k-1u], 44 chunks to 56320):
        #
        #   ibw   us (isolated)   single-chunk PCC   depth: fails at kv_actual
        #     1        44.7          0.9999216       never -- all 44, 0.98550 @ 55040
        #     2        28.7          0.9999339           20480
        #     4        21.6          0.9999294            7680
        #     8        16.5          0.9999034            5120
        #    14        17.9          0.9998523            3840
        #
        # So ~28 us/chunk is the irreducible price of depth correctness here, not a tuning oversight.
        # NOTE the trap: ibw=2 has the BEST single-chunk PCC of any value -- better than ibw=1, the one
        # that works -- and still dies at 20480. For a matmul feeding the KV cache the per-op PCC
        # ranking is INVERTED against depth behaviour, because its error is written to the cache and
        # re-read by every later chunk. Never justify raising this from an op-level PCC; only
        # depth56k-1u decides. Guarded by test_k3_accuracy_pinned_blocking.
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
        # act_mem_config is inert for this weight (_get_act_mem_config is only consulted for q_b_proj);
        # out_mem_config is the one that matters here.
        "act_mem_config": ttnn.L1_MEMORY_CONFIG,
        "out_mem_config": ttnn.L1_MEMORY_CONFIG,
        "out_dtype": ttnn.bfloat16,
    },
]

# q_b_proj: N widens 3072 -> 4608 per device (N_t 96 -> 144), so per_core_N must cover 144 over 11
# columns: 14 (154 >= 144). K_t = 48, in0_block_w = 8 divides it; out_subblock_w = 7 divides 14.
MLA_MATMUL_CONFIG["q_b_proj"][640] = [
    MLA_MATMUL_CONFIG["q_b_proj"][640],
    {
        "num_heads": _K3_HEADS,
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
]

# wkv_b1 / wkv_b2: batched per-head matmuls whose batch IS the local head count, 16 -> 24. These are
# the only entries the head increase genuinely breaks: MatmulMultiCoreReuse distributes
# batch * (M_t/per_core_M) * (N_t/per_core_N) blocks over cores, so K2.6's per_core_M=4 asks for
# 24 * (20/4) * 1 = 120 blocks on a 110-core grid. per_core_M=5 gives 24 * 4 = 96.
MLA_MATMUL_CONFIG["wkv_b1"][640] = [
    MLA_MATMUL_CONFIG["wkv_b1"][640],
    {
        "num_heads": _K3_HEADS,
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
]
MLA_MATMUL_CONFIG["wkv_b2"][640] = [
    MLA_MATMUL_CONFIG["wkv_b2"][640],
    {
        "num_heads": _K3_HEADS,
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
]

# g_proj (Kimi-K3 output gate) -- new op, no K2.6 counterpart. ttMLA all-gathers hidden_states to the
# full hidden size and N-shards this weight (mapper_tp1), so K = 7168 (K_t = 224) and N = 12288/tp =
# 3072 (N_t = 96 -> per_core_N = 9 covers it on 11 columns). Because g is complete per device, sigmoid
# fuses here instead of costing a separate eltwise pass; ttMLA keys off this fused_activation
# (_gate_sigmoid_fused) to decide whether to apply a standalone ttnn.sigmoid, so REMOVING it silently
# changes nothing but ADDING a second sigmoid elsewhere would double-apply.
# SIGMOID needs both params: VecMode::RC (=4) and approx (0 = accurate).
MLA_MATMUL_CONFIG["g_proj"] = {
    640: {
        "num_heads": _K3_HEADS,
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
        # L1 out: g_proj and o_proj have IDENTICAL tile counts (430,080) and both fill 110 cores, yet
        # measured 136.6 vs 118.0 us -- the only config difference was this field (g_proj DRAM, o_proj
        # L1). Placement cannot change numerics, so this is free accuracy-wise.
        "out_mem_config": ttnn.L1_MEMORY_CONFIG,
        "out_dtype": ttnn.bfloat16,
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
    # ttMLA._get_sdpa_program_config and docs/KIMI_K3_MLA.md §0.5.) Accuracy saturates by k=256;
    # 640 matches K2.6's validated tiling.
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
