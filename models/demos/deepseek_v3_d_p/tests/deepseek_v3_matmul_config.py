# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared matmul configurations for Deepseek V3 128k DIDT tests, PCC unit tests,
and the sweep tuner. Imported by tests/didt/test_deepseek_v3_128k_matmul.py,
tests/didt/sweep_deepseek_v3_matmul_tune.py, and tests/pcc/test_deepseek_v3_matmul_pcc.py.
"""

import math

import ttnn

# Per-chip grid for Deepseek V3 128k: 11×10 = 110 worker cores
GRID_SIZE = (11, 10)
TILE_SIZE = 32

# Optimal (in0_block_w, out_subblock_h, out_subblock_w) from sweep_deepseek_v3_matmul_tune.py
OPTIMAL_PROGRAM_CONFIG = {
    "dense_mlp_w1": (7, 1, 1),
    "dense_mlp_w2": (9, 1, 7),
    "dense_mlp_w3": (8, 1, 7),
    "gate": (14, 1, 1),
    "q_wq_b": (8, 1, 6),
    "routed_expert_w1": (28, 1, 3),
    "routed_expert_w2": (16, 1, 7),
    "routed_expert_w3": (28, 4, 1),
    "shared_expert_w1": (14, 1, 2),
    "shared_expert_w2": (16, 1, 1),
    "shared_expert_w3": (14, 1, 2),
    "v_out_out_proj": (8, 1, 3),
    "x_wkv_a": (14, 1, 1),
    "x_wq_a": (14, 1, 5),
}


def _find_largest_divisor(n, max_divisor=8):
    """Find the largest divisor of n that is <= max_divisor."""
    best = 1
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            if d <= max_divisor:
                best = max(best, d)
            comp = n // d
            if comp <= max_divisor:
                best = max(best, comp)
    return best


def get_prefill_matmul_program_config(M, K, N, grid_size=GRID_SIZE, optimal_config=None):
    """Compute MatmulMultiCoreReuseMultiCastProgramConfig for a prefill matmul.

    If optimal_config is (in0_block_w, out_subblock_h, out_subblock_w), use those
    values; otherwise use heuristics (largest divisor of K_tiles for in0_block_w, etc.).
    """
    grid_x, grid_y = grid_size
    M_tiles = math.ceil(M / TILE_SIZE)
    K_tiles = math.ceil(K / TILE_SIZE)
    N_tiles = math.ceil(N / TILE_SIZE)

    per_core_M = math.ceil(M_tiles / grid_y)
    per_core_N = math.ceil(N_tiles / grid_x)
    if optimal_config is not None:
        in0_block_w, out_subblock_h, out_subblock_w = optimal_config
    else:
        in0_block_w = _find_largest_divisor(K_tiles, 8)
        out_subblock_h = 1
        out_subblock_w = _find_largest_divisor(per_core_N, 4)

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


# ---------------------------------------------------------------------------
# Workload param lists: (M, K, N, ...) and workload_id for parametrization.
# MLA: (M, K, N, batch, in1_dtype, workload_id)
# Gate: single config (M, K, N, in1_dtype, workload_id)
# Dense / shared / routed: (M, K, N, workload_id)
# ---------------------------------------------------------------------------

# MLA matmuls (6 ops). batch=1 or 32 for batched heads.
MLA_MATMUL_PARAMS = [
    (4096, 1792, 1536, 1, ttnn.DataType.BFLOAT8_B, "x_wq_a"),
    (4096, 1792, 576, 1, ttnn.DataType.BFLOAT8_B, "x_wkv_a"),
    (4096, 1536, 6144, 1, ttnn.DataType.BFLOAT8_B, "q_wq_b"),
    (4096, 128, 512, 32, ttnn.DataType.BFLOAT8_B, "q_nope_wkv_b1"),
    (4096, 512, 128, 32, ttnn.DataType.BFLOAT8_B, "v_out_wkv_b2"),
    (4096, 4096, 7168, 1, ttnn.DataType.BFLOAT8_B, "v_out_out_proj"),
]

# Gate matmul: (M, K, N, in1_dtype, workload_id). Use math_fidelity=HiFi2 in caller.
GATE_MATMUL_CONFIG = (4096, 1792, 256, ttnn.DataType.BFLOAT16, "gate")

# Dense MLP: (M, K, N, workload_id)
DENSE_MLP_MATMUL_PARAMS = [
    (4096, 7168, 4608, "dense_mlp_w1"),
    (4096, 7168, 4608, "dense_mlp_w3"),
    (4096, 4608, 7168, "dense_mlp_w2"),
]

# Shared expert: (M, K, N, workload_id)
SHARED_EXPERT_MATMUL_PARAMS = [
    (4096, 7168, 512, "shared_expert_w1"),
    (4096, 7168, 512, "shared_expert_w3"),
    (4096, 512, 7168, "shared_expert_w2"),
]

# Routed expert: (M, K, N, workload_id)
ROUTED_EXPERT_MATMUL_PARAMS = [
    (1024, 7168, 2048, "routed_expert_w1"),
    (1024, 7168, 2048, "routed_expert_w3"),
    (1024, 2048, 7168, "routed_expert_w2"),
]
