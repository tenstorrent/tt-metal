# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
DIDT tests for Deepseek V3 matmuls at 128k sequence length (prefill only).

Tests cover per-chip matmul shapes for MLA (TP=4, SP=32), MoE gate,
dense MLP, shared expert, and routed expert operations on a 4×32 mesh
(128 chips) and single-chip configurations.

Per-chip grid: 11×10 (110 worker cores) only.
"""

import math

from loguru import logger
import pytest
import torch

from tests.didt.op_test_base import OpTestBase
import ttnn
from models.common.utility_functions import skip_for_wormhole_b0, is_blackhole

# Per-chip grid for Deepseek V3 128k DIDT tests: 11×10 = 110 worker cores
GRID_SIZE = (11, 10)
TILE_SIZE = 32


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


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


def _get_prefill_matmul_program_config(M, K, N, grid_size=GRID_SIZE):
    """Compute MatmulMultiCoreReuseMultiCastProgramConfig for a prefill matmul.

    Follows the same logic as DeepseekV3MLP._get_prefill_pc() in
    models/demos/deepseek_v3/tt/mlp/mlp.py.
    """
    grid_x, grid_y = grid_size
    M_tiles = math.ceil(M / TILE_SIZE)
    K_tiles = math.ceil(K / TILE_SIZE)
    N_tiles = math.ceil(N / TILE_SIZE)

    per_core_M = math.ceil(M_tiles / grid_y)
    per_core_N = math.ceil(N_tiles / grid_x)
    in0_block_w = _find_largest_divisor(K_tiles, 8)
    out_subblock_w = _find_largest_divisor(per_core_N, 4)

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class DeepseekV3MatmulTest(OpTestBase):
    """DIDT test wrapper for Deepseek V3 prefill matmuls."""

    def __init__(
        self,
        mesh_device,
        in0_shape,
        in1_shape,
        in0_mem_config,
        in1_mem_config,
        out_mem_config,
        in0_dtype,
        in1_dtype,
        out_dtype,
        in0_layout,
        in1_layout,
        program_config,
        compute_config,
        loop_count=1000,
        determinism_check_enabled=False,
        determinism_check_interval=False,
    ):
        super().__init__(
            mesh_device,
            in0_shape,
            in1_shape,
            in0_mem_config,
            in1_mem_config,
            out_mem_config,
            in0_dtype,
            in1_dtype,
            out_dtype,
            in0_layout,
            in1_layout,
            program_config,
            compute_config,
            loop_count,
            determinism_check_enabled,
            determinism_check_interval,
        )


# ---------------------------------------------------------------------------
# Shared parametrization
# ---------------------------------------------------------------------------

# Mesh device parametrization: single-chip and 4×32 (128 chips)
MESH_DEVICE_PARAMS = [
    pytest.param(1, id="1chips"),
    pytest.param((4, 32), id="4_galaxy_128chips"),
]


# ---------------------------------------------------------------------------
# Common test runner
# ---------------------------------------------------------------------------


def _run_matmul_test(
    mesh_device,
    M,
    K,
    N,
    in1_dtype,
    math_fidelity,
    didt_workload_iterations,
    determinism_check_interval,
    batch=1,
    grid_size=GRID_SIZE,
):
    """Set up and run a single Deepseek V3 matmul DIDT test."""

    # Memory configs: DRAM interleaved for prefill
    dram_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    # Program config – explicit for non-batched; auto-generated for batched
    if batch == 1:
        program_config = _get_prefill_matmul_program_config(M, K, N, grid_size)
    else:
        program_config = None

    # Compute kernel config
    ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    compute_config = ComputeConfigClass(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    in0_shape = [1, batch, M, K]
    in1_shape = [1, batch, K, N]

    logger.info(
        f"Deepseek V3 DIDT matmul: in0={in0_shape}, in1={in1_shape}, "
        f"in1_dtype={in1_dtype}, fidelity={math_fidelity}"
    )

    test = DeepseekV3MatmulTest(
        mesh_device,
        in0_shape=in0_shape,
        in1_shape=in1_shape,
        in0_mem_config=dram_mem_config,
        in1_mem_config=dram_mem_config,
        out_mem_config=dram_mem_config,
        in0_dtype=ttnn.DataType.BFLOAT16,
        in1_dtype=in1_dtype,
        out_dtype=ttnn.DataType.BFLOAT16,
        in0_layout=ttnn.TILE_LAYOUT,
        in1_layout=ttnn.TILE_LAYOUT,
        program_config=program_config,
        compute_config=compute_config,
        loop_count=didt_workload_iterations,
        determinism_check_enabled=determinism_check_interval > 0,
        determinism_check_interval=determinism_check_interval,
    )

    test.run_op_test()


# =============================================================================
# MLA matmul tests (6 ops)
# =============================================================================

MLA_MATMUL_PARAMS = [
    # x @ wq_a: x column parallel, 4096×1792 × 1792×1536
    pytest.param(4096, 1792, 1536, 1, ttnn.DataType.BFLOAT8_B, id="x_wq_a"),
    # x @ wkv_a: x column parallel, 4096×1792 × 1792×576
    pytest.param(4096, 1792, 576, 1, ttnn.DataType.BFLOAT8_B, id="x_wkv_a"),
    # q @ wq_b: q column-replicated, 4096×1536 × 1536×6144
    pytest.param(4096, 1536, 6144, 1, ttnn.DataType.BFLOAT8_B, id="q_wq_b"),
    # q_nope @ wkv_b1: 32 heads/chip batched, 4096×128 × 128×512
    pytest.param(4096, 128, 512, 32, ttnn.DataType.BFLOAT8_B, id="q_nope_wkv_b1"),
    # v_out @ wkv_b2: 32 heads/chip batched, 4096×512 × 512×128
    pytest.param(4096, 512, 128, 32, ttnn.DataType.BFLOAT8_B, id="v_out_wkv_b2"),
    # v_out @ out_proj: K sharded (16384/4=4096), N=7168 full
    pytest.param(4096, 4096, 7168, 1, ttnn.DataType.BFLOAT8_B, id="v_out_out_proj"),
]


@skip_for_wormhole_b0("Grid 11x10 requires Blackhole")
@pytest.mark.parametrize("M, K, N, batch, in1_dtype", MLA_MATMUL_PARAMS)
@pytest.mark.parametrize("mesh_device", MESH_DEVICE_PARAMS, indirect=["mesh_device"])
def test_deepseek_v3_mla_matmul(
    mesh_device,
    M,
    K,
    N,
    batch,
    in1_dtype,
    didt_workload_iterations,
    determinism_check_interval,
):
    """DIDT test for Deepseek V3 MLA matmuls (prefill, 128k seq len)."""
    _run_matmul_test(
        mesh_device=mesh_device,
        M=M,
        K=K,
        N=N,
        in1_dtype=in1_dtype,
        math_fidelity=ttnn.MathFidelity.LoFi,
        didt_workload_iterations=didt_workload_iterations,
        determinism_check_interval=determinism_check_interval,
        batch=batch,
    )


# =============================================================================
# Gate matmul test
# =============================================================================


@skip_for_wormhole_b0("Grid 11x10 requires Blackhole")
@pytest.mark.parametrize("mesh_device", MESH_DEVICE_PARAMS, indirect=["mesh_device"])
def test_deepseek_v3_gate_matmul(
    mesh_device,
    didt_workload_iterations,
    determinism_check_interval,
):
    """DIDT test for Deepseek V3 MoE gate matmul (prefill, 128k seq len).

    Per-chip shape: 4096×1792×256 (K=7168/4 after TP=4 sharding).
    """
    _run_matmul_test(
        mesh_device=mesh_device,
        M=4096,
        K=1792,
        N=256,
        in1_dtype=ttnn.DataType.BFLOAT16,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        didt_workload_iterations=didt_workload_iterations,
        determinism_check_interval=determinism_check_interval,
    )


# =============================================================================
# Dense MLP matmul tests
# =============================================================================

DENSE_MLP_MATMUL_PARAMS = [
    # w1 (gate_proj): 4096×7168 × 7168×4608 (intermediate=18432/4)
    pytest.param(4096, 7168, 4608, id="dense_mlp_w1"),
    # w3 (up_proj): 4096×7168 × 7168×4608
    pytest.param(4096, 7168, 4608, id="dense_mlp_w3"),
    # w2 (down_proj): 4096×4608 × 4608×7168
    pytest.param(4096, 4608, 7168, id="dense_mlp_w2"),
]


@skip_for_wormhole_b0("Grid 11x10 requires Blackhole")
@pytest.mark.parametrize("M, K, N", DENSE_MLP_MATMUL_PARAMS)
@pytest.mark.parametrize("mesh_device", MESH_DEVICE_PARAMS, indirect=["mesh_device"])
def test_deepseek_v3_dense_mlp_matmul(
    mesh_device,
    M,
    K,
    N,
    didt_workload_iterations,
    determinism_check_interval,
):
    """DIDT test for Deepseek V3 dense MLP matmuls (prefill, 128k seq len)."""
    _run_matmul_test(
        mesh_device=mesh_device,
        M=M,
        K=K,
        N=N,
        in1_dtype=ttnn.DataType.BFLOAT4_B,
        math_fidelity=ttnn.MathFidelity.LoFi,
        didt_workload_iterations=didt_workload_iterations,
        determinism_check_interval=determinism_check_interval,
    )


# =============================================================================
# Shared expert matmul tests
# =============================================================================

SHARED_EXPERT_MATMUL_PARAMS = [
    # w1 (gate_proj): 4096×7168 × 7168×512
    pytest.param(4096, 7168, 512, id="shared_expert_w1"),
    # w3 (up_proj): 4096×7168 × 7168×512
    pytest.param(4096, 7168, 512, id="shared_expert_w3"),
    # w2 (down_proj): 4096×512 × 512×7168
    pytest.param(4096, 512, 7168, id="shared_expert_w2"),
]


@skip_for_wormhole_b0("Grid 11x10 requires Blackhole")
@pytest.mark.parametrize("M, K, N", SHARED_EXPERT_MATMUL_PARAMS)
@pytest.mark.parametrize("mesh_device", MESH_DEVICE_PARAMS, indirect=["mesh_device"])
def test_deepseek_v3_shared_expert_matmul(
    mesh_device,
    M,
    K,
    N,
    didt_workload_iterations,
    determinism_check_interval,
):
    """DIDT test for Deepseek V3 shared expert matmuls (prefill, 128k seq len)."""
    _run_matmul_test(
        mesh_device=mesh_device,
        M=M,
        K=K,
        N=N,
        in1_dtype=ttnn.DataType.BFLOAT4_B,
        math_fidelity=ttnn.MathFidelity.LoFi,
        didt_workload_iterations=didt_workload_iterations,
        determinism_check_interval=determinism_check_interval,
    )


# =============================================================================
# Routed expert matmul tests
# =============================================================================

ROUTED_EXPERT_MATMUL_PARAMS = [
    # w1 (gate_proj): 1024×7168 × 7168×2048
    pytest.param(1024, 7168, 2048, id="routed_expert_w1"),
    # w3 (up_proj): 1024×7168 × 7168×2048
    pytest.param(1024, 7168, 2048, id="routed_expert_w3"),
    # w2 (down_proj): 1024×2048 × 2048×7168
    pytest.param(1024, 2048, 7168, id="routed_expert_w2"),
]


@skip_for_wormhole_b0("Grid 11x10 requires Blackhole")
@pytest.mark.parametrize("M, K, N", ROUTED_EXPERT_MATMUL_PARAMS)
@pytest.mark.parametrize("mesh_device", MESH_DEVICE_PARAMS, indirect=["mesh_device"])
def test_deepseek_v3_routed_expert_matmul(
    mesh_device,
    M,
    K,
    N,
    didt_workload_iterations,
    determinism_check_interval,
):
    """DIDT test for Deepseek V3 routed expert matmuls (prefill, 128k seq len).

    M=1024 = 4096/4 (chunked for routed experts).
    """
    _run_matmul_test(
        mesh_device=mesh_device,
        M=M,
        K=K,
        N=N,
        in1_dtype=ttnn.DataType.BFLOAT4_B,
        math_fidelity=ttnn.MathFidelity.LoFi,
        didt_workload_iterations=didt_workload_iterations,
        determinism_check_interval=determinism_check_interval,
    )
