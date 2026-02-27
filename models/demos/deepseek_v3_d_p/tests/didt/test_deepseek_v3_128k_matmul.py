# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
DIDT tests for Deepseek V3 prefill matmuls (128k sequence length).

Covers MLA, MoE gate, dense MLP, shared expert, and routed expert on
single-chip and 8×4 Galaxy. Per-chip grid: 11×10 (Blackhole only).
"""

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, skip_for_wormhole_b0
from models.demos.deepseek_v3_d_p.tests.deepseek_v3_matmul_config import DENSE_MLP_MATMUL_PARAMS as DENSE_MLP_PARAMS
from models.demos.deepseek_v3_d_p.tests.deepseek_v3_matmul_config import GATE_MATMUL_CONFIG, GRID_SIZE
from models.demos.deepseek_v3_d_p.tests.deepseek_v3_matmul_config import MLA_MATMUL_PARAMS as MLA_PARAMS
from models.demos.deepseek_v3_d_p.tests.deepseek_v3_matmul_config import OPTIMAL_PROGRAM_CONFIG
from models.demos.deepseek_v3_d_p.tests.deepseek_v3_matmul_config import (
    ROUTED_EXPERT_MATMUL_PARAMS as ROUTED_EXPERT_PARAMS,
)
from models.demos.deepseek_v3_d_p.tests.deepseek_v3_matmul_config import (
    SHARED_EXPERT_MATMUL_PARAMS as SHARED_EXPERT_PARAMS,
)
from models.demos.deepseek_v3_d_p.tests.deepseek_v3_matmul_config import get_prefill_matmul_program_config
from tests.didt.op_test_base import OpParameter, OpTestBase

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

# Single-chip and 8×4 Galaxy (32 chips).
MESH_DEVICE_PARAMS = [
    pytest.param(1, id="1chips"),
    pytest.param((8, 4), id="galaxy"),
]


class DeepseekV3MatmulTest(OpTestBase):
    """OpTestBase wrapper for Deepseek V3 prefill matmuls (DRAM, 11×10 grid)."""

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
        activation = OpParameter(in0_shape, in0_dtype, in0_layout, in0_mem_config)
        arguments = [OpParameter(in1_shape, in1_dtype, in1_layout, in1_mem_config)]
        super().__init__(
            mesh_device,
            activation,
            arguments,
            out_mem_config,
            out_dtype,
            program_config,
            compute_config,
            loop_count=loop_count,
            determinism_check_enabled=determinism_check_enabled,
            determinism_check_interval=determinism_check_interval,
        )


# ---------------------------------------------------------------------------
# Shared test runner
# ---------------------------------------------------------------------------


def _run_matmul_test(
    mesh_device,
    M: int,
    K: int,
    N: int,
    in1_dtype,
    math_fidelity,
    didt_workload_iterations,
    determinism_check_interval,
    batch: int = 1,
    grid_size: tuple[int, int] = GRID_SIZE,
    workload_id: str | None = None,
) -> None:
    """Run one Deepseek V3 matmul DIDT test (DRAM, optional optimal program config)."""
    dram_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    optimal_config = OPTIMAL_PROGRAM_CONFIG.get(workload_id) if workload_id else None
    program_config = (
        get_prefill_matmul_program_config(M, K, N, grid_size, optimal_config=optimal_config) if batch == 1 else None
    )
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
# MLA (6 matmuls): bfloat16 activations, bfloat8_b weights, HiFi2
# =============================================================================

MLA_MATMUL_PARAMS = [pytest.param(*row, id=row[5]) for row in MLA_PARAMS]


@skip_for_wormhole_b0("Grid 11x10 requires Blackhole")
@pytest.mark.parametrize("M, K, N, batch, in1_dtype, workload_id", MLA_MATMUL_PARAMS)
@pytest.mark.parametrize("mesh_device", MESH_DEVICE_PARAMS, indirect=["mesh_device"])
def test_deepseek_v3_mla_matmul(
    mesh_device,
    M,
    K,
    N,
    batch,
    in1_dtype,
    workload_id,
    didt_workload_iterations,
    determinism_check_interval,
):
    """MLA matmuls: bfloat16 in0, bfloat8_b in1, HiFi2."""
    _run_matmul_test(
        mesh_device=mesh_device,
        M=M,
        K=K,
        N=N,
        in1_dtype=in1_dtype,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        didt_workload_iterations=didt_workload_iterations,
        determinism_check_interval=determinism_check_interval,
        batch=batch,
        workload_id=workload_id,
    )


# =============================================================================
# Gate: bfloat16, HiFi2
# =============================================================================


@skip_for_wormhole_b0("Grid 11x10 requires Blackhole")
@pytest.mark.parametrize("mesh_device", MESH_DEVICE_PARAMS, indirect=["mesh_device"])
def test_deepseek_v3_gate_matmul(
    mesh_device,
    didt_workload_iterations,
    determinism_check_interval,
):
    """MoE gate matmul: 4096×1792×256, bfloat16, HiFi2."""
    M, K, N, in1_dtype, workload_id = GATE_MATMUL_CONFIG
    _run_matmul_test(
        mesh_device=mesh_device,
        M=M,
        K=K,
        N=N,
        in1_dtype=in1_dtype,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        didt_workload_iterations=didt_workload_iterations,
        determinism_check_interval=determinism_check_interval,
        workload_id=workload_id,
    )


# =============================================================================
# Dense MLP: bfloat4_b, LoFi (MoE)
# =============================================================================

DENSE_MLP_MATMUL_PARAMS = [pytest.param(*row, id=row[3]) for row in DENSE_MLP_PARAMS]


@skip_for_wormhole_b0("Grid 11x10 requires Blackhole")
@pytest.mark.parametrize("M, K, N, workload_id", DENSE_MLP_MATMUL_PARAMS)
@pytest.mark.parametrize("mesh_device", MESH_DEVICE_PARAMS, indirect=["mesh_device"])
def test_deepseek_v3_dense_mlp_matmul(
    mesh_device,
    M,
    K,
    N,
    workload_id,
    didt_workload_iterations,
    determinism_check_interval,
):
    """Dense MLP matmuls: bfloat4_b, LoFi."""
    _run_matmul_test(
        mesh_device=mesh_device,
        M=M,
        K=K,
        N=N,
        in1_dtype=ttnn.DataType.BFLOAT4_B,
        math_fidelity=ttnn.MathFidelity.LoFi,
        didt_workload_iterations=didt_workload_iterations,
        determinism_check_interval=determinism_check_interval,
        workload_id=workload_id,
    )


# =============================================================================
# Shared expert: bfloat4_b, LoFi (MoE)
# =============================================================================

SHARED_EXPERT_MATMUL_PARAMS = [pytest.param(*row, id=row[3]) for row in SHARED_EXPERT_PARAMS]


@skip_for_wormhole_b0("Grid 11x10 requires Blackhole")
@pytest.mark.parametrize("M, K, N, workload_id", SHARED_EXPERT_MATMUL_PARAMS)
@pytest.mark.parametrize("mesh_device", MESH_DEVICE_PARAMS, indirect=["mesh_device"])
def test_deepseek_v3_shared_expert_matmul(
    mesh_device,
    M,
    K,
    N,
    workload_id,
    didt_workload_iterations,
    determinism_check_interval,
):
    """Shared expert matmuls: bfloat4_b, LoFi."""
    _run_matmul_test(
        mesh_device=mesh_device,
        M=M,
        K=K,
        N=N,
        in1_dtype=ttnn.DataType.BFLOAT4_B,
        math_fidelity=ttnn.MathFidelity.LoFi,
        didt_workload_iterations=didt_workload_iterations,
        determinism_check_interval=determinism_check_interval,
        workload_id=workload_id,
    )


# =============================================================================
# Routed expert: bfloat4_b, LoFi (MoE)
# =============================================================================

ROUTED_EXPERT_MATMUL_PARAMS = [pytest.param(*row, id=row[3]) for row in ROUTED_EXPERT_PARAMS]


@skip_for_wormhole_b0("Grid 11x10 requires Blackhole")
@pytest.mark.parametrize("M, K, N, workload_id", ROUTED_EXPERT_MATMUL_PARAMS)
@pytest.mark.parametrize("mesh_device", MESH_DEVICE_PARAMS, indirect=["mesh_device"])
def test_deepseek_v3_routed_expert_matmul(
    mesh_device,
    M,
    K,
    N,
    workload_id,
    didt_workload_iterations,
    determinism_check_interval,
):
    """Routed expert matmuls: bfloat4_b, LoFi."""
    _run_matmul_test(
        mesh_device=mesh_device,
        M=M,
        K=K,
        N=N,
        in1_dtype=ttnn.DataType.BFLOAT4_B,
        math_fidelity=ttnn.MathFidelity.LoFi,
        didt_workload_iterations=didt_workload_iterations,
        determinism_check_interval=determinism_check_interval,
        workload_id=workload_id,
    )
