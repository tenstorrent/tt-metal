# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests that check PCC (correctness) of Deepseek V3 matmuls with the shared
optimal program configurations. Single-chip only; one matmul per workload,
compare TT output to torch reference.
"""

import torch

import pytest
import ttnn

from models.common.utility_functions import comp_pcc, is_blackhole, skip_for_wormhole_b0

from tests.didt.deepseek_v3_matmul_config import (
    DENSE_MLP_MATMUL_PARAMS,
    GATE_MATMUL_CONFIG,
    GRID_SIZE,
    MLA_MATMUL_PARAMS,
    OPTIMAL_PROGRAM_CONFIG,
    ROUTED_EXPERT_MATMUL_PARAMS,
    SHARED_EXPERT_MATMUL_PARAMS,
    get_prefill_matmul_program_config,
)
from tests.didt.op_test_base import OpParameter, OpTestBase


# Minimum PCC to pass (same as comp_pcc default)
PCC_THRESHOLD = 0.99


def _pcc_workloads():
    """All batch=1 workloads that have optimal configs, as (M, K, N, in1_dtype, math_fidelity, workload_id)."""
    out = []
    # MLA: only batch=1
    for row in MLA_MATMUL_PARAMS:
        M, K, N, batch, in1_dtype, workload_id = row
        if batch != 1:
            continue
        out.append((M, K, N, in1_dtype, ttnn.MathFidelity.LoFi, workload_id))
    # Gate
    M, K, N, in1_dtype, workload_id = GATE_MATMUL_CONFIG
    out.append((M, K, N, in1_dtype, ttnn.MathFidelity.HiFi2, workload_id))
    # Dense MLP
    for M, K, N, workload_id in DENSE_MLP_MATMUL_PARAMS:
        out.append((M, K, N, ttnn.DataType.BFLOAT4_B, ttnn.MathFidelity.LoFi, workload_id))
    # Shared expert
    for M, K, N, workload_id in SHARED_EXPERT_MATMUL_PARAMS:
        out.append((M, K, N, ttnn.DataType.BFLOAT4_B, ttnn.MathFidelity.LoFi, workload_id))
    # Routed expert
    for M, K, N, workload_id in ROUTED_EXPERT_MATMUL_PARAMS:
        out.append((M, K, N, ttnn.DataType.BFLOAT4_B, ttnn.MathFidelity.LoFi, workload_id))
    return out


PCC_MATMUL_PARAMS = _pcc_workloads()


@skip_for_wormhole_b0("Grid 11x10 requires Blackhole")
@pytest.mark.parametrize(
    "M, K, N, in1_dtype, math_fidelity, workload_id",
    [pytest.param(*row, id=row[5]) for row in PCC_MATMUL_PARAMS],
)
@pytest.mark.parametrize("mesh_device", [pytest.param(1, id="1chips")], indirect=["mesh_device"])
def test_deepseek_v3_matmul_pcc(
    mesh_device,
    M,
    K,
    N,
    in1_dtype,
    math_fidelity,
    workload_id,
):
    """Compare TT matmul output to torch reference; assert PCC >= threshold."""
    dram_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    optimal_config = OPTIMAL_PROGRAM_CONFIG.get(workload_id)
    program_config = get_prefill_matmul_program_config(M, K, N, GRID_SIZE, optimal_config=optimal_config)
    ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    compute_config = ComputeConfigClass(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    batch = 1
    in0_shape = [1, batch, M, K]
    in1_shape = [1, batch, K, N]

    activation = OpParameter(in0_shape, ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT, dram_mem_config)
    arguments = [OpParameter(in1_shape, in1_dtype, ttnn.TILE_LAYOUT, dram_mem_config)]
    test = OpTestBase(
        mesh_device,
        activation=activation,
        arguments=arguments,
        out_mem_config=dram_mem_config,
        out_dtype=ttnn.DataType.BFLOAT16,
        program_config=program_config,
        compute_config=compute_config,
        loop_count=1,
        determinism_check_enabled=False,
        determinism_check_interval=False,
    )
    test.set_seed()

    a_shape = test.activation.shape
    b_shape = test.arguments[0].shape
    A = test.generate_torch_activations(a_shape)
    B = test.generate_torch_input(b_shape)

    # Torch reference (bfloat16 matmul)
    golden = torch.matmul(
        A.to(torch.bfloat16),
        B.to(torch.bfloat16),
    )

    a_t = test.generate_tt_activations_from_torch(A)
    test.inputs = [
        test.generate_tt_input_from_torch(
            B,
            test.arguments[0].dtype,
            test.arguments[0].layout,
            test.arguments[0].mem_config,
            0,
        )
    ]
    test.activations = test.convert_activations_to_memory_config(a_t)

    out = test.run_device_operation()
    # Single device: one tensor
    device_tensors = ttnn.get_device_tensors(out.cpu())
    calculated = ttnn.to_torch(device_tensors[0])
    out.deallocate(True)
    test.deallocate_activations()
    test.inputs[0].deallocate(True)

    passed, pcc_value = comp_pcc(golden, calculated, pcc=PCC_THRESHOLD)
    assert passed, f"{workload_id} ({M}x{K}x{N}): PCC {pcc_value:.4f} < {PCC_THRESHOLD}"
