# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TG (Single Galaxy) MoE Reduce Test for Quad Galaxy Validation

Tests non-optimized reduce operations used in the MoE pipeline on a 4x8 mesh (32 devices).
These operations replace the optimized versions that are hardcoded for 8-device setups.

Key Design Principles:
1. **Standard Operations**: Tests ttnn.sum and ttnn.reduce_scatter (not optimized versions)
2. **Pipeline Configuration**: Uses same shapes/configs as E2E test
3. **Correctness Focus**: Validates reduction results match expected outputs

Operations Tested:
1. ttnn.sum(dim=0) - Reduces across selected experts (K dimension)
2. ttnn.reduce_scatter(dim=-1, cluster_axis=1) - Reduces across tensor parallel dimension

Configuration:
- Mesh: 4x8 (32 devices)
- cluster_axis: 0 (dispatch along axis-0)
- Reduction across 8 replicated devices (cluster_axis=1)
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc


def run_reduce_test(
    mesh_device,
    mesh_shape,
    cluster_axis,
    select_experts_k,
    tokens_per_device,
    hidden_size,
    num_iterations,
):
    """
    Run reduce operations test on TG 4x8 mesh.

    This test:
    1. Creates test tensor with shape [K, batch, hidden] (after combine + tilize + scale)
    2. Runs ttnn.sum(dim=0) to reduce across K experts
    3. Runs ttnn.reduce_scatter to reduce across tensor parallel dimension
    4. Validates outputs against golden references
    """
    torch.manual_seed(2003)

    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch_devices = mesh_shape[cluster_axis]
    num_replicated_devices = num_devices // num_dispatch_devices
    batch = tokens_per_device * num_dispatch_devices

    logger.info("=" * 80)
    logger.info(f"TG MoE Reduce Test Configuration:")
    logger.info(f"  Mesh shape: {mesh_shape} ({num_devices} devices)")
    logger.info(f"  Cluster axis: {cluster_axis}")
    logger.info(f"  Batch: {batch} (tokens_per_device={tokens_per_device})")
    logger.info(f"  Selected experts K: {select_experts_k}")
    logger.info(f"  Hidden size: {hidden_size}")
    logger.info(f"  Num replicated devices: {num_replicated_devices}")
    logger.info(f"  Iterations: {num_iterations}")
    logger.info("=" * 80)

    # Memory configs
    input_memory_config = ttnn.L1_MEMORY_CONFIG
    sum_output_memory_config = ttnn.L1_MEMORY_CONFIG
    reduce_scatter_output_memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Create test tensors for each iteration
    torch_input_tensors = []
    torch_sum_goldens = []
    torch_reduce_scatter_goldens = []

    for _ in range(num_iterations):
        # Input tensor shape: [K, batch, hidden_size] (per device after sharding)
        # This simulates the output of combine → tilize → scale operations
        torch_input = torch.rand(select_experts_k, tokens_per_device, hidden_size, dtype=torch.bfloat16) - 0.5
        torch_input_tensors.append(torch_input)

        # Golden for sum: reduce along dim=0 (K dimension)
        torch_sum_golden = torch.sum(torch_input, dim=0)  # Shape: [batch, hidden_size]
        torch_sum_goldens.append(torch_sum_golden)

        # Golden for reduce_scatter: sum across replicated devices, then scatter along hidden dim
        # Each device gets hidden_size / num_replicated_devices elements
        # Simulate the all-reduce by summing num_replicated_devices copies
        torch_replicated = torch_sum_golden.repeat(num_replicated_devices, 1, 1)  # [8, batch, hidden]
        torch_reduced = torch.sum(torch_replicated, dim=0)  # [batch, hidden]

        # Scatter along hidden dimension
        hidden_per_device = hidden_size // num_replicated_devices
        torch_reduce_scatter_golden = torch_reduced[:, :hidden_per_device]  # Take first chunk
        torch_reduce_scatter_goldens.append(torch_reduce_scatter_golden)

    # Create TT input tensors
    tt_input_tensors = []
    for torch_input in torch_input_tensors:
        tt_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),  # Replicate across all devices
        )
        tt_input_tensors.append(tt_input)

    logger.info("Running reduce operations...")

    # Run operations
    tt_sum_outputs = []
    tt_reduce_scatter_outputs = []

    for i in range(num_iterations):
        # Operation 1: Sum across expert dimension (dim=0)
        tt_sum_output = ttnn.sum(
            tt_input_tensors[i],
            dim=0,
            memory_config=sum_output_memory_config,
        )
        tt_sum_outputs.append(tt_sum_output)

        # Operation 2: Reduce-scatter across tensor parallel dimension
        # cluster_axis=1 means reduce across the replicated devices (8 devices)
        # dim=-1 means scatter along the last dimension (hidden_size)
        tt_reduce_scatter_output = ttnn.reduce_scatter(
            tt_sum_output,
            dim=-1,
            cluster_axis=1,
            num_links=4,
            memory_config=reduce_scatter_output_memory_config,
        )
        tt_reduce_scatter_outputs.append(tt_reduce_scatter_output)

        logger.info(f"Iteration {i + 1}/{num_iterations} completed")

    ttnn.synchronize_device(mesh_device)
    logger.info("All iterations completed")

    # Validate outputs
    logger.info("Validating outputs...")

    PCC_THRESHOLD = 0.99
    all_passed = True

    for i in range(num_iterations):
        logger.info(f"\nValidating iteration {i}:")

        # Validate sum output
        # Get output from first device only (all devices should have same result since input was replicated)
        tt_sum_device_tensors = ttnn.get_device_tensors(tt_sum_outputs[i])
        tt_sum_torch_device0 = ttnn.to_torch(tt_sum_device_tensors[0])

        sum_pcc_passed, sum_pcc = comp_pcc(tt_sum_torch_device0, torch_sum_goldens[i], pcc=PCC_THRESHOLD)
        logger.info(f"  Sum PCC: {sum_pcc:.6f}")

        if not sum_pcc_passed:
            logger.warning(f"FAILED sum validation at iteration {i}: PCC {sum_pcc}")
            all_passed = False

        # Validate reduce_scatter output
        # After reduce_scatter, each device in the replicated group gets a different chunk
        # We compare the first device's chunk
        tt_rs_device_tensors = ttnn.get_device_tensors(tt_reduce_scatter_outputs[i])
        tt_rs_torch_device0 = ttnn.to_torch(tt_rs_device_tensors[0])

        rs_pcc_passed, rs_pcc = comp_pcc(tt_rs_torch_device0, torch_reduce_scatter_goldens[i], pcc=PCC_THRESHOLD)
        logger.info(f"  Reduce-scatter PCC: {rs_pcc:.6f}")

        if not rs_pcc_passed:
            logger.warning(f"FAILED reduce_scatter validation at iteration {i}: PCC {rs_pcc}")
            all_passed = False

    logger.info(f"\nTG MoE Reduce Test: {'PASSED' if all_passed else 'FAILED'}")
    assert all_passed, "TG MoE Reduce test failed!"
    logger.info("TG MoE Reduce test passed!")


@pytest.mark.requires_device("TG")
@pytest.mark.skipif(
    (os.getenv("USE_TORUS_MODE") is None),
    reason="Requires ring fabric",
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 500000,
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((4, 8), (4, 8), id="4x8_tg"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("tokens_per_device", [32])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("num_iterations", [3])
def test_reduce_correctness(
    mesh_device,
    mesh_shape,
    cluster_axis,
    select_experts_k,
    tokens_per_device,
    hidden_size,
    num_iterations,
):
    """Correctness test for TG reduce operations."""
    run_reduce_test(
        mesh_device,
        mesh_shape,
        cluster_axis,
        select_experts_k,
        tokens_per_device,
        hidden_size,
        num_iterations,
    )
