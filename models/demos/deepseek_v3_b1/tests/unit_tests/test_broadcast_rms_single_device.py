# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Single-device test for fused Broadcast+RMSNorm op with skip_ccl=True.

When skip_ccl=True, the fused op runs only the RMSNorm portion without CCL broadcast,
making it suitable for single-device execution and testing.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.fused_ops.broadcast_rms.op import BroadcastRMSNorm


@pytest.mark.parametrize(
    "output_shape, input_shard_shape, tensor_mem_layout",
    [
        ([1, 7168], (1, 7168), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("fp32_dest_acc_en", [False])
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_broadcast_rms_single_device(
    mesh_device,
    output_shape,
    input_shard_shape,
    tensor_mem_layout,
    layout,
    input_dtype,
    epsilon,
    fp32_dest_acc_en,
):
    """
    Test fused Broadcast+RMSNorm op on a single device with skip_ccl=True for Debugging purpose.

    """
    logger.info(f"Testing BroadcastRMSNorm (skip_ccl=True) with shape={output_shape}, epsilon={epsilon}")

    # Set up sharded memory config (single core shard)
    input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec)
    output_mem_config = input_mem_config

    # Create input tensor
    torch_input = torch.rand(output_shape, dtype=torch.bfloat16)

    # Create gamma tensor
    torch_gamma = torch.randn(tuple(output_shape), dtype=torch.bfloat16)

    # Convert tensors to device using mesh_device fixture
    input_tensor = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Intermediate tensor (same shape/config as input)
    intermediate_tensor = ttnn.from_torch(
        torch.zeros_like(torch_input),
        device=mesh_device,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    gamma_tensor = ttnn.from_torch(
        torch_gamma,
        device=mesh_device,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=output_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Create output tensor
    output_tensor = ttnn.from_torch(
        torch.zeros(tuple(output_shape), dtype=torch.bfloat16),
        device=mesh_device,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=output_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Compute expected output using PyTorch reference
    torch_expected = BroadcastRMSNorm.golden(torch_input, torch_gamma, epsilon=epsilon)

    # Run fused operation with skip_ccl=True (single-device mode)
    logger.info("Running fused Broadcast+RMSNorm with skip_ccl=True")
    sender_coord = ttnn.MeshCoordinate(0, 0)  # Ignored when skip_ccl=True

    result = BroadcastRMSNorm.op(
        input_tensor,
        intermediate_tensor,
        gamma_tensor,
        sender_coord,
        output_tensor,
        semaphores=None,  # Not needed when skip_ccl=True
        skip_ccl=True,
        epsilon=epsilon,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )

    ttnn.synchronize_device(mesh_device)

    # Convert result back to torch
    output_tensor_torch = ttnn.to_torch(result, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Verify output
    assert (
        output_tensor_torch.shape == torch_expected.shape
    ), f"Shape mismatch: got {output_tensor_torch.shape}, expected {torch_expected.shape}"

    max_diff = torch.max(torch.abs(output_tensor_torch - torch_expected)).item()
    mean_diff = torch.mean(torch.abs(output_tensor_torch - torch_expected)).item()

    logger.info(f"Max absolute difference: {max_diff}")
    logger.info(f"Mean absolute difference: {mean_diff}")

    passing, pcc_message = comp_pcc(torch_expected, output_tensor_torch, 0.999)
    logger.info(pcc_message)
    assert passing, f"PCC check failed: {pcc_message}"

    logger.info("BroadcastRMSNorm single-device test passed!")
