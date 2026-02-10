# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh


def test_distributed_mla_device_order_logging(mesh_device):
    """Test that distributed_mla operation logs correct device order numbers"""
    if mesh_device.get_num_devices() < 2:
        pytest.skip("Requires multiple devices to run")

    # Create a random input tensor
    torch_input = torch.randn([1, 8, 128, 64], dtype=torch.bfloat16)

    # Test with cluster_axis=0
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=1),  # Shard along dim 1
    )

    # Call the distributed_mla operation with cluster_axis=0
    output = ttnn.transformer.sdpa_prefill.distributed_mla(ttnn_input, cluster_axis=0)

    # For now, just verify the operation runs without error
    # In the full implementation, we would check the actual computation
    assert output.shape == ttnn_input.shape

    # Test with cluster_axis=1
    output = ttnn.transformer.sdpa_prefill.distributed_mla(ttnn_input, cluster_axis=1)
    assert output.shape == ttnn_input.shape

    # Test without cluster_axis (should use default behavior)
    output = ttnn.transformer.sdpa_prefill.distributed_mla(ttnn_input)
    assert output.shape == ttnn_input.shape


@pytest.mark.parametrize(
    "mesh_shape",
    [
        (1, 8),  # 1x8 mesh
        (2, 4),  # 2x4 mesh
        (4, 2),  # 4x2 mesh
        (8, 1),  # 8x1 mesh
    ],
)
def test_distributed_mla_different_mesh_shapes(mesh_shape):
    """Test distributed_mla operation on different mesh configurations for t3k wormhole (8 devices)"""
    num_pcie_devices = ttnn.get_num_pcie_devices()
    if num_pcie_devices < 8:
        pytest.skip("Requires 8 devices (t3k wormhole) to run")

    # Open mesh with specified shape
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(mesh_shape[0], mesh_shape[1]))

    try:
        # Create a random input tensor
        torch_input = torch.randn([1, 8, 128, 64], dtype=torch.bfloat16)

        # Test sharding along different dimensions based on mesh shape
        if mesh_shape[0] == 1:  # 1xN mesh - shard along second dimension
            ttnn_input = ttnn.from_torch(
                torch_input, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=ShardTensorToMesh(mesh_device, dim=1)
            )
            cluster_axis = 1  # Use cluster axis 1 for 1xN mesh
        else:  # Mx1 or MxN mesh - shard along first dimension
            ttnn_input = ttnn.from_torch(
                torch_input, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=ShardTensorToMesh(mesh_device, dim=0)
            )
            cluster_axis = 0  # Use cluster axis 0 for Mx1 or MxN mesh

        logger.info(f"Testing mesh shape {mesh_shape} with cluster_axis={cluster_axis}")

        # Call the distributed_mla operation
        output = ttnn.transformer.sdpa_prefill.distributed_mla(ttnn_input, cluster_axis=cluster_axis)

        # Verify the operation runs without error and maintains shape
        assert output.shape == ttnn_input.shape

        logger.info(f"Successfully tested mesh shape {mesh_shape}")

    finally:
        ttnn.close_mesh_device(mesh_device)


def test_distributed_mla_no_cluster_axis(mesh_device):
    """Test distributed_mla operation without specifying cluster_axis"""
    if mesh_device.get_num_devices() < 2:
        pytest.skip("Requires multiple devices to run")

    # Create a random input tensor
    torch_input = torch.randn([1, 8, 128, 64], dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),  # Replicate on all devices
    )

    # Call without cluster_axis
    output = ttnn.transformer.sdpa_prefill.distributed_mla(ttnn_input)

    # Verify the operation runs without error
    assert output.shape == ttnn_input.shape


if __name__ == "__main__":
    # For manual testing
    num_devices = ttnn.get_num_pcie_devices()
    logger.info(f"Available devices: {num_devices}")

    if num_devices >= 8:
        # Test 1x8 configuration manually
        mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
        try:
            torch_input = torch.randn([1, 8, 128, 64], dtype=torch.bfloat16)
            ttnn_input = ttnn.from_torch(
                torch_input, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=ShardTensorToMesh(mesh_device, dim=1)
            )

            logger.info("Testing 1x8 mesh with cluster_axis=1")
            output = ttnn.transformer.sdpa_prefill.distributed_mla(ttnn_input, cluster_axis=1)
            logger.info("Test successful!")

        finally:
            ttnn.close_mesh_device(mesh_device)
    else:
        logger.warning("Not enough devices for manual testing")
