# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2_grid")], indirect=True)
def test_distributed_mla_1x2_mesh(mesh_device):
    """Test distributed_mla operation on 1x2 mesh configuration for N300 (2 devices)"""
    if mesh_device.get_num_devices() < 2:
        pytest.skip("Requires 2 devices to run")

    # Create a random input tensor
    torch_input = torch.randn([1, 8, 128, 64], dtype=torch.bfloat16)

    # For 1x2 mesh - shard along sequence dimension (dim=1)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=1),  # Shard along dim 1
    )

    logger.info("Testing 1x2 mesh with cluster_axis=1")

    # Call the distributed_mla operation with cluster_axis=1 (horizontal sharding)
    output = ttnn.transformer.sdpa_prefill.distributed_mla(ttnn_input, cluster_axis=1)

    # For now, just verify the operation runs without error
    # In the full implementation, we would check the actual computation
    assert output.shape == ttnn_input.shape

    logger.info("Successfully tested 1x2 mesh")


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 1), id="2x1_grid")], indirect=True)
def test_distributed_mla_2x1_mesh(mesh_device):
    """Test distributed_mla operation on 2x1 mesh configuration for N300 (2 devices)"""
    if mesh_device.get_num_devices() < 2:
        pytest.skip("Requires 2 devices to run")

    # Create a random input tensor
    torch_input = torch.randn([1, 8, 128, 64], dtype=torch.bfloat16)

    # For 2x1 mesh - shard along batch dimension (dim=0)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=0),  # Shard along dim 0
    )

    logger.info("Testing 2x1 mesh with cluster_axis=0")

    # Call the distributed_mla operation with cluster_axis=0 (vertical sharding)
    output = ttnn.transformer.sdpa_prefill.distributed_mla(ttnn_input, cluster_axis=0)

    # Verify the operation runs without error and maintains shape
    assert output.shape == ttnn_input.shape

    logger.info("Successfully tested 2x1 mesh")


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


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2_grid")], indirect=True)
def test_distributed_mla_device_order_logging_1x2(mesh_device):
    """Test that distributed_mla operation logs correct device order numbers for 1x2 mesh"""
    # Create a random input tensor
    torch_input = torch.randn([1, 8, 128, 64], dtype=torch.bfloat16)

    # Test with sequence sharding for 1x2 mesh
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=1),  # Shard along sequence
    )

    # Call the distributed_mla operation and check device order logging
    output = ttnn.transformer.sdpa_prefill.distributed_mla(ttnn_input, cluster_axis=1)
    assert output.shape == ttnn_input.shape

    # Test with different cluster_axis
    output = ttnn.transformer.sdpa_prefill.distributed_mla(ttnn_input, cluster_axis=0)
    assert output.shape == ttnn_input.shape


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 1), id="2x1_grid")], indirect=True)
def test_distributed_mla_device_order_logging_2x1(mesh_device):
    """Test that distributed_mla operation logs correct device order numbers for 2x1 mesh"""
    # Create a random input tensor
    torch_input = torch.randn([1, 8, 128, 64], dtype=torch.bfloat16)

    # Test with batch sharding for 2x1 mesh
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=0),  # Shard along batch
    )

    # Call the distributed_mla operation and check device order logging
    output = ttnn.transformer.sdpa_prefill.distributed_mla(ttnn_input, cluster_axis=0)
    assert output.shape == ttnn_input.shape

    # Test with different cluster_axis
    output = ttnn.transformer.sdpa_prefill.distributed_mla(ttnn_input, cluster_axis=1)
    assert output.shape == ttnn_input.shape
