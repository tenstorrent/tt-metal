# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
import transformers


#######
# Test MultiDevice Initialization, Open/Close
#######
def test_device_mesh_open_close_explicit(silicon_arch_name, silicon_arch_wormhole_b0):
    """Manually open and close multi-device"""
    num_pcie_devices = ttnn.get_num_pcie_devices()
    if num_pcie_devices <= 1:
        pytest.skip("Requires multiple devices to run")

    device_grid, device_ids = ttnn.DeviceGrid(2, 2), ttnn.get_pcie_device_ids()
    multi_device = ttnn.open_device_mesh(device_grid, device_ids)
    ttnn.close_device_mesh(multi_device)


def test_multi_device_open_close_full_device_mesh_fixture(device_mesh):
    """Using `device_mesh` pytest fixture defined in conftest.py"""
    pass


def test_multi_device_open_close_full_device_mesh_fixture(pcie_device_mesh):
    """Using `pcie_device_mesh` pytest fixture defined in conftest.py"""
    pass


def test_multi_device_open_close_using_context_manager(silicon_arch_name, silicon_arch_wormhole_b0):
    """Using context manager to open and close multi-device"""
    device_grid, device_ids = ttnn.DeviceGrid(2, 2), ttnn.get_device_ids()
    if len(device_ids) <= 1:
        pytest.skip()
    with ttnn.create_device_mesh(device_grid, device_ids) as device_mesh:
        # Do something with multi_device
        pass


#######
# Simple Multi-Device Tensor tests
#######


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_ttnn_to_and_from_multi_device_shard(pcie_device_mesh, layout, memory_config):
    """Shard a tensor across devices, compose it back and verify loopback tensor is same as the original tensor"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor

    torch_tensor = torch.rand((1, 1, 32, 256), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(torch_tensor, mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=3))
    ttnn_tensor = ttnn.to_layout(ttnn_tensor, layout=layout)
    ttnn_tensor = ttnn.to_device(ttnn_tensor, pcie_device_mesh, memory_config=memory_config)
    ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
    torch_loop_back_tensor = ttnn.to_torch(
        ttnn_loop_back_tensor, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=3)
    )

    assert torch.all(torch_tensor == torch_loop_back_tensor)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_multi_device_check_per_device_shard(pcie_device_mesh, layout, memory_config):
    """This test checks if the tensor is correctly sharded across devices"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor

    torch_tensor = torch.rand((1, 1, 32, 256), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(torch_tensor, mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=3))
    ttnn_tensor = ttnn.to_layout(ttnn_tensor, layout=layout)
    ttnn_tensor = ttnn.to_device(ttnn_tensor, pcie_device_mesh, memory_config=memory_config)
    ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)

    shard_offset, shard_size = 0, 64
    for device_tensor in ttnn.get_device_tensors(ttnn_loop_back_tensor):
        device_tensor_torch = ttnn.to_torch(device_tensor)
        assert torch.all(device_tensor_torch == torch_tensor[..., shard_offset : shard_offset + shard_size])
        shard_offset += shard_size


def test_multi_device_replicate(pcie_device_mesh):
    """Test ReplicateTensorToMesh to broadcast a tensor across multiple devices"""
    from ttnn import ReplicateTensorToMesh, ListMeshToTensor

    full_tensor = torch.rand((1, 1, 32, 128), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(full_tensor, mesh_mapper=ReplicateTensorToMesh(pcie_device_mesh))
    ttnn_tensor = ttnn.to_device(ttnn_tensor, pcie_device_mesh)
    ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
    loopback_replicated_tensors = ttnn.to_torch(ttnn_loop_back_tensor, mesh_composer=ListMeshToTensor(pcie_device_mesh))
    for loopback_replicated_tensor in loopback_replicated_tensors:
        assert torch.all(full_tensor == loopback_replicated_tensor)


def test_ttnn_multi_device_all_gather(pcie_device_mesh):
    """Multidevice API test for ttnn.all_gather CCL operation"""
    from ttnn import ShardTensorToMesh

    full_tensor = torch.rand((1, 1, 32, 128), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(full_tensor, mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=3))
    ttnn_tensor = ttnn.to_device(ttnn_tensor, pcie_device_mesh)
    ttnn_tensor = ttnn.all_gather(ttnn_tensor, dim=3, num_links=1)

    device_tensors: typing.List[ttnn.Tensor] = ttnn.get_device_tensors(ttnn_tensor)
    for device_tensor in device_tensors:
        device_tensor_torch = ttnn.to_torch(device_tensor)
        assert torch.all(device_tensor_torch == full_tensor)


def test_multi_device_single_op_unary(pcie_device_mesh):
    """Multidevice API test: Running tensor-parallel multi-device single-op unary"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor

    torch_input_tensor = torch.rand((1, 1, 32, 128), dtype=torch.bfloat16)
    torch_output_golden = torch.nn.functional.gelu(torch_input_tensor)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=3),
        device=pcie_device_mesh,
    )
    ttnn_output_tensor = ttnn.gelu(ttnn_input_tensor)

    ttnn_torch_output_tensor = ttnn.to_torch(
        ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=3)
    )
    assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.999)


def test_multi_device_single_op_binary(pcie_device_mesh):
    """Multidevice API test: Running tensor-parallel multi-device single-op binary"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor

    torch_input_a_tensor = torch.rand((1, 1, 32, 128), dtype=torch.bfloat16)
    torch_input_b_tensor = torch.rand((1, 1, 32, 128), dtype=torch.bfloat16)
    torch_output_golden = torch_input_a_tensor + torch_input_b_tensor

    ttnn_input_a_tensor = ttnn.from_torch(
        torch_input_a_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=pcie_device_mesh,
        mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=3),
    )
    ttnn_input_b_tensor = ttnn.from_torch(
        torch_input_b_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=pcie_device_mesh,
        mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=3),
    )
    ttnn_output_tensor = ttnn.add(ttnn_input_a_tensor, ttnn_input_b_tensor)

    ttnn_torch_output_tensor = ttnn.to_torch(
        ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=3)
    )
    assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.999)


def test_multi_device_multi_op(pcie_device_mesh):
    """Multidevice API test: Running tensor-parallel multi-device multi-op"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor

    torch_input_tensor = torch.rand((1, 1, 32, 128), dtype=torch.bfloat16)
    torch_output_golden = torch.nn.functional.gelu(torch_input_tensor)
    torch_output_golden = torch.exp(torch_output_golden)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=3),
        device=pcie_device_mesh,
    )
    ttnn_gelu_output = ttnn.gelu(ttnn_input_tensor)
    ttnn_output_tensor = ttnn.exp(ttnn_gelu_output)

    ttnn_torch_output_tensor = ttnn.to_torch(
        ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=3)
    )
    assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.999)


def test_multi_device_data_parallel_matmul_op(pcie_device_mesh):
    """Multidevice API: Data Parallel on matmul"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor, ReplicateTensorToMesh

    torch_input_a_tensor = torch.rand((4, 1, 32, 128), dtype=torch.bfloat16)
    torch_input_b_tensor = torch.rand((1, 1, 128, 32), dtype=torch.bfloat16)
    torch_output_golden = torch_input_a_tensor @ torch_input_b_tensor

    ttnn_input_a_tensor = ttnn.from_torch(
        torch_input_a_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=pcie_device_mesh,
        mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=0),
    )
    ttnn_input_b_tensor = ttnn.from_torch(
        torch_input_b_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=pcie_device_mesh,
        mesh_mapper=ReplicateTensorToMesh(pcie_device_mesh),
    )
    ttnn_output_tensor = ttnn_input_a_tensor @ ttnn_input_b_tensor

    ttnn_torch_output_tensor = ttnn.to_torch(
        ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=0)
    )
    assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.997)


def test_multi_device_data_parallel_matmul_op(pcie_device_mesh):
    """Multidevice API: Data Parallel on matmul"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor, ReplicateTensorToMesh

    torch_input_a_tensor = torch.rand((4, 1, 32, 128), dtype=torch.bfloat16)
    torch_input_b_tensor = torch.rand((1, 1, 128, 32), dtype=torch.bfloat16)
    torch_output_golden = torch_input_a_tensor @ torch_input_b_tensor

    ttnn_input_a_tensor = ttnn.from_torch(
        torch_input_a_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=pcie_device_mesh,
        mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=0),
    )
    ttnn_input_b_tensor = ttnn.from_torch(
        torch_input_b_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=pcie_device_mesh,
        mesh_mapper=ReplicateTensorToMesh(pcie_device_mesh),
    )
    ttnn_output_tensor = ttnn_input_a_tensor @ ttnn_input_b_tensor

    ttnn_torch_output_tensor = ttnn.to_torch(
        ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=0)
    )
    assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.997)
