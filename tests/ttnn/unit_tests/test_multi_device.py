# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import tt_lib as ttl
import ttnn
from loguru import logger


#######
# Test MultiDevice Initialization, Open/Close
#######
def test_device_mesh_open_close_explicit():
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


def test_multi_device_open_close_using_context_manager():
    """Using context manager to open and close multi-device"""
    device_grid, device_ids = ttnn.DeviceGrid(2, 2), ttnn.get_device_ids()
    with ttnn.create_device_mesh(device_grid, device_ids) as device_mesh:
        # Do something with multi_device
        pass


#######
# Simple Multi-Device Tensor tests
#######


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_ttnn_to_and_from_multi_device_shard(layout, memory_config):
    """MultiDevice APIs: APIs to map tensors onto device-mesh and loopback"""
    from ttnn import ShardTensorToMeshMapper, ConcatMeshToTensorComposer

    torch_tensor = torch.rand((1, 1, 32, 256), dtype=torch.bfloat16)

    with ttnn.create_device_mesh(ttnn.DeviceGrid(1, 4), ttnn.get_pcie_device_ids()) as device_mesh:
        ttnn_tensor = ttnn.from_torch(torch_tensor, mesh_mapper=ShardTensorToMeshMapper(device_mesh, dim=3))
        ttnn_tensor = ttnn.to_layout(ttnn_tensor, layout=layout)
        ttnn_tensor = ttnn.to_device(ttnn_tensor, device_mesh, memory_config=memory_config)
        ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
        torch_loop_back_tensor = ttnn.to_torch(
            ttnn_loop_back_tensor, mesh_composer=ConcatMeshToTensorComposer(device_mesh, dim=3)
        )

        assert torch.all(torch_tensor == torch_loop_back_tensor)


def test_multi_device_replicate(pcie_device_mesh):
    """MultiDevice APIs: APIs to map tensors onto device-mesh and loopback"""
    from ttnn import ReplicateTensorToMeshMapper, ListMeshToTensorComposer

    full_tensor = torch.rand((1, 1, 32, 128), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(full_tensor, mesh_mapper=ReplicateTensorToMeshMapper(pcie_device_mesh))
    ttnn_tensor = ttnn.to_device(ttnn_tensor, pcie_device_mesh)
    ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
    loopback_replicated_tensors = ttnn.to_torch(
        ttnn_loop_back_tensor, mesh_composer=ListMeshToTensorComposer(pcie_device_mesh)
    )
    for loopback_replicated_tensor in loopback_replicated_tensors:
        assert torch.all(full_tensor == loopback_replicated_tensor)


def test_ttnn_multi_device_all_gather(pcie_device_mesh):
    """MultiDevice APIs: APIs to map tensors onto device-mesh and loopback"""
    from ttnn import ShardTensorToMeshMapper

    full_tensor = torch.rand((1, 1, 32, 128), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(full_tensor, mesh_mapper=ShardTensorToMeshMapper(pcie_device_mesh, dim=3))
    ttnn_tensor = ttnn.to_device(ttnn_tensor, pcie_device_mesh)
    ttnn_tensor = ttnn.all_gather(ttnn_tensor, dim=3, num_links=1)

    device_tensors: typing.List[ttnn.Tensor] = ttnn.get_device_tensors(ttnn_tensor)
    for device_tensor in device_tensors:
        device_tensor_torch = ttnn.to_torch(device_tensor)
        assert torch.all(device_tensor_torch == full_tensor)
