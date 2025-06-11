# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


def run_sub_devices(device, create_fabric_sub_device=False):
    tensix_cores0 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(3, 3),
            ),
        }
    )
    tensix_cores1 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(4, 4),
                ttnn.CoreCoord(4, 4),
            ),
        }
    )
    sub_device_1 = ttnn.SubDevice([tensix_cores0])
    sub_device_2 = ttnn.SubDevice([tensix_cores1])
    sub_devices_1 = [sub_device_1, sub_device_2]
    sub_devices_2 = [sub_device_2]
    if create_fabric_sub_device:
        sub_device_manager1, fabric_sub_device_id1 = device.create_sub_device_manager_with_fabric(sub_devices_1, 3200)
        sub_device_manager2, fabric_sub_device_id2 = device.create_sub_device_manager_with_fabric(sub_devices_2, 3200)
        assert fabric_sub_device_id1 == ttnn.SubDeviceId(len(sub_devices_1))
        assert fabric_sub_device_id2 == ttnn.SubDeviceId(len(sub_devices_2))
    else:
        sub_device_manager1 = device.create_sub_device_manager(sub_devices_1, 3200)
        sub_device_manager2 = device.create_sub_device_manager(sub_devices_2, 3200)
    device.load_sub_device_manager(sub_device_manager1)
    ttnn.synchronize_device(device, sub_device_ids=[ttnn.SubDeviceId(1)])
    ttnn.synchronize_device(device, sub_device_ids=[ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)])
    device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    ttnn.synchronize_device(device)
    device.reset_sub_device_stall_group()
    ttnn.synchronize_device(device)
    device.load_sub_device_manager(sub_device_manager2)
    ttnn.synchronize_device(device, sub_device_ids=[ttnn.SubDeviceId(0)])
    device.clear_loaded_sub_device_manager()
    device.remove_sub_device_manager(sub_device_manager1)
    device.remove_sub_device_manager(sub_device_manager2)


def run_sub_devices_program(device, create_fabric_sub_device=False):
    is_mesh_device = isinstance(device, ttnn.MeshDevice)
    if is_mesh_device:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        num_devices = device.get_num_devices()
    else:
        inputs_mesh_mapper = None
        output_mesh_composer = None
        num_devices = 1
    tensix_cores0 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(4, 4),
                ttnn.CoreCoord(4, 4),
            ),
        }
    )
    tensix_cores1 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(3, 3),
            ),
        }
    )
    sub_device_1 = ttnn.SubDevice([tensix_cores0])
    sub_device_2 = ttnn.SubDevice([tensix_cores1])
    sub_devices = [sub_device_1, sub_device_2]
    if create_fabric_sub_device:
        sub_device_manager, fabric_sub_device_id = device.create_sub_device_manager_with_fabric(sub_devices, 3200)
        assert fabric_sub_device_id == ttnn.SubDeviceId(len(sub_devices))
    else:
        sub_device_manager = device.create_sub_device_manager(sub_devices, 3200)
    device.load_sub_device_manager(sub_device_manager)

    x = torch.randn(num_devices, 1, 64, 64, dtype=torch.bfloat16)
    device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    xt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=inputs_mesh_mapper,
    )
    device.set_sub_device_stall_group([ttnn.SubDeviceId(1)])
    xt_host = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=inputs_mesh_mapper,
    )

    ttnn.copy_host_to_device_tensor(xt_host, xt)

    grid_size = device.compute_with_storage_grid_size()
    shard_size = [32, 64]
    shard_scheme = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    yt = ttnn.interleaved_to_sharded(
        xt, grid_size, shard_size, shard_scheme, shard_orientation, output_dtype=ttnn.bfloat16
    )
    y = ttnn.to_torch(yt, device=device, mesh_composer=output_mesh_composer)

    eq = torch.equal(x, y)
    assert eq
    device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    y = ttnn.to_torch(yt.cpu(), mesh_composer=output_mesh_composer)

    eq = torch.equal(x, y)
    assert eq

    yt2 = ttnn.interleaved_to_sharded(
        xt, grid_size, shard_size, shard_scheme, shard_orientation, output_dtype=ttnn.bfloat16
    )
    event = ttnn.record_event(device, 0, [ttnn.SubDeviceId(1)])
    ttnn.wait_for_event(0, event)
    y2 = ttnn.to_torch(yt2, device=device, mesh_composer=output_mesh_composer)

    eq = torch.equal(x, y2)
    assert eq

    device.clear_loaded_sub_device_manager()
    device.remove_sub_device_manager(sub_device_manager)


@pytest.mark.parametrize("create_fabric_sub_device", (False, True))
def test_sub_devices(device, create_fabric_sub_device):
    run_sub_devices(device, create_fabric_sub_device)


@pytest.mark.parametrize("create_fabric_sub_device", (False, True))
def test_sub_devices_mesh(mesh_device, create_fabric_sub_device):
    run_sub_devices(mesh_device, create_fabric_sub_device)


@pytest.mark.parametrize("create_fabric_sub_device", (False, True))
def test_sub_device_program(device, create_fabric_sub_device):
    run_sub_devices_program(device, create_fabric_sub_device)


@pytest.mark.parametrize("create_fabric_sub_device", (False, True))
def test_sub_device_program_mesh(mesh_device, create_fabric_sub_device):
    run_sub_devices_program(mesh_device, create_fabric_sub_device)
