# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import skip_for_slow_dispatch


def run_sub_devices(device):
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


def run_sub_devices_program(device):
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


@skip_for_slow_dispatch()
def test_sub_devices(device):
    run_sub_devices(device)


@skip_for_slow_dispatch()
def test_sub_devices_mesh(mesh_device):
    run_sub_devices(mesh_device)


@skip_for_slow_dispatch()
def test_sub_device_program(device):
    run_sub_devices_program(device)


@skip_for_slow_dispatch()
def test_sub_device_program_mesh(mesh_device):
    run_sub_devices_program(mesh_device)


def run_sub_device_manager_switch_in_trace(mesh_device):
    # A trace that loads a two sub-device manager, runs work on both sub-devices at once, and clears it again, twice,
    # with full-grid ops before, between and after. Every output must match the same ops run without a trace.
    grid = mesh_device.compute_with_storage_grid_size()
    left_cols = grid.x // 2
    left = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(left_cols - 1, grid.y - 1))})
    right = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(left_cols, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([left]), ttnn.SubDevice([right])], 0)

    def sharded(cores):
        return ttnn.create_sharded_memory_config(
            (32, 64), core_grid=cores, strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
        )

    def to_device(shape, memory_config):
        return ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    full_a, full_b = (to_device((1, 1, 256, 256), ttnn.DRAM_MEMORY_CONFIG) for _ in range(2))
    left_a, left_b = (to_device((1, 1, left.num_cores() * 32, 64), sharded(left)) for _ in range(2))
    right_a, right_b = (to_device((1, 1, right.num_cores() * 32, 64), sharded(right)) for _ in range(2))
    outputs = []

    def run():
        outputs.append(ttnn.add(full_a, full_b))
        for window in range(2):
            mesh_device.load_sub_device_manager(manager)
            outputs.append(ttnn.multiply(left_a, left_b, memory_config=sharded(left)))
            outputs.append(ttnn.add(right_a, right_b, memory_config=sharded(right)))
            outputs.append(ttnn.subtract(left_a, left_b, memory_config=sharded(left)))
            mesh_device.clear_loaded_sub_device_manager()
            outputs.append(ttnn.multiply(full_a, full_b) if window == 0 else ttnn.subtract(full_a, full_b))

    def to_torch(tensors):
        return [ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)) for t in tensors]

    run()
    expected = to_torch(outputs)
    outputs.clear()

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    run()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    captured = list(outputs)
    for _ in range(3):
        for t in captured:
            ttnn.copy(ttnn.zeros_like(t), t)
        for _ in range(4):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        for got, want in zip(to_torch(captured), expected):
            assert torch.equal(got, want)

    # The host's dispatch state matches the device's after the replays: eager work in both managers still runs.
    assert torch.equal(to_torch([ttnn.add(full_a, full_b)])[0], expected[0])
    mesh_device.load_sub_device_manager(manager)
    assert torch.equal(to_torch([ttnn.add(right_a, right_b, memory_config=sharded(right))])[0], expected[2])
    mesh_device.clear_loaded_sub_device_manager()

    ttnn.release_trace(mesh_device, trace_id)
    mesh_device.remove_sub_device_manager(manager)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("device_params", [{"trace_region_size": 1 << 20}], indirect=True)
def test_sub_device_manager_switch_in_trace(device):
    run_sub_device_manager_switch_in_trace(device)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("device_params", [{"trace_region_size": 1 << 20}], indirect=True)
def test_sub_device_manager_switch_in_trace_mesh(mesh_device):
    run_sub_device_manager_switch_in_trace(mesh_device)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("device_params", [{"trace_region_size": 1 << 20}], indirect=True)
def test_sub_device_manager_switch_in_trace_left_loaded_mesh(mesh_device, expect_error):
    # A capture that ends with a sub-device manager still loaded switches back to the one it began in, inside the trace.
    # While the trace lives, the manager it switches through cannot be removed.
    grid = mesh_device.compute_with_storage_grid_size()
    left = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x // 2 - 1, grid.y - 1))})
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([left])], 0)
    sharded = ttnn.create_sharded_memory_config(
        (32, 64), core_grid=left, strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
    )
    a, b = (
        ttnn.from_torch(
            torch.randn((1, 1, left.num_cores() * 32, 64), dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=sharded,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for _ in range(2)
    )
    expected = ttnn.to_torch(ttnn.add(a, b), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    # Compile the op under the manager before capture.
    mesh_device.load_sub_device_manager(manager)
    ttnn.add(a, b, memory_config=sharded)
    mesh_device.clear_loaded_sub_device_manager()

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    mesh_device.load_sub_device_manager(manager)
    output = ttnn.add(a, b, memory_config=sharded)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    with expect_error(RuntimeError, "switches through it"):
        mesh_device.remove_sub_device_manager(manager)
    for _ in range(3):
        ttnn.copy(ttnn.zeros_like(output), output)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        got = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        assert torch.equal(got, expected)

    ttnn.release_trace(mesh_device, trace_id)
    mesh_device.remove_sub_device_manager(manager)
