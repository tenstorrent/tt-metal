# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tests for ttnn.experimental.sub_device.

These tests do not read back which cores an operation used. They rely on the dispatch check instead: inside a scope,
a program that is not on the scoped sub-device is rejected, so an operation that completes ran in the right place.
The check itself is tested with hand-placed programs in distributed_unit_tests (SubDeviceScope*).

Every layout here splits the grid in two: the last column, and all the columns before it.
"""

import threading
from contextlib import contextmanager

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_py_test_utils_common.utility_functions import skip_for_slow_dispatch

SIDE = 32 * 32  # big enough that an operation given the whole grid uses every core


def core_range(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


def grid_of(device):
    grid = device.compute_with_storage_grid_size()
    return grid.x, grid.y


@contextmanager
def sub_devices(device, *cores):
    manager = device.create_sub_device_manager([ttnn.SubDevice([crs]) for crs in cores], 0)
    device.load_sub_device_manager(manager)
    try:
        yield
    finally:
        ttnn.synchronize_device(device)
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)


def compute_then_last_column(device):
    x, y = grid_of(device)
    return sub_devices(device, core_range(0, 0, x - 2, y - 1), core_range(x - 1, 0, x - 1, y - 1))


def last_column_then_compute(device):
    x, y = grid_of(device)
    return sub_devices(device, core_range(x - 1, 0, x - 1, y - 1), core_range(0, 0, x - 2, y - 1))


def to_device(torch_tensor, device):
    mesh_mapper = ttnn.ReplicateTensorToMesh(device) if device.get_num_devices() > 1 else None
    return ttnn.from_torch(
        torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper
    )


def to_torch(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).float()


def inputs(device):
    torch_a = torch.randn(1, 1, SIDE, SIDE, dtype=torch.bfloat16)
    torch_b = torch.randn(1, 1, SIDE, SIDE, dtype=torch.bfloat16)
    return torch_a, torch_b, to_device(torch_a, device), to_device(torch_b, device)


# These operations pick their cores from compute_with_storage_grid_size().
GRID_SIZED = {
    "matmul": (lambda a, b: ttnn.matmul(a, b), lambda a, b: a.float() @ b.float()),
    "softmax": (lambda a, b: ttnn.softmax(a, dim=-1), lambda a, b: torch.softmax(a.float(), dim=-1)),
    "transpose": (lambda a, b: ttnn.transpose(a, -2, -1), lambda a, b: a.float().transpose(-2, -1)),
    "concat": (lambda a, b: ttnn.concat([a, b], dim=-1), lambda a, b: torch.cat([a.float(), b.float()], dim=-1)),
}

# These operations run on the cores of the device's first sub-device.
FIRST_SUB_DEVICE = {
    "add": (lambda a, b: ttnn.add(a, b), lambda a, b: a.float() + b.float()),
    "multiply": (lambda a, b: ttnn.multiply(a, b), lambda a, b: a.float() * b.float()),
    "relu": (lambda a, b: ttnn.relu(a), lambda a, b: torch.relu(a.float())),
}

OPERATIONS = GRID_SIZED | FIRST_SUB_DEVICE


def check_operations_run_on_the_scoped_sub_device(device, layout, sub_device_id):
    torch_a, torch_b, a, b = inputs(device)
    with layout(device):
        for op, reference in OPERATIONS.values():
            with ttnn.experimental.sub_device(sub_device_id):
                output = op(a, b)
            assert_with_pcc(reference(torch_a, torch_b), to_torch(output), 0.999)


@skip_for_slow_dispatch()
def test_scope_changes_what_the_device_reports(device, expect_error):
    x, y = grid_of(device)
    assert ttnn._ttnn.core.get_current_sub_device_id_for_thread() is None

    # With no manager loaded, sub-device 0 is the whole grid, so the scope changes nothing.
    with ttnn.experimental.sub_device(0):
        assert grid_of(device) == (x, y)

    with compute_then_last_column(device):
        assert grid_of(device) == (x, y)
        with ttnn.experimental.sub_device(0):
            assert grid_of(device) == (x - 1, y)
            # The last column does not start at (0, 0), so the grid that covers it is the whole grid.
            with ttnn.experimental.sub_device(ttnn.SubDeviceId(1)):
                assert grid_of(device) == (x, y)
            assert grid_of(device) == (x - 1, y)
        assert grid_of(device) == (x, y)

        with pytest.raises(ValueError, match="leaves the scope"):  # allow-pytest.raises: raised by the test itself
            with ttnn.experimental.sub_device(0):
                raise ValueError("leaves the scope")
        assert ttnn._ttnn.core.get_current_sub_device_id_for_thread() is None
        assert grid_of(device) == (x, y)

        # The manager has no sub-device 2, so the first call that needs it fails.
        with ttnn.experimental.sub_device(2):
            with expect_error(
                RuntimeError, "is for sub-device 2, but the loaded sub-device manager only has 2 sub-device"
            ):
                grid_of(device)


@skip_for_slow_dispatch()
def test_scope_is_per_thread(device):
    x, y = grid_of(device)
    seen = {}
    with compute_then_last_column(device):
        with ttnn.experimental.sub_device(0):
            thread = threading.Thread(target=lambda: seen.update(grid=grid_of(device)))
            thread.start()
            thread.join()
            assert grid_of(device) == (x - 1, y)
    assert seen["grid"] == (x, y)


@skip_for_slow_dispatch()
@pytest.mark.parametrize(
    "layout, sub_device_id",
    [(compute_then_last_column, 0), (last_column_then_compute, 1)],
    ids=["first_sub_device", "second_sub_device"],
)
def test_operations_run_on_the_scoped_sub_device(device, layout, sub_device_id):
    check_operations_run_on_the_scoped_sub_device(device, layout, sub_device_id)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_operations_run_on_the_scoped_sub_device_of_a_mesh(mesh_device):
    check_operations_run_on_the_scoped_sub_device(mesh_device, compute_then_last_column, 0)


@skip_for_slow_dispatch()
def test_scope_for_cores_off_the_origin(device, expect_error):
    torch_a, torch_b, a, b = inputs(device)
    with compute_then_last_column(device):
        with ttnn.experimental.sub_device(1):
            # These operations use the first sub-device, which inside the scope is the last column.
            for op, reference in FIRST_SUB_DEVICE.values():
                assert_with_pcc(reference(torch_a, torch_b), to_torch(op(a, b)), 0.999)
            # Matmul sizes itself from the grid, which here covers the whole device, so it ends up on sub-device 0.
            with expect_error(RuntimeError, "cannot run inside the scope for sub-device 1") as refusal:
                ttnn.matmul(a, b)
            # The error should say which program it was and where its kernels are.
            assert "(kernels: " in str(refusal.value) and "belong to sub-device(s) 0" in str(refusal.value)
        ttnn.synchronize_device(device)


@skip_for_slow_dispatch()
def test_a_program_on_another_sub_device_is_refused(device, expect_error):
    torch_a, torch_b, a, b = inputs(device)
    with compute_then_last_column(device):
        with ttnn.experimental.sub_device(0):
            with expect_error(RuntimeError, "cannot run inside the scope for sub-device 0"):
                ttnn.add(a, b, sub_device_id=ttnn.SubDeviceId(1))
            output = ttnn.add(a, b)
        assert_with_pcc(torch_a.float() + torch_b.float(), to_torch(output), 0.999)


@skip_for_slow_dispatch()
def test_program_cache_is_keyed_by_the_scoped_cores(device):
    torch_a, torch_b, a, b = inputs(device)
    x, y = grid_of(device)

    def run(op, reference):
        before = device.num_program_cache_entries()
        output = op(a, b)
        built = device.num_program_cache_entries() - before
        assert_with_pcc(reference(torch_a, torch_b), to_torch(output), 0.999)
        return built

    built_for_the_whole_grid = {name: run(*GRID_SIZED[name]) for name in GRID_SIZED}
    with ttnn.experimental.sub_device(0):
        for name in GRID_SIZED:
            assert run(*GRID_SIZED[name]) == 0, "sub-device 0 of an unsplit device is the whole grid"

    # The program cached for the whole grid must not be reused here. A new one is built, and only once.
    with compute_then_last_column(device):
        with ttnn.experimental.sub_device(0):
            for name in GRID_SIZED:
                assert run(*GRID_SIZED[name]) == built_for_the_whole_grid[name]
                assert run(*GRID_SIZED[name]) == 0

    # Splitting the grid differently needs new programs again.
    with sub_devices(device, core_range(0, 0, x - 3, y - 1), core_range(x - 2, 0, x - 1, y - 1)):
        with ttnn.experimental.sub_device(0):
            for name in GRID_SIZED:
                assert run(*GRID_SIZED[name]) == built_for_the_whole_grid[name]


@skip_for_slow_dispatch()
def test_metal_sizes_its_own_state_from_the_device_grid(device):
    x, y = grid_of(device)
    with sub_devices(device, core_range(0, 0, 1, 1), core_range(2, 0, x - 1, y - 1)):
        with ttnn.experimental.sub_device(0):
            assert grid_of(device) == (2, 2)
            # The scope covers a 2x2 corner, but creating a manager for the whole grid must still work.
            manager = device.create_sub_device_manager([ttnn.SubDevice([core_range(0, 0, x - 1, y - 1)])], 0)
            ttnn.synchronize_device(device)
            device.remove_sub_device_manager(manager)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("device_params", [{"num_command_queues": 2}], indirect=True)
def test_two_command_queues_on_two_sub_devices(device):
    # Each queue drives one sub-device and the host never waits in between, so both sub-devices are busy at once.
    torch_a, torch_b, a, b = inputs(device)
    outputs = []
    with compute_then_last_column(device):
        for _ in range(4):
            with ttnn.command_queue(0), ttnn.experimental.sub_device(0):
                outputs.append((ttnn.matmul(a, b), torch_a.float() @ torch_b.float()))
            with ttnn.command_queue(1), ttnn.experimental.sub_device(1):
                outputs.append((ttnn.add(a, b), torch_a.float() + torch_b.float()))
        ttnn.synchronize_device(device)
        for output, expected in outputs:
            assert_with_pcc(expected, to_torch(output), 0.999)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}], indirect=True)
def test_a_trace_captured_inside_a_scope_replays(device):
    shape = (1, 1, 64, 128)
    a = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

    def write(torch_input):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT), a)

    with compute_then_last_column(device):
        with ttnn.experimental.sub_device(0):
            write(torch.randn(shape, dtype=torch.bfloat16))
            ttnn.softmax(a, dim=-1)
            trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            output = ttnn.softmax(a, dim=-1)
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
        try:
            # The trace remembers where its programs run, so it can be replayed after the scope is closed.
            for _ in range(2):
                torch_input = torch.randn(shape, dtype=torch.bfloat16)
                write(torch_input)
                ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
                assert_with_pcc(torch.softmax(torch_input.float(), dim=-1), to_torch(output), 0.999)
        finally:
            ttnn.release_trace(device, trace_id)


@skip_for_slow_dispatch()
def test_a_command_list_recorded_inside_a_scope(device, expect_error):
    shape = (1, 1, 64, 128)
    x, y = grid_of(device)
    a = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    with compute_then_last_column(device):
        builder = ttnn.experimental.CommandListBuilder(device)
        command_list = None
        try:
            with ttnn.experimental.sub_device(0):
                # Recording a program checks the scope, just like enqueueing one.
                with expect_error(RuntimeError, "cannot run inside the scope for sub-device 0"):
                    with builder:
                        ttnn.relu(a, sub_core_grids=core_range(x - 1, 0, x - 1, y - 1))
                builder.clear()
                with builder:
                    output = ttnn.relu(a)
            command_list = builder.build(cq_id=0)
            torch_input = torch.randn(shape, dtype=torch.bfloat16)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT), a)
            command_list.enqueue(blocking=True)
            assert_with_pcc(torch.relu(torch_input.float()), to_torch(output), 0.999)
        finally:
            if command_list is not None:
                command_list.deallocate()
            builder.deallocate()


@skip_for_slow_dispatch()
def test_a_device_opens_and_closes_inside_a_scope(request, expect_error):
    with ttnn.experimental.sub_device(1):
        device = ttnn.open_device(device_id=request.config.getoption("device_id"))
        try:
            # No manager is loaded yet, so there is no sub-device 1.
            with expect_error(
                RuntimeError, "is for sub-device 1, but the loaded sub-device manager only has 1 sub-device"
            ):
                device.compute_with_storage_grid_size()
            with ttnn.experimental.sub_device(0):
                x, y = grid_of(device)
            # Once a manager is loaded, the scope applies to that manager's sub-device 1.
            with sub_devices(device, core_range(x - 1, 0, x - 1, y - 1), core_range(0, 0, x - 2, y - 1)):
                assert grid_of(device) == (x - 1, y)
                torch_a = torch.randn(1, 1, SIDE, SIDE, dtype=torch.bfloat16)
                output = ttnn.transpose(to_device(torch_a, device), -2, -1)
                assert_with_pcc(torch_a.float().transpose(-2, -1), to_torch(output), 0.999)
        finally:
            ttnn.close_device(device)
