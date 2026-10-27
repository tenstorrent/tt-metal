# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.sub_device / ttnn.worker_cores: running operations on part of a device's Tensix grid."""

from contextlib import contextmanager

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import skip_for_slow_dispatch


def core_range(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


def grid_of(device):
    grid = device.compute_with_storage_grid_size()
    return grid.x, grid.y


@contextmanager
def compute_and_last_column(device):
    """A sub-device manager splitting the grid as FSDP does: sub-device 0 for compute, 1 (the last column) for CCLs."""
    x, y = grid_of(device)
    compute = core_range(0, 0, x - 2, y - 1)
    last_column = core_range(x - 1, 0, x - 1, y - 1)
    manager = device.create_sub_device_manager([ttnn.SubDevice([compute]), ttnn.SubDevice([last_column])], 0)
    device.load_sub_device_manager(manager)
    try:
        yield compute, last_column
    finally:
        ttnn.synchronize_device(device)
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)


def program_placements(fn):
    """Runs `fn` under graph capture; returns its result and the sub-device of each program it ran, in order."""
    ttnn.graph.begin_graph_capture()
    try:
        result = fn()
    finally:
        graph = ttnn.graph.end_graph_capture()
    placements = [node["params"].get("sub_device_id") for node in graph if node["node_type"] == "program_execution"]
    return result, placements


def to_device(torch_tensor, device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )


def large_input(device):
    # Enough tiles that a factory sizing itself from the whole grid uses every core, the last column included.
    return torch.randn(1, 1, 32 * 32, 32 * 32, dtype=torch.bfloat16)


@skip_for_slow_dispatch()
def test_without_a_scope_a_whole_grid_op_spans_both_sub_devices(device, expect_error):
    """The problem the scope solves: once the grid is split, an op that sizes itself from the whole grid cannot run."""
    x = to_device(large_input(device), device)
    with compute_and_last_column(device):
        with expect_error(RuntimeError, "Programs must be executed on a single sub-device"):
            ttnn.typecast(x, ttnn.float32)
            ttnn.synchronize_device(device)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("sub_device_id", [0, 1])
def test_a_composite_op_runs_on_the_scoped_sub_device(device, sub_device_id):
    """composite_example launches two device operations from C++; the scope reaches both."""
    torch_input = large_input(device)
    x = to_device(torch_input, device)
    with compute_and_last_column(device):
        with ttnn.sub_device(sub_device_id):
            y, placements = program_placements(lambda: ttnn.composite_example(x))
        assert placements == [sub_device_id, sub_device_id]
        assert torch.equal(ttnn.to_torch(y), torch_input)


def test_a_factory_without_worker_cores_raises_inside_a_scope(device, expect_error):
    x = to_device(large_input(device), device)
    x_cores, y_cores = grid_of(device)
    with ttnn.worker_cores(core_range(0, 0, x_cores - 2, y_cores - 1)):
        # softmax's factories do not take worker cores (yet), so it must refuse rather than use the whole grid.
        with expect_error(RuntimeError, "does not take worker cores"):
            ttnn.softmax(x, dim=-1)
    # Outside the scope it runs as before.
    ttnn.softmax(x, dim=-1)


def test_get_worker_cores_follows_the_innermost_scope(device):
    x_cores, y_cores = grid_of(device)
    whole = core_range(0, 0, x_cores - 1, y_cores - 1)
    left, right = core_range(0, 0, 0, y_cores - 1), core_range(x_cores - 1, 0, x_cores - 1, y_cores - 1)
    assert ttnn.get_worker_cores(device) == whole
    with ttnn.worker_cores(left):
        assert ttnn.get_worker_cores(device) == left
        with ttnn.worker_cores(right):
            assert ttnn.get_worker_cores(device) == right
        assert ttnn.get_worker_cores(device) == left
    assert ttnn.get_worker_cores(device) == whole


@skip_for_slow_dispatch()
def test_get_worker_cores_resolves_a_sub_device(device):
    with compute_and_last_column(device) as (compute, last_column):
        with ttnn.sub_device(0):
            assert ttnn.get_worker_cores(device) == compute
        with ttnn.sub_device(ttnn.SubDeviceId(1)):
            assert ttnn.get_worker_cores(device) == last_column
