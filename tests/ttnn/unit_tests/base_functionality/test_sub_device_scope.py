# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.sub_device: operations record the sub-device they run on, and launch holds them to it."""

import re
from contextlib import contextmanager

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_py_test_utils_common.utility_functions import skip_for_slow_dispatch


def core_range(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


def grid_of(device):
    grid = device.compute_with_storage_grid_size()
    return grid.x, grid.y


@contextmanager
def sub_devices(device, *cores):
    """Loads a sub-device manager with one sub-device per core range set."""
    manager = device.create_sub_device_manager([ttnn.SubDevice([crs]) for crs in cores], 0)
    device.load_sub_device_manager(manager)
    try:
        yield cores
    finally:
        ttnn.synchronize_device(device)
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)


def compute_and_last_column(device):
    """Splits the grid as FSDP does: sub-device 0 for compute, 1 (the last column) for CCLs."""
    x, y = grid_of(device)
    return sub_devices(device, core_range(0, 0, x - 2, y - 1), core_range(x - 1, 0, x - 1, y - 1))


def program_placements(fn):
    """Runs `fn` under graph capture; returns its result and the sub-device of each program it ran, in order."""
    ttnn.graph.begin_graph_capture()
    try:
        result = fn()
    finally:
        graph = ttnn.graph.end_graph_capture()
    placements = [node["params"].get("sub_device_id") for node in graph if node["node_type"] == "program_execution"]
    return result, placements


def sub_devices_ran_on(fn):
    result, placements = program_placements(fn)
    assert placements, "no program ran"
    return result, set(placements)


def to_device(
    torch_tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    return ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def large_input():
    # Enough tiles that an op sizing itself from the whole grid uses every core, the last column included.
    return torch.randn(1, 1, 32 * 32, 32 * 32, dtype=torch.bfloat16)


OPS = [
    ("composite_example", lambda x: ttnn.composite_example(x), lambda t: t.float()),
]


@skip_for_slow_dispatch()
def test_an_op_sizing_itself_from_the_whole_grid_spans_both_sub_devices(device, expect_error):
    """The problem the contract solves: once the grid is split, an op that sizes itself from the whole grid cannot run.
    softmax does not record its sub-device yet."""
    x = to_device(large_input(), device)
    with compute_and_last_column(device):
        with expect_error(RuntimeError, "Programs must be executed on a single sub-device"):
            ttnn.softmax(x, dim=-1)
            ttnn.synchronize_device(device)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("name, op, reference", OPS, ids=[name for name, _, _ in OPS])
def test_outside_a_scope_an_unnamed_op_is_refused_on_a_split_device(device, name, op, reference, expect_error):
    """With several sub-devices loaded and nothing naming one (no scope, sub-device id, cores or sharded data), an op
    does not pick one: the first is the dispatch strip or the prefetcher's senders as often as the compute grid. With no
    sub-device manager, or one sub-device, it runs as it always did."""
    torch_input = large_input()
    x = to_device(torch_input, device)
    y, sub_devices_used = sub_devices_ran_on(lambda: op(x))
    assert sub_devices_used == {0}
    assert_with_pcc(reference(torch_input), ttnn.to_torch(y).float(), 0.999)
    x_cores, y_cores = grid_of(device)
    with sub_devices(device, core_range(0, 0, x_cores - 1, y_cores - 1)):
        y, sub_devices_used = sub_devices_ran_on(lambda: op(x))
        assert sub_devices_used == {0}
        assert_with_pcc(reference(torch_input), ttnn.to_torch(y).float(), 0.999)
    with compute_and_last_column(device):
        with expect_error(RuntimeError, "does not know which of the 2 loaded sub-devices to run on"):
            op(x)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("sub_device_id", [0, 1])
@pytest.mark.parametrize("name, op, reference", OPS, ids=[name for name, _, _ in OPS])
def test_ops_run_on_the_scoped_sub_device(device, sub_device_id, name, op, reference):
    torch_input = large_input()
    x = to_device(torch_input, device)
    with compute_and_last_column(device):
        with ttnn.sub_device(sub_device_id):
            y, sub_devices_used = sub_devices_ran_on(lambda: op(x))
        assert sub_devices_used == {sub_device_id}
        assert_with_pcc(reference(torch_input), ttnn.to_torch(y).float(), 0.999)


def test_an_op_that_records_no_sub_device_raises_inside_a_scope(device, expect_error):
    x = to_device(large_input(), device)
    with ttnn.sub_device(0):
        # softmax records no sub-device (yet), so it must refuse rather than use the whole grid.
        with expect_error(RuntimeError, "does not record the sub-device it runs on"):
            ttnn.softmax(x, dim=-1)
    # Outside the scope it runs as before.
    ttnn.softmax(x, dim=-1)


@skip_for_slow_dispatch()
def test_a_tensor_created_from_the_host_inside_a_scope_is_converted_on_the_host(device):
    """With no manager loaded, from_torch with a mesh mapper converts on the device, with a tilize that records no
    sub-device and a typecast. Inside a scope it converts on the host, as it does under a manager: tensors created from
    the host are not checked."""
    torch_input = torch.randn(1, 1, 64, 64, dtype=torch.float32)
    with ttnn.sub_device(0):
        x, placements = program_placements(
            lambda: ttnn.from_torch(
                torch_input,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        )
    assert placements == []
    assert torch.equal(ttnn.to_torch(ttnn.get_device_tensors(x)[0]), torch_input.to(torch.bfloat16))
