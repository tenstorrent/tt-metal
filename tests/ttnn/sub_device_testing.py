# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Helpers for testing operations that record the sub-device they run on (ttnn::AssignedSubDevice).

Layouts load a sub-device manager for the duration of a `with` block. Placements are read from graph capture: each
program execution records the sub-device its kernels ran on. The `check_*` functions are the checks every operation
that records its sub-device passes; an operation's own test file parametrizes them over its calls:

    @skip_for_slow_dispatch()
    @pytest.mark.parametrize("sub_device_id", [0, 1])
    def test_my_op_runs_on_the_scoped_sub_device(device, sub_device_id):
        check_runs_on_the_scoped_sub_device(device, sub_device_id, lambda x: ttnn.my_op(x), lambda t: my_reference(t))

Inputs default to large_input(): enough tiles that an operation sizing itself from the whole grid reaches its last
column, so an unmigrated operation cannot pass by coincidence.
"""

from contextlib import contextmanager

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


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


def first_row_and_the_rest(device):
    """Splits the grid as deepseek's MoE does: sub-device 0 is the first row, 1 the rest."""
    x, y = grid_of(device)
    return sub_devices(device, core_range(0, 0, x - 1, 0), core_range(0, 1, x - 1, y - 1))


def first_column_and_the_rest(device):
    """Sub-device 0 is the first column, 1 the rest: its corner, (1, 0), is off the first column."""
    x, y = grid_of(device)
    return sub_devices(device, core_range(0, 0, 0, y - 1), core_range(1, 0, x - 1, y - 1))


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


def check_an_unnamed_op_is_refused_on_a_split_device(device, op, reference, expect_error, torch_input=None, pcc=0.999):
    """With several sub-devices loaded and nothing naming one (no scope, sub-device id, cores or sharded data), an op
    does not pick one: the first is the dispatch strip or the prefetcher's senders as often as the compute grid. With no
    sub-device manager, or one sub-device, it runs as it always did."""
    torch_input = large_input() if torch_input is None else torch_input
    x = to_device(torch_input, device)
    y, sub_devices_used = sub_devices_ran_on(lambda: op(x))
    assert sub_devices_used == {0}
    assert_with_pcc(reference(torch_input), ttnn.to_torch(y).float(), pcc)
    x_cores, y_cores = grid_of(device)
    with sub_devices(device, core_range(0, 0, x_cores - 1, y_cores - 1)):
        y, sub_devices_used = sub_devices_ran_on(lambda: op(x))
        assert sub_devices_used == {0}
        assert_with_pcc(reference(torch_input), ttnn.to_torch(y).float(), pcc)
    with compute_and_last_column(device):
        with expect_error(RuntimeError, "does not know which of the 2 loaded sub-devices to run on"):
            op(x)


def check_runs_on_the_scoped_sub_device(device, sub_device_id, op, reference, torch_input=None, pcc=0.999):
    """Inside `with ttnn.sub_device(i)` every program of the op runs on sub-device i."""
    torch_input = large_input() if torch_input is None else torch_input
    x = to_device(torch_input, device)
    with compute_and_last_column(device):
        with ttnn.sub_device(sub_device_id):
            y, sub_devices_used = sub_devices_ran_on(lambda: op(x))
        assert sub_devices_used == {sub_device_id}
        assert_with_pcc(reference(torch_input), ttnn.to_torch(y).float(), pcc)


def check_a_cache_hit_inside_a_scope_reruns_on_the_same_cores(device, op, reference, pcc=0.999):
    """For an op that runs one program: a second call with other inputs of the same shape hits the program cache and
    runs on the scope's sub-device."""
    first, second = large_input(), large_input()
    x, y = to_device(first, device), to_device(second, device)
    with compute_and_last_column(device):
        with ttnn.sub_device(1):
            entries = device.num_program_cache_entries()
            out_x, placements_x = program_placements(lambda: op(x))
            out_y, placements_y = program_placements(lambda: op(y))
            assert device.num_program_cache_entries() == entries + 1
        assert placements_x == placements_y == [1]
        assert_with_pcc(reference(first), ttnn.to_torch(out_x).float(), pcc)
        assert_with_pcc(reference(second), ttnn.to_torch(out_y).float(), pcc)
