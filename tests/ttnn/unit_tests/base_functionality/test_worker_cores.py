# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.sub_device / ttnn.worker_cores: running operations on part of a device's Tensix grid."""

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


def sub_devices_ran_on(fn):
    """Runs `fn` under graph capture and returns its result and the sub-devices its programs ran on."""
    result, placements = program_placements(fn)
    return result, set(placements)


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
def test_typecast_runs_on_the_scoped_sub_device(device, sub_device_id):
    torch_input = large_input(device)
    x = to_device(torch_input, device)
    with compute_and_last_column(device):
        with ttnn.sub_device(sub_device_id):
            y, sub_devices = sub_devices_ran_on(lambda: ttnn.typecast(x, ttnn.float32))
        assert sub_devices == {sub_device_id}
        assert torch.equal(ttnn.to_torch(y), torch_input.float())


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


@skip_for_slow_dispatch()
def test_sub_core_grids_are_honoured_for_a_sharded_input(device):
    """typecast used to drop sub_core_grids for a sharded input it could not cast in place, and use the whole grid."""
    x_cores, _ = grid_of(device)
    shard_grid = core_range(0, 0, 3, 0)
    sharded = ttnn.create_sharded_memory_config(
        (32 * 8, 32 * 4),
        shard_grid,
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    torch_input = torch.randn(1, 1, 32 * 32, 32 * 4, dtype=torch.float32)
    # float32 to bfloat16 changes the tile size, so the in-place sharded factory cannot take it.
    x = to_device(torch_input, device, dtype=ttnn.float32, memory_config=sharded)
    with compute_and_last_column(device) as (_, last_column):
        y, sub_devices = sub_devices_ran_on(lambda: ttnn.typecast(x, ttnn.bfloat16, sub_core_grids=last_column))
        assert sub_devices == {1}
        assert torch.equal(ttnn.to_torch(y), torch_input.bfloat16())


def test_a_factory_without_worker_cores_raises_inside_a_scope(device, expect_error):
    x = to_device(large_input(device), device)
    x_cores, y_cores = grid_of(device)
    with ttnn.worker_cores(core_range(0, 0, x_cores - 2, y_cores - 1)):
        # softmax's factories do not take worker cores (yet), so it must refuse rather than use the whole grid.
        with expect_error(RuntimeError, "does not take worker cores"):
            ttnn.softmax(x, dim=-1)
    # Outside the scope it runs as before.
    ttnn.softmax(x, dim=-1)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("sub_device_id", [0, 1])
@pytest.mark.parametrize(
    "name, op, reference",
    [
        ("exp", lambda x: ttnn.exp(x), lambda t: torch.exp(t)),
        ("add", lambda x: ttnn.add(x, x), lambda t: t + t),
        ("multiply", lambda x: ttnn.multiply(x, 3.0), lambda t: t * 3.0),
    ],
)
def test_eltwise_runs_on_the_scoped_sub_device(device, sub_device_id, name, op, reference):
    torch_input = large_input(device)
    x = to_device(torch_input, device)
    with compute_and_last_column(device):
        with ttnn.sub_device(sub_device_id):
            y, sub_devices = sub_devices_ran_on(lambda: op(x))
        assert sub_devices == {sub_device_id}
        assert_with_pcc(reference(torch_input.float()), ttnn.to_torch(y).float(), 0.999)


@skip_for_slow_dispatch()
@pytest.mark.parametrize(
    "name, op, reference",
    [
        ("exp", lambda x: ttnn.exp(x), lambda t: torch.exp(t)),
        ("add", lambda x: ttnn.add(x, x), lambda t: t + t),
    ],
)
def test_a_cache_hit_inside_a_scope_reruns_on_the_same_cores(device, name, op, reference):
    """On a cache hit unary and binary_ng rebuild each core's runtime arguments, on the cores the program used."""
    first, second = large_input(device), large_input(device)
    x, y = to_device(first, device), to_device(second, device)
    with compute_and_last_column(device):
        with ttnn.sub_device(1):
            entries = device.num_program_cache_entries()
            out_x, placements_x = program_placements(lambda: op(x))
            out_y, placements_y = program_placements(lambda: op(y))
            assert device.num_program_cache_entries() == entries + 1
        assert placements_x == placements_y == [1]
        assert_with_pcc(reference(first.float()), ttnn.to_torch(out_x).float(), 0.999)
        assert_with_pcc(reference(second.float()), ttnn.to_torch(out_y).float(), 0.999)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("sub_device_id", [0, 1])
def test_matmul_runs_on_the_scoped_sub_device(device, sub_device_id):
    """Sub-device 1 is the last column: the matmul grid has to move off (0, 0) to its corner."""
    torch_a = torch.randn(1, 1, 512, 1024, dtype=torch.bfloat16)
    torch_b = torch.randn(1, 1, 1024, 512, dtype=torch.bfloat16)
    a, b = to_device(torch_a, device), to_device(torch_b, device)
    with compute_and_last_column(device):
        with ttnn.sub_device(sub_device_id):
            y, sub_devices = sub_devices_ran_on(lambda: ttnn.matmul(a, b))
        assert sub_devices == {sub_device_id}
        assert_with_pcc(torch_a.float() @ torch_b.float(), ttnn.to_torch(y).float(), 0.999)


@skip_for_slow_dispatch()
def test_matmul_with_the_whole_grid_as_core_grid_runs_on_the_scope(device):
    """tt-train passes core_grid=the device grid. A grid larger than the scope falls back to the scope's, as one larger
    than the device always fell back to the device's."""
    torch_a = torch.randn(1, 1, 512, 1024, dtype=torch.bfloat16)
    torch_b = torch.randn(1, 1, 1024, 512, dtype=torch.bfloat16)
    a, b = to_device(torch_a, device), to_device(torch_b, device)
    x_cores, y_cores = grid_of(device)
    with compute_and_last_column(device):
        with ttnn.sub_device(0):
            y, sub_devices = sub_devices_ran_on(
                lambda: ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=y_cores, x=x_cores))
            )
        assert sub_devices == {0}
        assert_with_pcc(torch_a.float() @ torch_b.float(), ttnn.to_torch(y).float(), 0.999)


def test_matmul_program_config_outside_the_scope_raises(device, expect_error):
    x_cores, y_cores = grid_of(device)
    a = to_device(torch.randn(1, 1, 32 * y_cores, 64 * x_cores), device)
    b = to_device(torch.randn(1, 1, 64 * x_cores, 32 * x_cores), device)
    whole_grid_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(x_cores, y_cores),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        transpose_mcast=False,
        fused_activation=None,
    )
    with ttnn.worker_cores(core_range(0, 0, x_cores - 2, y_cores - 1)):
        # One output block per core of the whole grid: the program needs the last column too.
        with expect_error(RuntimeError, "outside the worker-cores scope"):
            ttnn.matmul(a, b, program_config=whole_grid_config)


def test_the_scope_is_part_of_the_program_cache_key(device):
    torch_input = large_input(device)
    x = to_device(torch_input, device)
    x_cores, y_cores = grid_of(device)
    scopes = [None, core_range(0, 0, x_cores - 2, y_cores - 1), core_range(x_cores - 1, 0, x_cores - 1, y_cores - 1)]

    def typecast_in(cores):
        if cores is None:
            return ttnn.typecast(x, ttnn.float32)
        with ttnn.worker_cores(cores):
            return ttnn.typecast(x, ttnn.float32)

    entries = device.num_program_cache_entries()
    for _ in range(2):
        for cores in scopes:
            assert torch.equal(ttnn.to_torch(typecast_in(cores)), torch_input.float())
    assert device.num_program_cache_entries() == entries + len(scopes)


def test_a_sharded_input_outside_the_scope_raises(device, expect_error):
    x_cores, y_cores = grid_of(device)
    sharded = ttnn.create_sharded_memory_config(
        (32 * 8, 32 * 4),
        core_range(0, 0, 3, 0),
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    x = to_device(torch.randn(1, 1, 32 * 32, 32 * 4), device, memory_config=sharded)
    with ttnn.worker_cores(core_range(x_cores - 1, 0, x_cores - 1, y_cores - 1)):
        # bfloat16 to bfloat16 keeps the tile size: the in-place sharded factory, which can only use the shard grid.
        with expect_error(RuntimeError, "the input is sharded on .* outside the worker cores"):
            ttnn.typecast(x, ttnn.bfloat16)


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
