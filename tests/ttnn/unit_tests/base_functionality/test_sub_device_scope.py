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


OPS = [
    ("typecast", lambda x: ttnn.typecast(x, ttnn.float32), lambda t: t.float()),
    ("exp", lambda x: ttnn.exp(x), lambda t: torch.exp(t.float())),
    ("add", lambda x: ttnn.add(x, x), lambda t: t.float() + t.float()),
    ("multiply_scalar", lambda x: ttnn.multiply(x, 3.0), lambda t: t.float() * 3.0),
    ("composite_example", lambda x: ttnn.composite_example(x), lambda t: t.float()),
    ("matmul", lambda x: ttnn.matmul(x, x), lambda t: t.float() @ t.float()),
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


@skip_for_slow_dispatch()
def test_sub_core_grids_narrow_placement_within_the_scope(device):
    torch_input = large_input()
    x = to_device(torch_input, device)
    x_cores, y_cores = grid_of(device)
    with compute_and_last_column(device):
        with ttnn.sub_device(1):
            two_cores = core_range(x_cores - 1, 0, x_cores - 1, 1)
            y, sub_devices_used = sub_devices_ran_on(lambda: ttnn.typecast(x, ttnn.float32, sub_core_grids=two_cores))
        assert sub_devices_used == {1}
        assert torch.equal(ttnn.to_torch(y), torch_input.float())


@skip_for_slow_dispatch()
def test_sub_core_grids_outside_the_scope_raise(device, expect_error):
    x = to_device(large_input(), device)
    with compute_and_last_column(device) as (compute, _):
        with ttnn.sub_device(1):
            with expect_error(RuntimeError, "outside sub-device 1"):
                ttnn.typecast(x, ttnn.float32, sub_core_grids=compute)


@skip_for_slow_dispatch()
def test_an_explicit_sub_device_must_agree_with_the_scope(device, expect_error):
    torch_input = large_input()
    x = to_device(torch_input, device)
    with compute_and_last_column(device):
        with ttnn.sub_device(1):
            with expect_error(
                RuntimeError, "was asked to run on sub-device 0 inside a sub-device scope on sub-device 1"
            ):
                ttnn.add(x, x, sub_device_id=ttnn.SubDeviceId(0))
            y, sub_devices_used = sub_devices_ran_on(lambda: ttnn.add(x, x, sub_device_id=ttnn.SubDeviceId(1)))
        assert sub_devices_used == {1}
        assert_with_pcc(torch_input.float() * 2, ttnn.to_torch(y).float(), 0.999)


@skip_for_slow_dispatch()
def test_the_scope_is_part_of_the_program_cache_key(device):
    torch_input = large_input()
    x = to_device(torch_input, device)

    def typecast_on(sub_device_id):
        with ttnn.sub_device(sub_device_id):
            return ttnn.typecast(x, ttnn.float32)

    with compute_and_last_column(device):
        entries = device.num_program_cache_entries()
        for _ in range(2):
            for sub_device_id in (0, 1):
                y, placements = program_placements(lambda: typecast_on(sub_device_id))
                assert placements == [sub_device_id]
                assert torch.equal(ttnn.to_torch(y), torch_input.float())
        assert device.num_program_cache_entries() == entries + 2


CACHED_OPS = [entry for entry in OPS if entry[0] in ("exp", "add", "multiply_scalar")]


@skip_for_slow_dispatch()
@pytest.mark.parametrize("name, op, reference", CACHED_OPS, ids=[name for name, _, _ in CACHED_OPS])
def test_a_cache_hit_inside_a_scope_reruns_on_the_same_cores(device, name, op, reference):
    first, second = large_input(), large_input()
    x, y = to_device(first, device), to_device(second, device)
    with compute_and_last_column(device):
        with ttnn.sub_device(1):
            entries = device.num_program_cache_entries()
            out_x, placements_x = program_placements(lambda: op(x))
            out_y, placements_y = program_placements(lambda: op(y))
            assert device.num_program_cache_entries() == entries + 1
        assert placements_x == placements_y == [1]
        assert_with_pcc(reference(first), ttnn.to_torch(out_x).float(), 0.999)
        assert_with_pcc(reference(second), ttnn.to_torch(out_y).float(), 0.999)


def sharded_on(cores):
    return ttnn.create_sharded_memory_config(
        (32 * 8, 32 * 4),
        cores,
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@skip_for_slow_dispatch()
@pytest.mark.parametrize("sub_device_id", [0, 1])
def test_typecast_of_a_sharded_input_follows_the_scope(device, sub_device_id, expect_error):
    """The input is sharded on sub-device 0's cores. There the in-place sharded factory runs on the shards and its
    output takes the same shards. On sub-device 1 the interleaved factory reads them over the NoC; an output sharded on
    sub-device 0's cores would land among the circular buffers of whatever runs there, so it is sharded on sub-device
    1's. Both factories compute the same values, so the test also checks which one ran."""
    sharded = sharded_on(core_range(0, 0, 3, 0))
    torch_input = torch.randn(1, 1, 32 * 32, 32 * 4, dtype=torch.float32)
    x = to_device(torch_input, device, dtype=ttnn.float32, memory_config=sharded)
    with compute_and_last_column(device):
        with ttnn.sub_device(sub_device_id):
            # float32 to int32 keeps the tile size, which the in-place sharded factory needs.
            if sub_device_id == 1:
                with expect_error(RuntimeError, re.escape("allocates an output in L1 (sharded on")):
                    ttnn.typecast(x, ttnn.int32)
            x_cores, _ = grid_of(device)
            memory_config = sharded if sub_device_id == 0 else sharded_on(core_range(x_cores - 1, 0, x_cores - 1, 3))
            ttnn.graph.begin_graph_capture()
            try:
                y = ttnn.typecast(x, ttnn.int32, memory_config=memory_config)
            finally:
                graph = ttnn.graph.end_graph_capture()
        placements = [node["params"].get("sub_device_id") for node in graph if node["node_type"] == "program_execution"]
        factories = [
            node["params"]["program_factory_type"] for node in graph if "program_factory_type" in (node["params"] or {})
        ]
        assert placements == [sub_device_id]
        expected = "TypecastShardedProgramFactory" if sub_device_id == 0 else "TypecastProgramFactory"
        assert len(factories) == 1 and factories[0].endswith(expected), factories
        assert torch.equal(ttnn.to_torch(y), torch_input.to(torch.int32))


@skip_for_slow_dispatch()
def test_scopes_nest_and_the_innermost_applies(device):
    torch_input = large_input()
    x = to_device(torch_input, device)
    with compute_and_last_column(device):
        with ttnn.sub_device(0):
            _, outer = program_placements(lambda: ttnn.exp(x))
            with ttnn.sub_device(ttnn.SubDeviceId(1)):
                _, inner = program_placements(lambda: ttnn.exp(x))
            _, outer_again = program_placements(lambda: ttnn.exp(x))
    assert (outer, inner, outer_again) == ([0], [1], [0])


def matmul_inputs(device):
    torch_a = torch.randn(1, 1, 512, 1024, dtype=torch.bfloat16)
    torch_b = torch.randn(1, 1, 1024, 512, dtype=torch.bfloat16)
    return torch_a, torch_b, to_device(torch_a, device), to_device(torch_b, device)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("sub_device_id", [0, 1])
def test_matmul_runs_on_the_scoped_sub_device(device, sub_device_id):
    """Sub-device 1 is the last column: the matmul grid has to move off (0, 0) to its corner."""
    torch_a, torch_b, a, b = matmul_inputs(device)
    with compute_and_last_column(device):
        with ttnn.sub_device(sub_device_id):
            y, sub_devices_used = sub_devices_ran_on(lambda: ttnn.matmul(a, b))
        assert sub_devices_used == {sub_device_id}
        assert_with_pcc(torch_a.float() @ torch_b.float(), ttnn.to_torch(y).float(), 0.999)


@skip_for_slow_dispatch()
def test_matmul_with_the_whole_grid_as_core_grid_runs_on_the_scope(device):
    """tt-train passes core_grid=the device grid. A grid larger than the cores the matmul may use falls back to theirs,
    as one larger than the device always fell back to the device's."""
    torch_a, torch_b, a, b = matmul_inputs(device)
    x_cores, y_cores = grid_of(device)
    with compute_and_last_column(device):
        with ttnn.sub_device(0):
            y, sub_devices_used = sub_devices_ran_on(
                lambda: ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=y_cores, x=x_cores))
            )
        assert sub_devices_used == {0}
        assert_with_pcc(torch_a.float() @ torch_b.float(), ttnn.to_torch(y).float(), 0.999)


@skip_for_slow_dispatch()
def test_matmul_with_an_explicit_sub_device_outside_a_scope(device):
    torch_a, torch_b, a, b = matmul_inputs(device)
    with compute_and_last_column(device):
        y, sub_devices_used = sub_devices_ran_on(lambda: ttnn.matmul(a, b, sub_device_id=ttnn.SubDeviceId(1)))
        assert sub_devices_used == {1}
        assert_with_pcc(torch_a.float() @ torch_b.float(), ttnn.to_torch(y).float(), 0.999)


@skip_for_slow_dispatch()
def test_a_matmul_program_config_larger_than_the_scope_raises(device, expect_error):
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
    with compute_and_last_column(device):
        with ttnn.sub_device(0):
            # One output block per core of the whole grid: the program needs the last column too.
            with expect_error(RuntimeError, "outside the matmul's sub-device 0"):
                ttnn.matmul(a, b, program_config=whole_grid_config)


def sharded_on_the_last_column(device):
    x_cores, _ = grid_of(device)
    return sharded_on(core_range(x_cores - 1, 0, x_cores - 1, 3))


SHARDED_OPS = [
    # float32 to int32 keeps the tile size, which typecast's in-place sharded factory needs.
    ("typecast", lambda x, mc: ttnn.typecast(x, ttnn.int32, memory_config=mc), lambda t: t.to(torch.int32)),
    ("exp", lambda x, mc: ttnn.exp(x, memory_config=mc), lambda t: torch.exp(t)),
    ("add", lambda x, mc: ttnn.add(x, x, memory_config=mc), lambda t: t + t),
]


@skip_for_slow_dispatch()
@pytest.mark.parametrize("name, op, reference", SHARDED_OPS, ids=[name for name, _, _ in SHARDED_OPS])
def test_outside_a_scope_ops_run_on_the_sub_device_their_shards_live_on(device, name, op, reference):
    """Here sub-device 1, rather than the first one."""
    sharded = sharded_on_the_last_column(device)
    torch_input = torch.randn(1, 1, 32 * 32, 32 * 4, dtype=torch.float32)
    x = to_device(torch_input, device, dtype=ttnn.float32, memory_config=sharded)
    with compute_and_last_column(device):
        y, sub_devices_used = sub_devices_ran_on(lambda: op(x, sharded))
        assert sub_devices_used == {1}
        assert_with_pcc(reference(torch_input).float(), ttnn.to_torch(y).float(), 0.999)


@skip_for_slow_dispatch()
def test_outside_a_scope_sub_core_grids_pick_the_sub_device(device):
    torch_input = large_input()
    x = to_device(torch_input, device)
    x_cores, _ = grid_of(device)
    with compute_and_last_column(device):
        two_cores = core_range(x_cores - 1, 0, x_cores - 1, 1)
        y, sub_devices_used = sub_devices_ran_on(lambda: ttnn.typecast(x, ttnn.float32, sub_core_grids=two_cores))
        assert sub_devices_used == {1}
        assert torch.equal(ttnn.to_torch(y), torch_input.float())


@skip_for_slow_dispatch()
@pytest.mark.parametrize("name, op, reference", SHARDED_OPS, ids=[name for name, _, _ in SHARDED_OPS])
def test_inside_a_scope_ops_on_another_sub_devices_shards(device, name, op, reference, expect_error):
    """Each op reads the shards on sub-device 1 over the NoC from its own sub-device, rather than running its natively
    sharded path where the shards live. An output sharded there as well is refused: it would land among the circular
    buffers of whatever runs on sub-device 1."""
    sharded = sharded_on_the_last_column(device)
    torch_input = torch.randn(1, 1, 32 * 32, 32 * 4, dtype=torch.float32)
    x = to_device(torch_input, device, dtype=ttnn.float32, memory_config=sharded)
    with compute_and_last_column(device):
        with ttnn.sub_device(0):
            with expect_error(RuntimeError, re.escape("allocates an output in L1 (sharded on")):
                op(x, sharded)
            # Typecast keeps the input's memory layout: its output is sharded on sub-device 0's cores.
            elsewhere = sharded_on(core_range(0, 0, 3, 0)) if name == "typecast" else ttnn.DRAM_MEMORY_CONFIG
            y, sub_devices_used = sub_devices_ran_on(lambda: op(x, elsewhere))
        assert sub_devices_used == {0}
        assert_with_pcc(reference(torch_input).float(), ttnn.to_torch(y).float(), 0.999)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("name, op, reference", SHARDED_OPS[1:], ids=[name for name, _, _ in SHARDED_OPS[1:]])
def test_sub_core_grids_narrower_than_the_shards_read_them_over_the_noc(device, name, op, reference):
    """The natively sharded path runs on the shards' cores, so with sub_core_grids that leave some out it is not taken
    (it used to crash, or fail building its program)."""
    sharded = sharded_on(core_range(0, 0, 3, 0))
    torch_input = torch.randn(1, 1, 32 * 32, 32 * 4, dtype=torch.float32)
    x = to_device(torch_input, device, dtype=ttnn.float32, memory_config=sharded)
    two_cores = core_range(0, 0, 1, 0)
    if name == "exp":
        y = ttnn.exp(x, memory_config=sharded, sub_core_grids=two_cores)
    else:
        y = ttnn.add(x, x, memory_config=sharded, sub_core_grids=two_cores)
    assert_with_pcc(reference(torch_input).float(), ttnn.to_torch(y).float(), 0.999)


@skip_for_slow_dispatch()
def test_a_spec_less_sharded_output_is_placed_on_the_scoped_sub_device(device):
    """An output sharded without a shard spec is spread over the op's sub-device, so the scope can consume it."""
    torch_input = large_input()
    x = to_device(torch_input, device)
    height_sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    with compute_and_last_column(device) as (_, last_column):
        with ttnn.sub_device(1):
            y, placements = program_placements(lambda: ttnn.exp(x, memory_config=height_sharded))
            assert last_column.contains(y.memory_config().shard_spec.grid)
            z, more_placements = program_placements(lambda: ttnn.exp(y))
        assert placements + more_placements == [1, 1]
        assert_with_pcc(torch.exp(torch.exp(torch_input.float())), ttnn.to_torch(z).float(), 0.99)


@skip_for_slow_dispatch()
def test_l1_interleaved_outputs_are_refused_in_a_scope_on_a_split_device(device, expect_error):
    """Interleaved L1 spans every sub-device's cores, where the others keep the circular buffers of programs in
    flight; such an output corrupted a matmul running on the other sub-device."""
    torch_input = large_input()
    x = to_device(torch_input, device)
    with compute_and_last_column(device):
        with ttnn.sub_device(1):
            with expect_error(
                RuntimeError, re.escape("allocates an output in L1 (interleaved) inside a scope on sub-device 1")
            ):
                ttnn.typecast(x, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
            y = ttnn.typecast(x, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        assert torch.equal(ttnn.to_torch(y), torch_input.float())
    with ttnn.sub_device(0):
        # One sub-device: nothing runs beside it.
        ttnn.typecast(x, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)


def width_sharded(cores, height):
    return ttnn.create_sharded_memory_config(
        (height, 32),
        cores,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@skip_for_slow_dispatch()
@pytest.mark.parametrize("scoped", [False, True])
def test_a_matmul_reading_sharded_a_runs_where_its_shards_are(device, scoped):
    """A is sharded on sub-device 1, away from its corner. The matmul runs on sub-device 1 and its grid starts at
    A's first shard, where it reads A in place (it used to move to the sub-device's corner, computing garbage)."""
    torch_a = torch.randn(1, 1, 32, 256, dtype=torch.bfloat16)
    torch_b = torch.randn(1, 1, 256, 256, dtype=torch.bfloat16)
    x_cores, _ = grid_of(device)
    x0 = min(2, x_cores - 8)  # eight cores in row 4, away from sub-device 1's corner where the grid allows
    with first_row_and_the_rest(device):
        a = to_device(torch_a, device, memory_config=width_sharded(core_range(x0, 4, x0 + 7, 4), 32))
        b = to_device(torch_b, device)
        if scoped:
            with ttnn.sub_device(1):
                y, placements = program_placements(lambda: ttnn.matmul(a, b))
        else:
            y, placements = program_placements(lambda: ttnn.matmul(a, b))
        assert placements == [1]
        assert_with_pcc(torch_a.float() @ torch_b.float(), ttnn.to_torch(y).float(), 0.999)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("scoped", [False, True])
def test_a_gather_matmul_runs_its_ring_in_its_shards_sub_device(device, scoped):
    """The gather config runs on the ring of A's shards, here in sub-device 1 away from its corner. It takes its ring
    cores from the matmul's sub-device (it used to take them from the first sub-device, which holds none of them)."""
    torch_a = torch.randn(1, 1, 32, 256, dtype=torch.bfloat16)
    torch_b = torch.randn(1, 1, 256, 256, dtype=torch.bfloat16)
    x_cores, _ = grid_of(device)
    x0 = min(2, x_cores - 8)
    ring = core_range(x0, 3, x0 + 7, 3)

    def on_the_ring(height, width):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(ring, [height, width], ttnn.ShardOrientation.ROW_MAJOR),
        )

    config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 1),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        mcast_in0=False,
        gather_in0=True,
    )
    with first_row_and_the_rest(device):
        a = to_device(torch_a, device, memory_config=on_the_ring(32, 32))
        b = to_device(torch_b, device, memory_config=on_the_ring(256, 32))
        run = lambda: ttnn.matmul(a, b, program_config=config, memory_config=on_the_ring(32, 32))
        if scoped:
            with ttnn.sub_device(1):
                y, placements = program_placements(run)
        else:
            y, placements = program_placements(run)
        assert placements == [1]
        assert_with_pcc(torch_a.float() @ torch_b.float(), ttnn.to_torch(y).float(), 0.999)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("scoped", [False, True])
def test_a_matmul_config_with_allowed_worker_cores_runs_there(device, scoped):
    """The user's allowed_worker_cores are where the kernels and the output go, inside the sub-device or outside a
    scope naming it."""
    torch_a = torch.randn(1, 1, 128, 256, dtype=torch.bfloat16)
    torch_b = torch.randn(1, 1, 256, 128, dtype=torch.bfloat16)
    x_cores, y_cores = grid_of(device)
    x0, y0 = min(5, x_cores - 4), min(4, y_cores - 4)  # a 4x4 block inside sub-device 1 (rows 2 and below)
    allowed = core_range(x0, y0, x0 + 3, y0 + 3)
    config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(4, 4),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        transpose_mcast=False,
        fused_activation=None,
        allowed_worker_cores=allowed,
    )
    block_sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)
    with sub_devices(device, core_range(0, 0, x_cores - 1, 1), core_range(0, 2, x_cores - 1, y_cores - 1)):
        a, b = to_device(torch_a, device), to_device(torch_b, device)
        run = lambda: ttnn.matmul(a, b, program_config=config, memory_config=block_sharded)
        if scoped:
            with ttnn.sub_device(1):
                y, placements = program_placements(run)
        else:
            y, placements = program_placements(run)
        assert placements == [1]
        assert y.memory_config().shard_spec.grid == allowed
        assert_with_pcc(torch_a.float() @ torch_b.float(), ttnn.to_torch(y).float(), 0.999)


@skip_for_slow_dispatch()
def test_a_block_sharded_matmul_off_the_origin_is_refused(device, expect_error):
    """With no sub-devices: the automatic 2D config for A block-sharded away from (0, 0) runs its grid at the origin,
    which does not hold A's shards; it used to compute garbage silently. On the origin it runs."""
    torch_a = torch.randn(1, 1, 512, 512, dtype=torch.bfloat16)
    torch_b = torch.randn(1, 1, 512, 512, dtype=torch.bfloat16)
    b = to_device(torch_b, device)
    for cores, runs in [(core_range(2, 4, 5, 7), False), (core_range(0, 0, 3, 3), True)]:
        block = ttnn.create_sharded_memory_config(
            (128, 128),
            cores,
            ttnn.ShardStrategy.BLOCK,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        a = to_device(torch_a, device, memory_config=block)
        if not runs:
            with expect_error(RuntimeError, "this config reads A's shards in place"):
                ttnn.matmul(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            continue
        y = ttnn.matmul(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        assert_with_pcc(torch_a.float() @ torch_b.float(), ttnn.to_torch(y).float(), 0.999)


@skip_for_slow_dispatch()
@pytest.mark.parametrize(
    "orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR], ids=["row_major", "col_major"]
)
def test_a_2d_matmul_on_block_sharded_a_runs_at_its_sub_devices_corner(device, orientation, expect_error):
    """The automatic 2D config for A block-sharded at sub-device 1's corner is placed there and reads A in place. The
    corner is off the first column for ROW_MAJOR shards, whose blocks are multicast along rows, and off the first row
    for COL_MAJOR ones (along columns): A's senders were numbered by absolute coordinate, so no core sent A's first
    block and every core hung. A sharded one core past the corner is still refused."""
    torch_a = torch.randn(1, 1, 512, 512, dtype=torch.bfloat16)
    torch_b = torch.randn(1, 1, 512, 512, dtype=torch.bfloat16)
    if orientation == ttnn.ShardOrientation.ROW_MAJOR:
        layout, (x0, y0), (dx, dy) = first_column_and_the_rest(device), (1, 0), (1, 0)
    else:
        layout, (x0, y0), (dx, dy) = first_row_and_the_rest(device), (0, 1), (0, 1)

    def block_sharded_from(x, y):
        return ttnn.create_sharded_memory_config(
            (128, 128),
            core_range(x, y, x + 3, y + 3),
            ttnn.ShardStrategy.BLOCK,
            orientation,
            use_height_and_width_as_shard_shape=True,
        )

    with layout:
        a = to_device(torch_a, device, memory_config=block_sharded_from(x0, y0))
        b = to_device(torch_b, device)
        with ttnn.sub_device(1):
            out, placements = program_placements(lambda: ttnn.matmul(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        assert placements == [1]
        assert_with_pcc(torch_a.float() @ torch_b.float(), ttnn.to_torch(out).float(), 0.999)
        past_the_corner = to_device(torch_a, device, memory_config=block_sharded_from(x0 + dx, y0 + dy))
        with ttnn.sub_device(1):
            with expect_error(RuntimeError, "this config reads A's shards in place"):
                ttnn.matmul(past_the_corner, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def width_sharded_on_two_cores_from(x, height):
    return ttnn.create_sharded_memory_config(
        (height, 128),
        core_range(x, 0, x + 1, 0),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@skip_for_slow_dispatch()
@pytest.mark.parametrize("misplaced", [None, "input B", "the bias", "the output"])
def test_a_1d_matmul_reads_b_and_bias_and_writes_its_output_where_its_grid_starts(device, misplaced, expect_error):
    """A 1D config multicasting A reads L1-sharded B and bias, and writes a sharded output, in place on its grid's
    cores. Its grid starts at sub-device 1's corner, (1, 0): with B and the bias there, and an output the matmul places
    itself, it runs; one laid out from the origin instead is refused rather than read or written on cores that do not
    hold it."""
    torch_a = torch.randn(1, 1, 32, 256, dtype=torch.bfloat16)
    torch_b = torch.randn(1, 1, 256, 256, dtype=torch.bfloat16)
    torch_bias = torch.randn(1, 1, 1, 256, dtype=torch.bfloat16)
    config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(2, 1),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=4,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    with first_column_and_the_rest(device):
        a = to_device(torch_a, device, memory_config=width_sharded_on_two_cores_from(1, 32))
        b_x, bias_x = (0 if misplaced == "input B" else 1), (0 if misplaced == "the bias" else 1)
        b = to_device(torch_b, device, memory_config=width_sharded_on_two_cores_from(b_x, 256))
        bias = to_device(torch_bias, device, memory_config=width_sharded_on_two_cores_from(bias_x, 32))
        if misplaced == "the output":
            output = to_device(torch.zeros(1, 1, 32, 256), device, memory_config=width_sharded_on_two_cores_from(0, 32))
            run = lambda: ttnn.linear(a, b, bias=bias, program_config=config, optional_output_tensor=output)
        else:
            run = lambda: ttnn.linear(
                a, b, bias=bias, program_config=config, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
            )
        with ttnn.sub_device(1):
            if misplaced is not None:
                with expect_error(RuntimeError, f"{misplaced} is sharded on"):
                    run()
                return
            out, placements = program_placements(run)
        assert placements == [1]
        assert_with_pcc(torch_a.float() @ torch_b.float() + torch_bias.float(), ttnn.to_torch(out).float(), 0.999)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("misplaced", ["input B", "the output"])
def test_a_2d_matmul_refuses_b_or_an_output_sharded_from_the_origin_off_its_grid(device, misplaced, expect_error):
    """A 2D config reads L1-sharded B on its grid's first row, and writes a sharded output on its grid, in place. Its
    grid starts at sub-device 1's corner, (1, 0): B, or an output the caller allocated, laid out from the origin is
    refused rather than read or written on cores that do not hold it."""
    torch_a = torch.randn(1, 1, 128, 256, dtype=torch.bfloat16)
    torch_b = torch.randn(1, 1, 256, 128, dtype=torch.bfloat16)
    config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(4, 4),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        transpose_mcast=False,
        fused_activation=None,
    )
    with first_column_and_the_rest(device):
        a = to_device(torch_a, device)
        if misplaced == "input B":
            b_on_the_first_row = ttnn.create_sharded_memory_config(
                (256, 32),
                core_range(0, 0, 3, 0),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            b = to_device(torch_b, device, memory_config=b_on_the_first_row)
            output = {"memory_config": ttnn.DRAM_MEMORY_CONFIG}
        else:
            b = to_device(torch_b, device)
            block = ttnn.create_sharded_memory_config(
                (32, 32),
                core_range(0, 0, 3, 3),
                ttnn.ShardStrategy.BLOCK,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            output = {"optional_output_tensor": to_device(torch.zeros(1, 1, 128, 128), device, memory_config=block)}
        with ttnn.sub_device(1):
            with expect_error(RuntimeError, f"{misplaced} is sharded on"):
                ttnn.matmul(a, b, program_config=config, **output)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("transpose", ["a", "b"])
def test_a_matmul_transposing_an_input_never_computes_the_untransposed_product(device, transpose, expect_error):
    """ttnn.matmul and its prim choose the program config on the cores of the same sub-device, so the config that runs
    is the one that transposes A or B inside its program; one that ignores transposes, computing A @ B for A @ B^T, is
    never swapped in. Without a scope on a split device the matmul names no sub-device and is refused."""
    torch_a = torch.randn(1, 1, 1024, 1024, dtype=torch.bfloat16)
    torch_b = torch.randn(1, 1, 1024, 1024, dtype=torch.bfloat16)
    a, b = to_device(torch_a, device), to_device(torch_b, device)
    kwargs = {"transpose_a": True} if transpose == "a" else {"transpose_b": True}
    reference = (
        (torch_a.float().transpose(-1, -2) @ torch_b.float())
        if transpose == "a"
        else (torch_a.float() @ torch_b.float().transpose(-1, -2))
    )
    with compute_and_last_column(device):
        with ttnn.sub_device(1):
            y, placements = program_placements(lambda: ttnn.matmul(a, b, **kwargs))
        assert placements == [1]
        assert_with_pcc(reference, ttnn.to_torch(y).float(), 0.999)
    with first_row_and_the_rest(device):
        with expect_error(RuntimeError, "does not know which of the 2 loaded sub-devices to run on"):
            ttnn.matmul(a, b, **kwargs)


@skip_for_slow_dispatch()
def test_linear_with_a_post_processed_bias_runs_the_add_on_the_matmuls_sub_device(device):
    """A width-1 bias is added after the matmul by ttnn.add. Given only the matmul's sub_device_id it used to run on
    the first sub-device, racing the matmul; now it follows the matmul's."""
    torch_a = torch.randn(1, 1, 1024, 1024, dtype=torch.bfloat16)
    torch_b = torch.randn(1, 1, 1024, 1024, dtype=torch.bfloat16)
    torch_bias = torch.randn(1, 1, 1, 1, dtype=torch.bfloat16)
    a, b, bias = to_device(torch_a, device), to_device(torch_b, device), to_device(torch_bias, device)
    reference = torch_a.float() @ torch_b.float() + torch_bias.float()
    with first_row_and_the_rest(device):
        for _ in range(4):
            y, placements = program_placements(lambda: ttnn.linear(a, b, bias=bias, sub_device_id=ttnn.SubDeviceId(1)))
            assert placements == [1, 1]
            assert_with_pcc(reference, ttnn.to_torch(y).float().reshape(reference.shape), 0.999)


@skip_for_slow_dispatch()
def test_binary_ops_carry_their_sub_device_to_the_typecasts_they_launch(device):
    """A uint8 operand is cast to uint16 first, on the binary op's sub-device."""
    torch_input = torch.randint(0, 100, (1, 1, 64, 64), dtype=torch.uint8)
    x = to_device(torch_input, device, dtype=ttnn.uint8)
    with compute_and_last_column(device):
        y, placements = program_placements(lambda: ttnn.eq(x, x, sub_device_id=ttnn.SubDeviceId(1)))
        assert set(placements) == {1}
        assert len(placements) >= 2
        assert torch.all(ttnn.to_torch(y) == 1)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}], indirect=True)
def test_a_trace_captured_in_a_scope_replays_on_its_sub_device(device, expect_error):
    """A trace is captured under the assignment it was warmed up with: the scope is part of each program's key, so a
    warm-up on another sub-device does not serve it."""
    torch_input = large_input()
    x = to_device(torch_input, device)
    with compute_and_last_column(device):
        with ttnn.sub_device(1):
            ttnn.exp(x)  # compile
            trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            y = ttnn.exp(x)
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        assert_with_pcc(torch.exp(torch_input.float()), ttnn.to_torch(y).float(), 0.999)
        ttnn.release_trace(device, trace_id)

        other = to_device(torch.randn(1, 1, 64, 64, dtype=torch.bfloat16), device)
        with ttnn.sub_device(0):
            ttnn.exp(other)  # warmed up on sub-device 0 only
        device.set_program_cache_misses_allowed(False)
        try:
            with ttnn.sub_device(1):
                with expect_error(RuntimeError, "cache misses are forbidden"):
                    ttnn.exp(other)
        finally:
            device.set_program_cache_misses_allowed(True)


@skip_for_slow_dispatch()
def test_a_2d_matmul_whose_sharded_a_reaches_past_its_grid_runs(device):
    """A is block-sharded over 8x8 cores while the 2D config's grid is 4 wide: the cores past the grid only send A.
    The matmul reads A in place from the grid's first cores, so it runs with no sub-device manager, as it always did."""
    torch_a = torch.randn(1, 1, 256, 1024, dtype=torch.bfloat16)
    torch_b = torch.randn(1, 1, 1024, 128, dtype=torch.bfloat16)
    block = ttnn.create_sharded_memory_config(
        (32, 128),
        core_range(0, 0, 7, 7),
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    a = to_device(torch_a, device, memory_config=block)
    b = to_device(torch_b, device)
    config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(4, 8),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        transpose_mcast=False,
        fused_activation=None,
    )
    y = ttnn.matmul(a, b, program_config=config, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert_with_pcc(torch_a.float() @ torch_b.float(), ttnn.to_torch(y).float(), 0.999)


@skip_for_slow_dispatch()
def test_a_reuse_matmul_on_sharded_a_runs_on_its_shards_sub_device(device):
    """The reuse config runs on its sharded input's cores, whatever grid it was given."""
    x_cores, _ = grid_of(device)
    torch_a = torch.randn(1, 8, 32, 64, dtype=torch.bfloat16)
    torch_b = torch.randn(1, 8, 64, 64, dtype=torch.bfloat16)
    height = ttnn.create_sharded_memory_config(
        (32, 64),
        core_range(0, 1, 7, 1),
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=1,
        per_core_N=2,
    )
    with first_row_and_the_rest(device):
        a = to_device(torch_a, device, memory_config=height)
        b = to_device(torch_b, device)
        y, placements = program_placements(
            lambda: ttnn.matmul(a, b, program_config=config, sub_device_id=ttnn.SubDeviceId(1))
        )
        assert placements == [1]
        assert_with_pcc(torch_a.float() @ torch_b.float(), ttnn.to_torch(y).float(), 0.999)


def test_a_deallocated_or_host_input_raises_as_it_always_did(device, expect_error):
    x = to_device(torch.randn(1, 1, 32, 32, dtype=torch.bfloat16), device)
    host = ttnn.from_torch(torch.randn(1, 1, 32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "must be on device"):
        ttnn.add(host, x)
    ttnn.deallocate(x)
    with expect_error(RuntimeError, "is not allocated"):
        ttnn.add(x, x)
    with expect_error(RuntimeError, "is not allocated"):
        ttnn.typecast(x, ttnn.float32)
