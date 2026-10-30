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
    ("typecast", lambda x: ttnn.typecast(x, ttnn.float32), lambda t: t.float()),
    ("exp", lambda x: ttnn.exp(x), lambda t: torch.exp(t.float())),
    ("add", lambda x: ttnn.add(x, x), lambda t: t.float() + t.float()),
    ("multiply_scalar", lambda x: ttnn.multiply(x, 3.0), lambda t: t.float() * 3.0),
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
