# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Hand-written regressions for the scatter routing seams the generated sweep in
# test_scatter_codegen_routing.py does not reach: the demotion predicate's axis spelling, small
# row-axis shapes, an output placement that differs from the input's, empty operands, the bf16
# reduce path, the unknown reduce string, the out-of-scope clauses that stay native, the index
# dtypes, an explicit core grid, host-resident operands, an operand padded past tile rounding, the
# calls whose resident sticks or pending copies no longer fit the L1 that is actually free, and
# the program-cache key that follows the L1 frontier.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

_force_native = ttnn._ttnn.operations.data_movement.scatter_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.scatter_force_codegen


_TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}


def _unique_index(index_shape, axis, axis_len):
    # Unique along `axis` and spanning the whole input extent. With a duplicate in a lane the
    # winning source element is whichever the implementation visits last, which neither route
    # promises, so a bit-exact comparison would be asserting walk order rather than the result.
    permuted_shape = list(index_shape)
    permuted_shape[axis] = max(axis_len, index_shape[axis])
    permuted = torch.argsort(torch.rand(permuted_shape), dim=axis) % axis_len
    return permuted.narrow(axis, 0, index_shape[axis]).to(torch.int32).contiguous()


def _make_case(
    shape,
    index_shape,
    dim,
    layout,
    device,
    input_memory_config=None,
    index_dtype=ttnn.int32,
    index_max=None,
    input_dtype=ttnn.bfloat16,
    index_layout=None,
):
    axis = dim if dim >= 0 else len(shape) + dim
    torch_dtype = _TORCH_DTYPE[input_dtype]
    x = torch.rand(shape, dtype=torch_dtype)
    if index_max is None:
        index = _unique_index(index_shape, axis, int(shape[axis]))
    else:
        # Duplicates on purpose: a reduce test needs several updates per output element.
        index = torch.randint(0, int(index_max), index_shape, dtype=torch.int32)
    src = torch.rand(index_shape, dtype=torch_dtype)
    xt = ttnn.from_torch(x, dtype=input_dtype, layout=layout, device=device, memory_config=input_memory_config)
    it = ttnn.from_torch(
        index, dtype=index_dtype, layout=layout if index_layout is None else index_layout, device=device
    )
    st = ttnn.from_torch(src, dtype=input_dtype, layout=layout, device=device)
    return xt, it, st


def _run_auto_against_native(device, xt, it, st, dim, **kwargs):
    """Returns (auto output, program-cache entries added by the auto route beyond what native cached)."""
    device.clear_program_cache()
    golden = ttnn.to_torch(_force_native(xt, dim, it, st, **kwargs))
    entries_before = device.num_program_cache_entries()
    out = ttnn.scatter(xt, dim, it, st, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    return out, device.num_program_cache_entries() - entries_before


@pytest.mark.parametrize("dim", [3, -1])
def test_scatter_codegen_demotion_matches_either_axis_spelling(device, dim):
    # is_demoted() normalizes the caller's raw dim before its unit-row check, so the last axis of a
    # rank-4 tensor demotes the same whether spelled 3 or -1. The L1 output keeps the row-major detour
    # from serving the unit row, which is what makes the case demote at all.
    xt, it, st = _make_case([1, 1, 1, 64], [1, 1, 1, 32], dim, ttnn.TILE_LAYOUT, device)
    _, added = _run_auto_against_native(device, xt, it, st, dim, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert added == 0, f"dim={dim} routed a unit TILE row the detour cannot serve to codegen; expected native"


@pytest.mark.parametrize(
    "shape,index_shape,layout",
    [
        ([1, 1, 32, 64], [1, 1, 16, 64], ttnn.ROW_MAJOR_LAYOUT),
        ([1, 1, 32, 64], [1, 1, 16, 64], ttnn.TILE_LAYOUT),
        ([1, 1, 64, 128], [1, 1, 32, 128], ttnn.ROW_MAJOR_LAYOUT),
    ],
    ids=lambda v: str(v) if isinstance(v, list) else None,
)
def test_scatter_codegen_row_axis_small_shapes_route_to_codegen(device, shape, index_shape, layout):
    # is_demoted() names no exact shape, so small bf16 row-axis scatters route to codegen and must
    # match native.
    xt, it, st = _make_case(shape, index_shape, -2, layout, device)
    _, added = _run_auto_against_native(device, xt, it, st, -2)
    assert added > 0, "auto fell back to native; expected codegen to serve this call"


def test_scatter_codegen_tile_dram_input_l1_output_runs_tile_factory(device):
    # Ht == 1 qualifies for the row-major detour on shape alone, but the detour's ROW_MAJOR factories
    # cannot write an output whose buffer type differs from the input's; the call must take the TILE
    # factory instead of being rerouted into a validate failure.
    xt, it, st = _make_case([1, 1, 32, 64], [1, 1, 32, 32], -1, ttnn.TILE_LAYOUT, device)
    assert xt.memory_config().buffer_type == ttnn.BufferType.DRAM
    out, added = _run_auto_against_native(device, xt, it, st, -1, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    assert out.layout == ttnn.TILE_LAYOUT
    assert added > 0, "auto fell back to native; expected the codegen TILE factory to serve this call"


@pytest.mark.parametrize(
    "output_memory_config,expect_codegen",
    [(None, True), (ttnn.L1_MEMORY_CONFIG, False)],
    ids=["same_placement_detour", "mismatched_placement_demoted"],
)
def test_scatter_codegen_unit_row_tile_demotion_follows_detour(device, output_memory_config, expect_codegen):
    # A unit TILE row is only kept on codegen when the row-major detour serves it. With the output in
    # the input's own placement the detour runs; with a different placement the detour is unavailable
    # and the case is demoted to native rather than paying the TILE factories' padded-row cost.
    xt, it, st = _make_case([1, 1, 1, 64], [1, 1, 1, 32], -1, ttnn.TILE_LAYOUT, device)
    kwargs = {} if output_memory_config is None else {"memory_config": output_memory_config}
    _, added = _run_auto_against_native(device, xt, it, st, -1, **kwargs)
    if expect_codegen:
        assert added > 0, "auto fell back to native; expected the codegen row-major detour to serve this call"
    else:
        assert added == 0, "auto routed a unit TILE row the detour cannot serve to codegen; expected native"


def _error_of(fn):
    # The first line is the TT_FATAL/TT_THROW text; the rest is a backtrace that differs per call.
    try:
        fn()
    except RuntimeError as e:
        return str(e).splitlines()[0]
    return None


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["row_major", "tile"])
@pytest.mark.parametrize(
    "shape,index_shape", [([0, 32], [0, 16]), ([2, 0, 32], [2, 0, 16])], ids=["[0,32]", "[2,0,32]"]
)
def test_scatter_codegen_empty_operands_stay_native(device, shape, index_shape, layout, expect_error):
    # A zero extent is a divisor in the codegen host arithmetic (CB page size, page map), which dies
    # with SIGFPE rather than an exception, so empties must never reach the codegen factories: auto
    # must behave exactly as native does on them (native raises on a zero leading extent; a zero
    # scatter axis, e.g. [0] or [32, 0], crashes native itself and is not probed here), and the
    # forced entry refuses.
    xt, it, st = _make_case(shape, index_shape, -1, layout, device)
    device.clear_program_cache()
    native_error = _error_of(lambda: _force_native(xt, -1, it, st))
    entries_before = device.num_program_cache_entries()
    auto_error = _error_of(lambda: ttnn.scatter(xt, -1, it, st))
    assert auto_error == native_error, "auto diverged from native on an empty operand"
    assert device.num_program_cache_entries() == entries_before, "auto routed an empty tensor to codegen"
    with expect_error(RuntimeError, "scatter_force_codegen invoked for a case the codegen path does not support"):
        _force_codegen(xt, -1, it, st)


def test_scatter_codegen_host_operands_raise_like_native(device, expect_error):
    # Host-resident operands satisfy every static scope clause, and the routing helpers past the
    # gate read L1 sizes off the operands' device, so the gate itself has to turn them away: auto
    # must raise exactly what native raises instead of dereferencing a null device, and the forced
    # entry refuses.
    xt, it, st = _make_case([1, 1, 32, 64], [1, 1, 32, 32], -1, ttnn.TILE_LAYOUT, device=None)
    assert xt.storage_type() != ttnn.StorageType.DEVICE
    native_error = _error_of(lambda: _force_native(xt, -1, it, st))
    assert native_error is not None, "native accepted host operands"
    assert _error_of(lambda: ttnn.scatter(xt, -1, it, st)) == native_error, "auto diverged from native on host operands"
    with expect_error(RuntimeError, "scatter_force_codegen invoked for a case the codegen path does not support"):
        _force_codegen(xt, -1, it, st)


@pytest.mark.parametrize("operand", ["input", "index", "src"])
def test_scatter_codegen_over_padded_tile_operand_stays_native(device, operand, expect_error):
    # tilize_with_val_padding can pad a TILE operand past tile rounding. The codegen TILE factories
    # count pages from that padded shape while the output they allocate and the page map they fold
    # assume tile-rounded padding of the logical shape, so such an operand must stay native, whose
    # untilize-with-unpadding bookend strips the padding before anything is scattered. The gate
    # checks all three operands, so each is padded in turn.
    shape = [1, 8, 30, 30]
    x = torch.rand(shape, dtype=torch.bfloat16)
    index = _unique_index(shape, 3, shape[3])
    src = torch.rand(shape, dtype=torch.bfloat16)

    def padded(host, dtype):
        t = ttnn.tilize_with_val_padding(
            ttnn.from_torch(host, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device), [1, 8, 64, 64], 0
        )
        assert list(t.padded_shape) == [1, 8, 64, 64]
        return t

    def plain(host, dtype):
        return ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    xt = padded(x, ttnn.bfloat16) if operand == "input" else plain(x, ttnn.bfloat16)
    it = padded(index, ttnn.int32) if operand == "index" else plain(index, ttnn.int32)
    st = padded(src, ttnn.bfloat16) if operand == "src" else plain(src, ttnn.bfloat16)
    out, added = _run_auto_against_native(device, xt, it, st, -1)
    assert_equal(torch.scatter(x, 3, index.to(torch.int64), src), ttnn.to_torch(out))
    assert added == 0, "auto routed an over-padded TILE operand to codegen; expected native"
    with expect_error(RuntimeError, "scatter_force_codegen invoked for a case the codegen path does not support"):
        _force_codegen(xt, -1, it, st)


def test_scatter_codegen_row_major_under_l1_pressure_stays_native(device):
    # The ROW_MAJOR factories keep the whole input and output stick resident in L1, so a stick that
    # fits an empty L1 can still collide with buffers already living there. The live-L1 gate must
    # send such a call to native, which streams the stick in chunks, rather than let the program's
    # CB allocation fail. 300000 bf16 elements: two resident 600 kB sticks fit a free L1 on both
    # Wormhole and Blackhole but not one with 512 kB already taken.
    xt, it, st = _make_case([1, 300000], [1, 300000], -1, ttnn.ROW_MAJOR_LAYOUT, device)
    _, added = _run_auto_against_native(device, xt, it, st, -1)
    assert added > 0, "auto fell back to native on a free L1; expected the codegen ROW_MAJOR factory"
    blocker = ttnn.from_torch(
        torch.zeros([1, 262144], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    _, added = _run_auto_against_native(device, xt, it, st, -1)
    ttnn.deallocate(blocker)
    assert added == 0, "auto routed a ROW_MAJOR call whose sticks no longer fit live L1 to codegen; expected native"


def test_scatter_codegen_row_major_sliced_source_counts_against_live_l1(device):
    # A last-axis source wider than the index reaches the ROW_MAJOR factory as a slice allocated in
    # the source's own placement. With the source in L1 that slice is pending when the route is
    # chosen, so the live-L1 gate must charge it: an equal amount of L1 that is already allocated
    # (the blocker) leaves the route on codegen, the same amount still pending must send it native.
    # Sized from the device's L1 bank so the same two routes hold on Wormhole and Blackhole: four
    # resident sticks (source, blocker, input, output) take 8/9 of the bank and fit; the wide source,
    # its slice, input and output take 10/9 and do not.
    width = (ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank // 9) // 32 * 32
    xt, it, _ = _make_case([1, width], [1, width], -1, ttnn.ROW_MAJOR_LAYOUT, device)

    def in_l1(t):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )

    st = in_l1(torch.rand([1, width], dtype=torch.bfloat16))
    blocker = in_l1(torch.zeros([1, width], dtype=torch.bfloat16))
    _, added = _run_auto_against_native(device, xt, it, st, -1)
    ttnn.deallocate(blocker)
    ttnn.deallocate(st)
    assert added > 0, "auto fell back to native beside an allocated blocker; expected the codegen ROW_MAJOR factory"
    wide = in_l1(torch.rand([1, 2 * width], dtype=torch.bfloat16))
    _, added = _run_auto_against_native(device, xt, it, wide, -1)
    ttnn.deallocate(wide)
    assert (
        added == 0
    ), "auto routed a call whose pending source slice no longer fits live L1 to codegen; expected native"


def test_scatter_codegen_tile_row_too_wide_for_interleaved_streams(device):
    # The interleaved TILE factory keeps a whole row of input and output tiles resident; a row the
    # per-core L1 cannot hold twice has to stream one tile at a time instead. The width comes from the
    # device so the row just overflows the budget on Wormhole and Blackhole alike, and 160 tile rows
    # fill either grid, so the row split alone would not have chosen streaming.
    torch.manual_seed(0)
    budget = ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank
    tile_bytes = 32 * 32 * 2
    width = 32 * (budget // (2 * tile_bytes) + 1)
    rows, index_w = 32 * 160, 32
    # Unique positions per row without sorting a row this wide: stride the 32 columns across it and
    # shift by the row, which cannot collide because 31 strides stay short of the width.
    index = (torch.arange(index_w) * (width // index_w) + torch.arange(rows).unsqueeze(1)) % width
    x = torch.rand([1, 1, rows, width], dtype=torch.bfloat16)
    src = torch.rand([1, 1, rows, index_w], dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    it = ttnn.from_torch(
        index.reshape(1, 1, rows, index_w).to(torch.int32), dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device
    )
    st = ttnn.from_torch(src, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    assert_equal(ttnn.to_torch(_force_native(xt, -1, it, st)), ttnn.to_torch(_force_codegen(xt, -1, it, st)))


def test_scatter_codegen_streaming_with_more_columns_than_cores(device):
    # Streaming splits work by output column; with more columns than cores the split is uneven and
    # the ordinal-numbered kernels must stride by the core count without drifting. Width 6408: its
    # 12816 B stick is not 32 B aligned, which keeps the single tile row off the row-major detour.
    xt, it, st = _make_case([1, 1, 32, 6408], [1, 1, 32, 100], -1, ttnn.TILE_LAYOUT, device)
    assert_equal(ttnn.to_torch(_force_native(xt, -1, it, st)), ttnn.to_torch(_force_codegen(xt, -1, it, st)))


@pytest.mark.parametrize("reduce", ["add", "multiply"])
def test_scatter_codegen_row_major_bf16_reduce_matches_native(device, reduce):
    # Positions drawn from the first 8 of 64 columns, so every hit column accumulates several
    # updates: the FP32 accumulator's rounding order is what this pins against native.
    xt, it, st = _make_case([1, 1, 32, 64], [1, 1, 32, 32], -1, ttnn.ROW_MAJOR_LAYOUT, device, index_max=8)
    golden = ttnn.to_torch(_force_native(xt, -1, it, st, reduce=reduce))
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, -1, it, st, reduce=reduce)))


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["row_major", "tile"])
@pytest.mark.parametrize("index_dtype", [ttnn.int32, ttnn.uint32], ids=["int32", "uint32"])
def test_scatter_codegen_index_dtype_matches_native(device, layout, index_dtype):
    # A 400 B stick is not 32 B aligned, so the TILE leg declines the row-major detour and reaches
    # the TILE readers; a 32 B multiple would be untilized and served by the ROW_MAJOR factory.
    xt, it, st = _make_case([1, 1, 64, 200], [1, 1, 64, 100], -1, layout, device, index_dtype=index_dtype)
    golden = ttnn.to_torch(_force_native(xt, -1, it, st))
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, -1, it, st)))


def test_scatter_codegen_uint16_index_stays_native(device):
    xt, it, st = _make_case([1, 1, 64, 64], [1, 1, 64, 32], -1, ttnn.ROW_MAJOR_LAYOUT, device, index_dtype=ttnn.uint16)
    _, added = _run_auto_against_native(device, xt, it, st, -1)
    assert added == 0, "auto routed a uint16 index to codegen; expected native"


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["row_major", "tile"])
def test_scatter_codegen_sub_core_grids_matches_native(device, layout):
    # The grid, not the default, drives the split: Ht=2 tile rows against Wt_output=7 columns, so on
    # the full grid column_cores=7 > Ht selects the column-split (streaming) factory, while a
    # two-core grid caps column_cores at 2 and lands the same call on the row-resident factory. The
    # 400 B stick is not 32 B aligned, so the TILE leg declines the row-major detour and reaches
    # those factories at all.
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    xt, it, st = _make_case([1, 1, 64, 200], [1, 1, 64, 100], -1, layout, device)
    golden = ttnn.to_torch(_force_native(xt, -1, it, st, sub_core_grids=grid))
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, -1, it, st, sub_core_grids=grid)))


@pytest.mark.parametrize(
    "input_dtype", [ttnn.bfloat16, ttnn.float32], ids=["bfloat16_codegen_leg", "float32_native_leg"]
)
def test_scatter_codegen_unknown_reduce_string_raises(device, input_dtype, expect_error):
    # scatter_reduction_mode() parses the reduce string ahead of routing, on both legs; a string
    # the codegen kernels do not implement must raise instead of landing as a plain overwrite.
    xt, it, st = _make_case([1, 1, 32, 64], [1, 1, 32, 32], -1, ttnn.ROW_MAJOR_LAYOUT, device, input_dtype=input_dtype)
    with expect_error(RuntimeError, "reduce must be either 'add' or 'multiply'"):
        ttnn.scatter(xt, -1, it, st, reduce="amax")


@pytest.mark.parametrize(
    "input_dtype,layout,index_layout,reduce",
    [
        (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, None),
        (ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "add"),
        (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT, None),
    ],
    ids=["float32_input", "tile_reduce", "mixed_layouts"],
)
def test_scatter_codegen_out_of_scope_cases_stay_native(device, input_dtype, layout, index_layout, reduce):
    # The clauses of the codegen scope nothing else in this file exercises. A TILE reduce matters
    # most: past the gate its only guard is a kernel static_assert, which would surface as a JIT
    # compile failure rather than a routing assertion.
    xt, it, st = _make_case(
        [1, 1, 32, 64], [1, 1, 32, 32], -1, layout, device, input_dtype=input_dtype, index_layout=index_layout
    )
    kwargs = {} if reduce is None else {"reduce": reduce}
    _, added = _run_auto_against_native(device, xt, it, st, -1, **kwargs)
    assert added == 0, "auto routed an out-of-scope case to codegen; expected native"


def test_scatter_codegen_index_empty_on_a_non_scatter_axis_stays_native(device):
    # An index with a zero extent on a non-scatter axis against a non-empty input is a no-op native
    # answers with a copy of the input; the gate sends it there on the index's volume before any
    # codegen host arithmetic could divide by that extent.
    x = torch.rand([2, 3, 4], dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    it = ttnn.from_torch(
        torch.zeros([2, 0, 4], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    st = ttnn.from_torch(
        torch.zeros([2, 0, 4], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    device.clear_program_cache()
    native_error = _error_of(lambda: _force_native(xt, -1, it, st))
    entries_before = device.num_program_cache_entries()
    auto_error = _error_of(lambda: ttnn.scatter(xt, -1, it, st))
    assert auto_error == native_error, "auto diverged from native on an empty index"
    assert device.num_program_cache_entries() == entries_before, "auto routed an empty index to codegen"


@pytest.mark.parametrize(
    "shape,index_shape,dim",
    [([1, 1, 32, 64], [1, 1, 32, 32], 10), ([1, 1, 32, 64], [1, 1, 32, 32], -10), ([2] * 9 + [64], [2] * 9 + [32], -1)],
    ids=["dim=10", "dim=-10", "rank-10"],
)
def test_scatter_codegen_out_of_range_dim_and_deep_rank_follow_native(device, shape, index_shape, dim):
    # The uint16 rows of test_scatter_failing_cases never reach these clauses because the index dtype
    # rejects them first. With an int32 index the dim clause and the page-rank clause are what keep
    # the call off codegen: auto must raise or answer exactly as native, with no codegen entry built.
    xt, it, st = _make_case(
        shape, index_shape, dim if -len(shape) <= dim < len(shape) else -1, ttnn.TILE_LAYOUT, device
    )
    device.clear_program_cache()
    native_error = _error_of(lambda: _force_native(xt, dim, it, st))
    entries_before = device.num_program_cache_entries()
    auto_error = _error_of(lambda: ttnn.scatter(xt, dim, it, st))
    assert auto_error == native_error, "auto diverged from native"
    if native_error is None:
        assert_equal(ttnn.to_torch(_force_native(xt, dim, it, st)), ttnn.to_torch(ttnn.scatter(xt, dim, it, st)))
    assert device.num_program_cache_entries() == entries_before, "auto routed to codegen; expected native"


def test_scatter_codegen_sharded_output_stays_native(device):
    # The factories size the output CB from an interleaved page, so a sharded output placement is
    # outside the call contract and must route native; the values alone cannot tell the routes apart.
    xt, it, st = _make_case([1, 1, 32, 64], [1, 1, 32, 32], -1, ttnn.TILE_LAYOUT, device)
    sharded = ttnn.create_sharded_memory_config(
        [1, 1, 32, 64], core_grid=ttnn.CoreGrid(x=1, y=1), strategy=ttnn.ShardStrategy.HEIGHT
    )
    _, added = _run_auto_against_native(device, xt, it, st, -1, memory_config=sharded)
    assert added == 0, "auto routed a sharded output placement to codegen; expected native"


def test_scatter_codegen_row_major_output_in_another_buffer_type_stays_native(device):
    # The ROW_MAJOR factories size the output stick from the input's own aligned page, which only
    # holds when both live in the same buffer type; a DRAM input with an L1 output routes native.
    xt, it, st = _make_case([1, 1, 32, 64], [1, 1, 32, 32], -1, ttnn.ROW_MAJOR_LAYOUT, device)
    assert xt.memory_config().buffer_type == ttnn.BufferType.DRAM
    out, added = _run_auto_against_native(device, xt, it, st, -1, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    assert added == 0, "auto routed a ROW_MAJOR call with a mismatched output buffer type to codegen; expected native"


def test_scatter_codegen_program_cache_key_follows_the_l1_frontier(device):
    # The ROW_MAJOR factory sizes its index/src chunk from the L1 free when the program is built, and
    # the program hash keys on that chunk: a program cached against a clear L1 must not be replayed
    # once an unrelated L1 tensor has moved the frontier, or its CBs would land on that tensor. Two
    # sticks take the bank less 64 kB, so a 48 kB blocker turns the default 8192-element chunk into
    # a smaller one and the second forced call has to build a second program.
    bank = ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank
    width = ((bank - 65536) // 4) // 32 * 32
    xt, it, st = _make_case([1, width], [1, width], -1, ttnn.ROW_MAJOR_LAYOUT, device)
    device.clear_program_cache()
    golden = ttnn.to_torch(_force_native(xt, -1, it, st))
    entries_before = device.num_program_cache_entries()
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, -1, it, st)))
    assert device.num_program_cache_entries() == entries_before + 1
    blocker = ttnn.from_torch(
        torch.zeros([1, 24576], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    out = _force_codegen(xt, -1, it, st)
    ttnn.deallocate(blocker)
    assert_equal(golden, ttnn.to_torch(out))
    assert device.num_program_cache_entries() == entries_before + 2, "a moved L1 frontier replayed the cached program"
