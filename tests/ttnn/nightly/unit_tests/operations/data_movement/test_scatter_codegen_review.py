# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Hand-written regressions for the scatter routing seams the generated sweep in
# test_scatter_codegen_routing.py does not reach: the demotion predicate's axis spelling, small
# row-axis shapes, an output placement that differs from the input's, empty operands, the bf16
# reduce path, the unknown reduce string, the out-of-scope clauses that stay native, the index
# dtypes, an explicit core grid, host-resident operands, an operand padded past tile rounding and
# a row-major call whose resident sticks no longer fit the L1 that is actually free.

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


def test_scatter_codegen_over_padded_tile_operand_stays_native(device, expect_error):
    # tilize_with_val_padding can pad a TILE operand past tile rounding. The codegen TILE factories
    # count pages from that padded shape while the output they allocate and the page map they fold
    # assume tile-rounded padding of the logical shape, so such an operand must stay native, whose
    # untilize-with-unpadding bookend strips the padding before anything is scattered.
    shape = [1, 8, 30, 30]
    x = torch.rand(shape, dtype=torch.bfloat16)
    index = _unique_index(shape, 3, shape[3])
    src = torch.rand(shape, dtype=torch.bfloat16)
    xt = ttnn.tilize_with_val_padding(
        ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device), [1, 8, 64, 64], 0.0
    )
    assert list(xt.padded_shape) == [1, 8, 64, 64]
    it = ttnn.from_torch(index, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    st = ttnn.from_torch(src, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
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
