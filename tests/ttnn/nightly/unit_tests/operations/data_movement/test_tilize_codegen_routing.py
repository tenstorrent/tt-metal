# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Contracts: (1) every case the codegen gate rejects falls back to native; (2) an accepted case
# dispatched twice on the codegen path stays a program-cache hit and rebinds its buffers; (3) no
# case is perf-demoted, so small inputs are asserted to stay on codegen in the small-cases section;
# (4) for the cache-transition legs, earlier tensors stay alive across later calls, and
# program-cache entries are counted around the op call alone.

import contextlib

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.data_movement.test_pad import skip_unless_dm_codegen_arch
from tests.ttnn.utils_for_testing import assert_equal
from tt_py_test_utils_common.utility_functions import skip_for_slow_dispatch

# `ttnn.tilize` takes no implementation argument -- it routes on its own. The forced legs below are
# the verification-only entries in the private module; see tilize_force.hpp.
_force_native = ttnn._ttnn.operations.data_movement.tilize_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.tilize_force_codegen


@pytest.fixture(autouse=True)
def _require_codegen_arch(device):
    # Elsewhere every call is native and the codegen legs have nothing to exercise.
    skip_unless_dm_codegen_arch(device)


def _make_input(shape, dtype):
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    # Signed and wider than [0, 1): an integer-output golden of [0, 1) is all zeros, which a stale
    # binding or an overwritten earlier output would still match.
    return (torch.rand(shape, dtype=torch.bfloat16) - 0.5) * 200


def _golden(tensor, **kwargs):
    # A native golden, so the program cache holds what native leaves in it on every leg.
    return ttnn.to_torch(_force_native(tensor, **kwargs))


def _codegen_addrs(tensors):
    # Every buffer address of one tensor or of a list of them, so a list op is compared per input.
    each = tensors if isinstance(tensors, (list, tuple)) else [tensors]
    return tuple(t.buffer_address() for t in each)


def _codegen_added(device, call):
    # The count is read immediately around the call, so setup, readback and golden computation
    # cannot move it: only the op itself can.
    before = device.num_program_cache_entries()
    out = call()
    return out, device.num_program_cache_entries() - before


def _codegen_assert_distinct_addresses(calls):
    # `calls` holds `(input, kwargs, output)` per call, every tensor still alive; a refused call has
    # no output. Tensors of one call may share an address (an op may return its own input), tensors
    # of different calls may not: a program that kept an earlier address would otherwise read or
    # write the other call's recycled buffer and still pass.
    owner = {}
    for index, (inp, call_kwargs, out) in enumerate(calls):
        tensors = [inp, *(v for v in call_kwargs.values() if hasattr(v, "buffer_address"))]
        if out is not None:
            tensors.append(out)
        for address in {a for t in tensors for a in _codegen_addrs(t)}:
            first = owner.setdefault(address, index)
            assert first == index, (
                f"two live tensors from different calls ({first} and {index}) share a buffer address ({address}); "
                "the hit proves nothing"
            )


def _codegen_fresh(shape, kwargs, dtype, layout, device, placement):
    # New data in a new allocation on every call. A secondary input is built again with it.
    tensor = ttnn.from_torch(
        _make_input(shape, dtype), dtype=dtype, layout=layout, device=device, memory_config=placement
    )
    return tensor, kwargs


def _codegen_draw(earlier, shape, kwargs, dtype, layout, device, placement):
    # `_codegen_fresh` and its golden, drawn again while the golden repeats an earlier call's: a
    # stale binding to that call would read the same values, so the value checks could not tell it
    # from a correct one. When every draw repeats, the last is kept: the checks only lose power.
    for _ in range(4):
        tensor, call_kwargs = _codegen_fresh(shape, kwargs, dtype, layout, device, placement)
        golden = _golden(tensor, **call_kwargs)
        if not any(torch.equal(golden, each) for each in earlier):
            break
    return tensor, call_kwargs, golden


_CACHE_HIT = [
    ([32, 32], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG),
]
_CACHE_HIT_IDS = [
    "[32, 32]@L1|memory_config=BufferType.DRAM|int32|row_major",
]


@pytest.mark.parametrize("shape,kwargs,dtype,layout,placement", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_tilize_codegen_auto_routes_to_codegen(device, shape, kwargs, dtype, layout, placement):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=placement)
    device.clear_program_cache()
    golden = _golden(xt, **kwargs)
    entries_before = device.num_program_cache_entries()
    out = ttnn.tilize(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed a supported case to native (program cache did not grow); expected codegen"
    assert device.num_program_cache_entries() > entries_before, msg


@pytest.mark.parametrize("shape,kwargs,dtype,layout,placement", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_tilize_codegen_program_cache_hit(device, shape, kwargs, dtype, layout, placement):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=placement)
    golden = _golden(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
    # Same spec, a distinct allocation: the cached program must rebind its Buffer*s
    # instead of reusing the first dispatch's addresses.
    yt = ttnn.from_torch(_make_input(shape, dtype), dtype=dtype, layout=layout, device=device, memory_config=placement)
    second_golden = _golden(yt, **kwargs)
    assert all(
        a != b for a, b in zip(_codegen_addrs(yt), _codegen_addrs(xt))
    ), "the second input reused the first allocation; the hit proves nothing"
    entries_before_hit = device.num_program_cache_entries()
    out = _force_codegen(yt, **kwargs)
    msg = "second forced-codegen dispatch missed the program cache"
    assert device.num_program_cache_entries() == entries_before_hit, msg
    assert_equal(second_golden, ttnn.to_torch(out))


_SEQ_FRESH = [
    (
        [1, 1, 32, 32],
        {"memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
]
_SEQ_FRESH_IDS = [
    "[1, 1, 32, 32]|memory_config=BufferType.DRAM|bfloat16|row_major",
]


@pytest.mark.parametrize("shape,kwargs,dtype,layout,placement", _SEQ_FRESH, ids=_SEQ_FRESH_IDS)
def test_tilize_codegen_cache_hit_on_fresh_buffers(device, shape, kwargs, dtype, layout, placement):
    device.clear_program_cache()
    live = []
    for call in range(3):
        inp, call_kwargs, golden = _codegen_draw([g for *_, g in live], shape, kwargs, dtype, layout, device, placement)
        out, added = _codegen_added(device, lambda: _force_codegen(inp, **call_kwargs))
        live.append((inp, call_kwargs, out, golden))
        if call == 0:
            if added == 0:
                pytest.skip("the first forced-codegen dispatch built no program, so there is nothing to reuse")
        else:
            assert added == 0, f"dispatch {call} on fresh buffers missed the program cache"
        assert_equal(golden, ttnn.to_torch(out))
    _codegen_assert_distinct_addresses([(i, k, o) for i, k, o, _ in live])
    for *_, out, golden in live[:2]:
        assert_equal(golden, ttnn.to_torch(out))


_SEQ_SWITCH = [
    (
        [1, 1, 32, 32],
        {"memory_config": ttnn.DRAM_MEMORY_CONFIG},
        [64, 32],
        {"memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
]
_SEQ_SWITCH_IDS = [
    "[1, 1, 32, 32]|memory_config=BufferType.DRAM|bfloat16|row_major->[64, 32]|memory_config=BufferType.DRAM|bfloat16|row_major",
]


@pytest.mark.parametrize("shape,kwargs,b_shape,b_kwargs,dtype,layout,placement", _SEQ_SWITCH, ids=_SEQ_SWITCH_IDS)
def test_tilize_codegen_cache_survives_a_program_switch(
    device, shape, kwargs, b_shape, b_kwargs, dtype, layout, placement
):
    device.clear_program_cache()
    a_first, a_first_kwargs = _codegen_fresh(shape, kwargs, dtype, layout, device, placement)
    a_first_golden = _golden(a_first, **a_first_kwargs)
    a_first_out, added = _codegen_added(device, lambda: _force_codegen(a_first, **a_first_kwargs))
    if added == 0:
        pytest.skip("the first forced-codegen dispatch built no program, so there is nothing to reuse")
    assert_equal(a_first_golden, ttnn.to_torch(a_first_out))
    b_input, b_call_kwargs = _codegen_fresh(b_shape, b_kwargs, dtype, layout, device, placement)
    b_golden = _golden(b_input, **b_call_kwargs)
    b_out = _force_codegen(b_input, **b_call_kwargs)
    assert_equal(b_golden, ttnn.to_torch(b_out))
    a_again, a_again_kwargs, a_again_golden = _codegen_draw(
        [a_first_golden], shape, kwargs, dtype, layout, device, placement
    )
    a_again_out, added = _codegen_added(device, lambda: _force_codegen(a_again, **a_again_kwargs))
    assert added == 0, "returning to the first program after another one missed the program cache"
    assert_equal(a_again_golden, ttnn.to_torch(a_again_out))
    _codegen_assert_distinct_addresses(
        [
            (a_first, a_first_kwargs, a_first_out),
            (b_input, b_call_kwargs, b_out),
            (a_again, a_again_kwargs, a_again_out),
        ]
    )
    assert_equal(a_first_golden, ttnn.to_torch(a_first_out))


def _native_route(device, xt, **kwargs):
    # The forced native call builds native's program, so a routed call that lands on native is a
    # cache hit; one that lands on codegen adds an entry.
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    before = device.num_program_cache_entries()
    out = ttnn.tilize(xt, **kwargs)
    return golden, out, device.num_program_cache_entries() - before


def _height_sharded_l1(shape, num_cores):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(grid, [shape[-2] // num_cores, shape[-1]], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


# (shape, input placement, input dtype, tilize kwargs): each case is outside the codegen scope for one reason.
_SUB_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 1))})
_TRANSPOSED_TILE = ttnn.Tile([32, 32], transpose_tile=True)
_FALLBACK = {
    "sharded_input": (
        [1, 1, 64, 64],
        _height_sharded_l1([1, 1, 64, 64], 2),
        ttnn.bfloat16,
        {"memory_config": ttnn.L1_MEMORY_CONFIG},
    ),
    "block_float_output": ([1, 1, 64, 64], ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16, {"dtype": ttnn.bfloat8_b}),
    "use_multicore_false": ([1, 1, 64, 64], ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16, {"use_multicore": False}),
    "sub_core_grids": ([1, 1, 64, 64], ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16, {"sub_core_grids": _SUB_GRID}),
    "use_low_perf": ([1, 1, 64, 64], ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16, {"use_low_perf": True}),
    "tile_16x32": ([1, 1, 64, 64], ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16, {"tile": ttnn.Tile([16, 32])}),
    "tile_transposed_32x32": ([1, 1, 64, 64], ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16, {"tile": _TRANSPOSED_TILE}),
    "float32_to_int32": ([1, 1, 64, 64], ttnn.DRAM_MEMORY_CONFIG, ttnn.float32, {"dtype": ttnn.int32}),
    "bfloat16_to_int32": ([1, 1, 64, 64], ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16, {"dtype": ttnn.int32}),
}
# Only the route is asserted for these: native's answer to them is not checked by this file.
_ROUTE_ONLY = {"tile_transposed_32x32", "bfloat16_to_int32"}
_TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}


@pytest.mark.parametrize("case", list(_FALLBACK))
def test_tilize_out_of_scope_case_falls_back_to_native(device, case):
    shape, placement, in_dtype, kwargs = _FALLBACK[case]
    torch.manual_seed(0)
    host = torch.rand(shape, dtype=_TORCH_DTYPE[in_dtype])
    xt = ttnn.from_torch(host, dtype=in_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=placement)
    golden, out, added = _native_route(device, xt, **kwargs)
    assert added == 0, "auto built a new program for an out-of-scope case; expected native's cached program"
    assert out.layout == ttnn.TILE_LAYOUT
    if case in _ROUTE_ONLY:
        return
    assert_equal(golden, ttnn.to_torch(out))
    if "dtype" not in kwargs:
        assert_equal(host, ttnn.to_torch(out))


# The forced codegen entry raises instead of falling back to native. It takes no tile argument, and a row-major
# input does not keep a non-default tile, so the tile cases are covered by the auto test above only.
@pytest.mark.parametrize("case", ["sharded_input", "block_float_output", "float32_to_int32", "bfloat16_to_int32"])
def test_tilize_force_codegen_raises_out_of_scope(device, case, expect_error):
    shape, placement, in_dtype, kwargs = _FALLBACK[case]
    host = torch.rand(shape, dtype=_TORCH_DTYPE[in_dtype])
    xt = ttnn.from_torch(host, dtype=in_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=placement)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, **kwargs)


# One tile row wide enough to split across cores (Column2D); the generated legs above run on the cases in
# _CACHE_HIT, so these reuse their bodies.
_CACHE_HIT_COLUMN2D = [
    (
        [1, 1, 32, 1024],
        {"memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
]
_CACHE_HIT_COLUMN2D_IDS = ["column2d_bfloat16"]


@pytest.mark.parametrize("shape,kwargs,dtype,layout,placement", _CACHE_HIT_COLUMN2D, ids=_CACHE_HIT_COLUMN2D_IDS)
def test_tilize_codegen_column2d_auto_routes_to_codegen(device, shape, kwargs, dtype, layout, placement):
    test_tilize_codegen_auto_routes_to_codegen(device, shape, kwargs, dtype, layout, placement)


@pytest.mark.parametrize("shape,kwargs,dtype,layout,placement", _CACHE_HIT_COLUMN2D, ids=_CACHE_HIT_COLUMN2D_IDS)
def test_tilize_codegen_column2d_program_cache_hit(device, shape, kwargs, dtype, layout, placement):
    test_tilize_codegen_program_cache_hit(device, shape, kwargs, dtype, layout, placement)


# More tile rows than any Wormhole or Blackhole grid has cores, 16 tiles wide: the row route on the chunked
# stick reader, which binds its input buffer separately from the other readers.
_CACHE_HIT_CHUNKED = [
    (
        [1, 1, 32 * 144, 32 * 16],
        {"memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
]
_CACHE_HIT_CHUNKED_IDS = ["chunked_bfloat16"]


@pytest.mark.parametrize("shape,kwargs,dtype,layout,placement", _CACHE_HIT_CHUNKED, ids=_CACHE_HIT_CHUNKED_IDS)
def test_tilize_codegen_chunked_program_cache_hit(device, shape, kwargs, dtype, layout, placement):
    grid = device.compute_with_storage_grid_size()
    assert shape[-2] // 32 > grid.x * grid.y, "the case must keep more tile rows than cores"
    test_tilize_codegen_program_cache_hit(device, shape, kwargs, dtype, layout, placement)


# Codegen does not convert dtypes, so a conversion on any route's shape stays on native under auto and the forced
# entry refuses it. Only the route is asserted for the auto call.
_CONVERSION_SHAPES = {
    "row": [1, 1, 2 * 32, 32],
    "column2d": [1, 1, 32, 1024],
    "multi_row": [1, 1, 8 * 32, 8 * 32],
}


@pytest.mark.parametrize("shape", list(_CONVERSION_SHAPES.values()), ids=list(_CONVERSION_SHAPES))
def test_tilize_auto_keeps_conversion_on_native(device, shape, expect_error):
    kwargs = {"memory_config": ttnn.DRAM_MEMORY_CONFIG, "dtype": ttnn.int32}
    torch.manual_seed(0)
    host = _make_input(shape, ttnn.bfloat16)
    xt = ttnn.from_torch(
        host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    device.clear_program_cache()
    _, out, added = _native_route(device, xt, **kwargs)
    assert added == 0, "auto built a new program for a dtype conversion; expected native's cached program"
    assert out.dtype == ttnn.int32 and out.layout == ttnn.TILE_LAYOUT
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, **kwargs)


# --- Live-L1 CB plan ---
# The CB plan is sized from the L1 free at dispatch and is part of the program-cache key. A row-route
# case (more tile rows than cores, so no column split and no write batching) double-buffers its whole
# 32-tile row when L1 is clear; with the free L1 pinned below that footprint it must compile a
# column-chunked plan instead of replaying the cached one.
_PLAN_SHAPE = [1, 1, 160 * 32, 32 * 32]
_BF16_TILE_BYTES = 32 * 32 * 2
_FULL_ROW_DOUBLE_BUFFERED = 2 * 32 * 2 * _BF16_TILE_BYTES  # input + output CBs, two rows deep
_PIN_HEADROOM = 96 * 1024


def _pin_l1_headroom(device, headroom_bytes):
    """Lowers the live L1 frontier to about `headroom_bytes` above the CB base on every bank."""
    info = ttnn._ttnn.reports.get_device_info(device)
    tiles_per_bank = (info.cb_limit - headroom_bytes) // _BF16_TILE_BYTES
    resident = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32 * tiles_per_bank, 32 * info.l1_num_banks]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    )
    return resident, resident.buffer_address() - info.address_at_first_l1_cb_buffer


# The output stays in DRAM: an L1 output of this shape would not fit under the pin.
@pytest.mark.parametrize(
    "kwargs",
    [{"memory_config": ttnn.DRAM_MEMORY_CONFIG}],
    ids=["bfloat16"],
)
def test_tilize_codegen_cb_plan_follows_live_l1(device, kwargs):
    grid = device.compute_with_storage_grid_size()
    assert _PLAN_SHAPE[-2] // 32 >= grid.x * grid.y, "the plan case must keep more tile rows than cores"
    torch.manual_seed(0)
    host = _make_input(_PLAN_SHAPE, ttnn.bfloat16)
    xt = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    expected = host
    device.clear_program_cache()

    out, added = _codegen_added(device, lambda: ttnn.tilize(xt, **kwargs))
    assert_equal(expected, ttnn.to_torch(out))
    assert added > 0, "auto served the unpressured call on native; expected codegen"
    entries_clear = device.num_program_cache_entries()

    resident, headroom = _pin_l1_headroom(device, _PIN_HEADROOM)
    try:
        assert (
            8 * _BF16_TILE_BYTES <= headroom < _FULL_ROW_DOUBLE_BUFFERED
        ), f"pinned headroom {headroom} B does not force a smaller plan"
        out = ttnn.tilize(xt, **kwargs)
        assert_equal(expected, ttnn.to_torch(out))
        msg = "the pinned call replayed the program planned for clear L1"
        assert device.num_program_cache_entries() == entries_clear + 1, msg
    finally:
        ttnn.deallocate(resident)

    # Releasing the pin restores the original plan, which is still cached.
    out = ttnn.tilize(xt, **kwargs)
    assert_equal(expected, ttnn.to_torch(out))
    assert (
        device.num_program_cache_entries() == entries_clear + 1
    ), "releasing the pin did not return to the cached plan"


# Below one input and one output tile of free L1 no codegen plan fits: auto must not build a codegen program and
# the forced entry refuses. Native needs at least as much L1 as one codegen chunk, so it may refuse too; only
# its route is asserted.
def test_tilize_codegen_below_one_tile_floor_stays_off_codegen(device, expect_error):
    kwargs = {"memory_config": ttnn.DRAM_MEMORY_CONFIG}
    torch.manual_seed(0)
    host = _make_input([1, 1, 64, 64], ttnn.bfloat16)
    xt = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    device.clear_program_cache()
    # Primes the native program, so only a route to codegen grows the cache on the auto call.
    _force_native(xt, **kwargs)
    entries = device.num_program_cache_entries()

    resident, headroom = _pin_l1_headroom(device, _BF16_TILE_BYTES // 2)
    try:
        assert headroom < 2 * _BF16_TILE_BYTES, f"pinned headroom {headroom} B still holds a one-tile plan"
        with expect_error(RuntimeError, "does not fit the L1 free now"):
            _force_codegen(xt, **kwargs)
        try:
            out = ttnn.tilize(xt, **kwargs)
        except RuntimeError as error:
            assert "circular buffers" not in str(error), f"auto dispatched codegen without a plan: {error}"
        else:
            assert_equal(host, ttnn.to_torch(out))
        assert device.num_program_cache_entries() == entries, "auto built a codegen program below the one-tile floor"
    finally:
        ttnn.deallocate(resident)


# --- Trace capture ---
# Trace capture compiles nothing, so the plan it needs must already be cached. The plan follows the
# free L1, hence the capture only finds it when warm-up ran under the same L1 pressure.
_TRACE_DEVICE_PARAMS = {"trace_region_size": 4 * 1024 * 1024}


def _best_effort(action):
    try:
        action()
    except RuntimeError:
        pass


@contextlib.contextmanager
def _capturing(device):
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        yield trace_id
    finally:
        _best_effort(lambda: ttnn.end_trace_capture(device, trace_id, cq_id=0))


@pytest.mark.parametrize(
    "pin_before_warmup",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                reason="https://github.com/tenstorrent/tt-metal/issues/46533: a live-L1-derived plan "
                "misses the cache when the free L1 changes between warm-up and capture, and a capture "
                "cannot compile the plan it then needs",
            ),
        ),
    ],
    ids=["warm_under_pressure", "warm_then_pressure"],
)
@pytest.mark.parametrize("device_params", [_TRACE_DEVICE_PARAMS], indirect=True)
@skip_for_slow_dispatch("trace capture requires fast dispatch")
def test_tilize_codegen_trace_capture_under_pressure(device, pin_before_warmup):
    # A plan cached by an earlier test would let the capture succeed without warm-up having compiled it.
    device.clear_program_cache()
    kwargs = {"memory_config": ttnn.DRAM_MEMORY_CONFIG}
    torch.manual_seed(0)
    host = _make_input(_PLAN_SHAPE, ttnn.bfloat16)
    swapped = _make_input(_PLAN_SHAPE, ttnn.bfloat16)
    xt = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    trace_id = None
    resident = None
    try:
        if pin_before_warmup:
            resident, _ = _pin_l1_headroom(device, _PIN_HEADROOM)
        warm = _force_codegen(xt, **kwargs)
        assert_equal(host, ttnn.to_torch(warm))
        ttnn.deallocate(warm)
        entries_after_warmup = device.num_program_cache_entries()
        if not pin_before_warmup:
            resident, _ = _pin_l1_headroom(device, _PIN_HEADROOM)

        with _capturing(device) as trace_id:
            output = _force_codegen(xt, **kwargs)
        ttnn.synchronize_device(device)
        assert (
            device.num_program_cache_entries() == entries_after_warmup
        ), "capture needed a plan warm-up had not cached"

        # Replace the input in place, so a replay that never reached the device would differ.
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(swapped, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT), xt)
        ttnn.synchronize_device(device)
        ttnn.execute_trace(device, trace_id, blocking=True)
        ttnn.synchronize_device(device)
        assert_equal(swapped, ttnn.to_torch(output))
    finally:
        if trace_id is not None:
            _best_effort(lambda: ttnn.release_trace(device, trace_id))
        if resident is not None:
            ttnn.deallocate(resident)


# --- Large inputs ---
# One per codegen path the small cases stay off: several tile rows per core on the compact reader
# (8x3x224x224), the chunked reader with grouped tile writes (4096x4096, 1024x4096, 512x32768), and one wide
# tile row split across cores (32x32768).
@pytest.mark.parametrize(
    "shape,dtype,in_buffer",
    [
        ((8, 3, 224, 224), ttnn.bfloat16, ttnn.BufferType.L1),
        ((8, 3, 224, 224), ttnn.bfloat16, ttnn.BufferType.DRAM),
        ((1, 1, 4096, 4096), ttnn.bfloat16, ttnn.BufferType.DRAM),
        ((1, 1, 1024, 4096), ttnn.float32, ttnn.BufferType.DRAM),
        ((1, 1, 512, 32768), ttnn.bfloat16, ttnn.BufferType.DRAM),
        ((1, 1, 32, 32768), ttnn.bfloat16, ttnn.BufferType.DRAM),
    ],
    ids=[
        "8x3x224x224_bf16_l1",
        "8x3x224x224_bf16_dram",
        "4096x4096_bf16",
        "1024x4096_fp32",
        "512x32768_bf16",
        "32x32768_bf16",
    ],
)
def test_tilize_codegen_forced_large(device, shape, dtype, in_buffer):
    torch.manual_seed(0)
    host = torch.randn(shape, dtype=_TORCH_DTYPE[dtype])
    placement = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, in_buffer)
    x = ttnn.from_torch(host, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=placement)

    out = _force_codegen(x)

    assert out.layout == ttnn.TILE_LAYOUT
    assert_equal(ttnn.to_torch(x), ttnn.to_torch(out))


# --- Small cases stay on codegen ---
# Nothing is demoted: the smallest in-scope inputs (one tile column, one or two tile rows, integer or
# bf16) route to codegen under auto.
_SMALL_CASES = [
    # shape, input dtype, tilize kwargs, id
    ([1, 1, 32, 32], ttnn.int32, {}, "wt1_1_tile_row"),
    ([1, 1, 2 * 32, 32], ttnn.int32, {}, "wt1_2_tile_rows"),
    ([2, 1, 32, 32], ttnn.uint16, {}, "wt1_2_tile_rows_from_batch"),
    ([1, 1, 2 * 32, 32], ttnn.uint32, {}, "wt1_2_tile_rows_uint32"),
    ([1, 1, 2 * 32, 32], ttnn.bfloat16, {}, "wt1_2_tile_rows_bf16"),
]


@pytest.mark.parametrize("shape,in_dtype,kwargs,case_id", _SMALL_CASES, ids=[c[3] for c in _SMALL_CASES])
def test_tilize_codegen_small_cases_stay_on_codegen(device, shape, in_dtype, kwargs, case_id):
    torch.manual_seed(0)
    host = _make_input(shape, in_dtype)
    xt = ttnn.from_torch(host, dtype=in_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    # The device copy, not host: an integer input dtype truncates the fractional bf16 draw on upload.
    expected = ttnn.to_torch(xt)
    device.clear_program_cache()
    # Primes the native program, so only a route to codegen grows the cache on the auto call.
    _force_native(xt, **kwargs)
    out, added = _codegen_added(device, lambda: ttnn.tilize(xt, **kwargs))
    assert added > 0, f"{case_id}: auto sent the case to native"
    assert_equal(expected, ttnn.to_torch(out))
