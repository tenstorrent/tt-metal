# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Contracts: (1) every case the codegen scope or call contract rejects falls back to native and matches
# reshape_force_native; (2) a demoted case runs native; (3) a routed tile-rows call traces and replays new data;
# (4) the segment-map TILE cases the merge gate's test_reshape_codegen.py leaves out are bit-exact, route as fitted,
# and key the CB depth live L1 picks; (5) a routed call runs with the least free L1 native runs with; (6) a segment map
# over its size caps runs native; (7) a plan chosen under L1 pressure traces when warmed up under that pressure.

import contextlib

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.unit_tests.operations.data_movement.test_reshape_codegen import (
    _PATHS,
    _assert_bits,
    _check_bit_exact,
    _check_edge_shape,
    _check_program_key_stable,
    _check_routing,
    _l1_hog_leaving,
    _make,
    _to_device,
)
from tests.ttnn.utils_for_testing import assert_equal
from tt_py_test_utils_common.utility_functions import skip_for_slow_dispatch

# `ttnn.reshape` routes on its own; the forced legs are the verification-only entries (reshape_force.hpp).
_force_native = ttnn._ttnn.operations.data_movement.reshape_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.reshape_force_codegen

_DRAM = ttnn.DRAM_MEMORY_CONFIG
_L1 = ttnn.L1_MEMORY_CONFIG


def _make_input(shape, dtype, seed=0):
    torch.manual_seed(seed)
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


_CALL_CONTRACT_CASE = ([12, 100], {"shape": [12, 10, 10]}, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG)


def test_reshape_codegen_unsupported_output_memory_config_falls_back(device):
    shape, kwargs, dtype, layout, placement = _CALL_CONTRACT_CASE
    xt = _to_device(device, _make_input(shape, dtype), dtype, layout, placement)
    sharded = ttnn.create_sharded_memory_config(
        list(_force_native(xt, **kwargs).shape), core_grid=ttnn.CoreGrid(x=1, y=1), strategy=ttnn.ShardStrategy.HEIGHT
    )
    golden = _force_native(xt, **kwargs, memory_config=sharded)
    out = ttnn.reshape(xt, **kwargs, memory_config=sharded)
    assert out.memory_config() == golden.memory_config()
    assert_equal(ttnn.to_torch(golden), ttnn.to_torch(out))


# (id, in shape, out shape, dtype, layout): outside the codegen scope.
_OUT_OF_SCOPE = [
    ("row_major", [512, 256], [1024, 128], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ("bfloat8_b", [1, 1, 64, 128], [1, 1, 128, 64], ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
]


@pytest.mark.parametrize("case,in_shape,out_shape,dtype,layout", _OUT_OF_SCOPE, ids=[c[0] for c in _OUT_OF_SCOPE])
def test_reshape_codegen_out_of_scope_falls_back(device, case, in_shape, out_shape, dtype, layout):
    xt = _to_device(device, _make_input(in_shape, dtype), dtype, layout)
    golden = ttnn.to_torch(_force_native(xt, out_shape))
    assert_equal(golden, ttnn.to_torch(ttnn.reshape(xt, out_shape)))


_SHARD_STRATEGIES = {
    "height": (ttnn.ShardStrategy.HEIGHT, ttnn.CoreGrid(x=2, y=1)),
    "width": (ttnn.ShardStrategy.WIDTH, ttnn.CoreGrid(x=2, y=1)),
    "block": (ttnn.ShardStrategy.BLOCK, ttnn.CoreGrid(x=2, y=2)),
}


@pytest.mark.parametrize("strategy", list(_SHARD_STRATEGIES), ids=list(_SHARD_STRATEGIES))
def test_reshape_codegen_sharded_input_falls_back(device, strategy):
    shape, out_shape = [1, 1, 64, 256], [1, 1, 128, 128]
    source = _make_input(shape, ttnn.bfloat16)
    shard_strategy, core_grid = _SHARD_STRATEGIES[strategy]
    sharded = ttnn.create_sharded_memory_config(shape, core_grid=core_grid, strategy=shard_strategy)
    xt = _to_device(device, source, ttnn.bfloat16, ttnn.TILE_LAYOUT, sharded)
    golden = ttnn.to_torch(_force_native(xt, out_shape, memory_config=_DRAM))
    out = ttnn.to_torch(ttnn.reshape(xt, out_shape, memory_config=_DRAM))
    assert_equal(golden, out)
    assert_equal(source.reshape(out_shape), out)


def test_reshape_codegen_explicit_pad_value_falls_back(device):
    shape, out_shape = [1, 64, 1024], [1, 64, 8, 128]
    xt = _to_device(device, _make_input(shape, ttnn.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    golden = _force_native(xt, out_shape, pad_value=0.0)
    out = ttnn.reshape(xt, out_shape, pad_value=0.0)
    assert_equal(golden.cpu().to_torch_with_padded_shape(), out.cpu().to_torch_with_padded_shape())


def test_reshape_codegen_explicit_padded_shape_falls_back(device):
    shape, logical, padded = [1, 2, 30, 256], [1, 1, 60, 256], [1, 1, 64, 256]
    source = _make_input(shape, ttnn.bfloat16)
    xt = _to_device(device, source, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    golden = ttnn.to_torch(_force_native(xt, logical))
    out = ttnn.reshape(xt, ttnn.Shape(logical), ttnn.Shape(padded))
    assert list(out.shape) == logical
    assert_equal(golden, ttnn.to_torch(out))
    assert_equal(source.reshape(logical), ttnn.to_torch(out))


def test_reshape_codegen_demoted_case_runs_native(device):
    # A width change saves only 2x the reads on far fewer cores, so the tile-rows program is demoted.
    shape, out_shape = [1, 1, 64, 96], [1, 1, 32, 192]
    xt = _to_device(device, _make_input(shape, ttnn.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    golden = ttnn.to_torch(_force_native(xt, out_shape))
    # Native compiled its program; auto taking codegen would compile a new one.
    entries = device.num_program_cache_entries()
    assert_equal(golden, ttnn.to_torch(ttnn.reshape(xt, out_shape)))
    assert device.num_program_cache_entries() == entries, "auto did not run native"


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32768}], indirect=True)
def test_reshape_routes_tile_rows_case_under_trace(device):
    shape, out_shape, dtype = [1, 64, 1024], [1, 64, 8, 128], ttnn.bfloat16
    xt = _to_device(device, _make_input(shape, dtype), dtype, ttnn.TILE_LAYOUT)
    _force_codegen(xt, out_shape)
    entries = device.num_program_cache_entries()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = ttnn.reshape(xt, out_shape)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    # Auto must reuse the codegen program: trace capture cannot compile a new one.
    assert device.num_program_cache_entries() == entries
    fresh = _make_input(shape, dtype, seed=1)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(fresh, dtype=dtype, layout=ttnn.TILE_LAYOUT), xt)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.release_trace(device, tid)
    assert_equal(fresh.reshape(out_shape), ttnn.to_torch(out))


# (id, in shape, out shape, dtype): segment-map programs beyond the gate's map cases.
_TILE_PATHS = [
    ("tile_mapped_int32", [1, 1, 32, 100], [1, 1, 100, 32], ttnn.int32),
    ("tile_mapped_uint16", [1, 1, 32, 100], [1, 1, 100, 32], ttnn.uint16),
    ("tile_mapped_wide", [1, 8, 64, 1000], [1, 8, 1000, 64], ttnn.bfloat16),
]


@pytest.mark.parametrize("mem", [_DRAM, _L1], ids=["dram", "l1"])
@pytest.mark.parametrize("case,in_shape,out_shape,dtype", _TILE_PATHS, ids=[p[0] for p in _TILE_PATHS])
def test_reshape_codegen_tile_bit_exact(device, case, in_shape, out_shape, dtype, mem):
    _check_bit_exact(device, in_shape, out_shape, dtype, ttnn.TILE_LAYOUT, mem, mem)


# Segment-map shapes with tile padding in the input, the output or both; the program writes zeros there.
_MAPPED_EDGE = [
    ("mapped_h50", [3, 50, 100], [3, 100, 50]),
    ("mapped_rank2", [50, 100], [100, 50]),
    ("mapped_h1", [5, 1, 100], [5, 4, 25]),
    ("mapped_rank5", [2, 3, 1, 50, 100], [2, 3, 50, 2, 50]),
]
_MAPPED_EDGE_CASES = [(*e, dtype) for e in _MAPPED_EDGE for dtype in (ttnn.bfloat16, ttnn.float32)]


@pytest.mark.parametrize(
    "case,in_shape,out_shape,dtype",
    _MAPPED_EDGE_CASES,
    ids=[f"{c[0]}-{c[3].name.lower()}" for c in _MAPPED_EDGE_CASES],
)
def test_reshape_codegen_mapped_edge_shapes(device, case, in_shape, out_shape, dtype):
    _check_edge_shape(device, in_shape, out_shape, dtype)


_TILE_KEYED = [p for p in _PATHS if p[0] == "tile_mapped_fp32"]


@pytest.mark.parametrize("mem", [_DRAM, _L1], ids=["dram", "l1"])
@pytest.mark.parametrize("path,in_shape,out_shape,dtype,layout", _TILE_KEYED, ids=[p[0] for p in _TILE_KEYED])
def test_reshape_codegen_tile_program_key_stable_while_full_plan_fits(
    device, path, in_shape, out_shape, dtype, layout, mem
):
    _check_program_key_stable(device, in_shape, out_shape, dtype, layout, mem)


def test_reshape_codegen_mapped_shrinks_input_depth_under_l1_pressure(device):
    # The fp32 segment map's map row is under 2 KiB, so its 8-, 4- and 1-deep input CBs take about 38, 22 and 10 KiB.
    # 30 and 14 KiB free pick the 4- and 1-deep programs, each a new cache entry, and auto still takes them. Native
    # needs a map page and two tiles as well, so no free size runs native where no segment map fits.
    in_shape, out_shape, dtype = [1, 1, 32, 100], [1, 1, 100, 32], ttnn.float32
    source = _make(in_shape, dtype)
    expected = source.reshape(out_shape)
    x = _to_device(device, source, dtype, ttnn.TILE_LAYOUT)
    _assert_bits(expected, ttnn.to_torch(_force_codegen(x, out_shape, memory_config=_DRAM)), dtype)
    entries = device.num_program_cache_entries()
    for step, free_kib in enumerate((30, 14), start=1):
        hog = _l1_hog_leaving(device, free_kib * 1024)
        try:
            _assert_bits(expected, ttnn.to_torch(_force_codegen(x, out_shape, memory_config=_DRAM)), dtype)
            assert device.num_program_cache_entries() == entries + step
            _assert_bits(expected, ttnn.to_torch(ttnn.reshape(x, out_shape, memory_config=_DRAM)), dtype)
            assert device.num_program_cache_entries() == entries + step
        finally:
            ttnn.deallocate(hog)


# (id, in shape, out shape, dtype, layout, expected route), as in the gate's _ROUTES.
_TILE_ROUTES = [
    ("tile_unaligned_width_fp32", [1, 1, 32, 100], [1, 1, 100, 32], ttnn.float32, ttnn.TILE_LAYOUT, "codegen"),
    ("tile_unaligned_width_uint16", [1, 1, 32, 100], [1, 1, 100, 32], ttnn.uint16, ttnn.TILE_LAYOUT, "codegen"),
]


@pytest.mark.parametrize("case,in_shape,out_shape,dtype,layout,route", _TILE_ROUTES, ids=[r[0] for r in _TILE_ROUTES])
def test_reshape_codegen_tile_routing(device, case, in_shape, out_shape, dtype, layout, route):
    _check_routing(device, in_shape, out_shape, dtype, layout, route)


# Row-major bf16 pages of this many bytes, the L1 alignment, so the hog sets free L1 to the alignment.
_HOG_PAGE_BYTES = 16
_CLASH = "clash with L1 buffers"


def _l1_hog_leaving_exact(device, free_bytes_per_bank):
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    pages_per_bank = (view.largest_contiguous_bytes_free_per_bank - free_bytes_per_bank) // _HOG_PAGE_BYTES
    if pages_per_bank <= 0:
        pytest.skip("L1 already has less free space than the test needs")
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape((pages_per_bank * view.num_banks, _HOG_PAGE_BYTES // 2)),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        _L1,
    )


def _native_runs_with(device, x, out_shape, free_bytes):
    hog = _l1_hog_leaving_exact(device, free_bytes)
    logger.info(f'[EXPECTED_ERROR BEGIN] RuntimeError message="{_CLASH}"')
    try:
        ttnn.deallocate(_force_native(x, out_shape, memory_config=_DRAM))
        return True
    except RuntimeError as error:
        if _CLASH not in str(error):
            raise
        return False
    finally:
        logger.info(f'[EXPECTED_ERROR END] RuntimeError message="{_CLASH}"')
        ttnn.deallocate(hog)


# (id, in shape, out shape, dtype, free L1 per bank native is known to run with). Routed to codegen; the segment
# map's smallest plan is the fallback whenever no larger plan fits.
_L1_FLOOR = [
    ("tile_rows_bf16", [1, 256, 4096], [1, 256, 32, 128], ttnn.bfloat16, 32 * 1024),
    ("mapped_fp32", [1, 1, 32, 100], [1, 1, 100, 32], ttnn.float32, 32 * 1024),
]


@pytest.mark.parametrize("case,in_shape,out_shape,dtype,ceiling", _L1_FLOOR, ids=[c[0] for c in _L1_FLOOR])
def test_reshape_codegen_runs_at_native_l1_floor(device, case, in_shape, out_shape, dtype, ceiling):
    source = _make(in_shape, dtype)
    x = _to_device(device, source, dtype, ttnn.TILE_LAYOUT)
    assert _native_runs_with(device, x, out_shape, ceiling), f"native does not run with {ceiling} B free"
    # Native's least free L1, to the hog's granularity: it fails at `low` and runs at `high`.
    low, high = 0, ceiling
    while high - low > _HOG_PAGE_BYTES:
        middle = (low + high) // 2 // _HOG_PAGE_BYTES * _HOG_PAGE_BYTES
        if _native_runs_with(device, x, out_shape, middle):
            high = middle
        else:
            low = middle
    hog = _l1_hog_leaving_exact(device, high)
    try:
        out = ttnn.reshape(x, out_shape, memory_config=_DRAM)
        _assert_bits(source.reshape(out_shape), ttnn.to_torch(out), dtype)
    finally:
        ttnn.deallocate(hog)


def test_reshape_codegen_mapped_over_caps_runs_native(device, expect_error):
    # Every datum of a width-1 input is its own segment, so each output tile's map row holds 1024 entries (16 KiB);
    # 1025 output tiles put the map over its 16 MiB cap.
    in_shape, out_shape = [1, 1025 * 1024, 1], [1, 1025 * 32, 32]
    source = torch.rand(in_shape, dtype=torch.bfloat16)
    x = _to_device(device, source, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "within its size caps"):
        _force_codegen(x, out_shape, memory_config=_DRAM)
    assert_equal(source.reshape(out_shape), ttnn.to_torch(_force_native(x, out_shape, memory_config=_DRAM)))
    # Native compiled its program; auto taking codegen would compile a new one.
    entries = device.num_program_cache_entries()
    assert_equal(source.reshape(out_shape), ttnn.to_torch(ttnn.reshape(x, out_shape, memory_config=_DRAM)))
    assert device.num_program_cache_entries() == entries, "auto did not run native"


def _best_effort(action):
    # Once a capture has gone wrong these calls fail too, and the original error is the one worth surfacing.
    try:
        action()
    except RuntimeError:
        pass


@contextlib.contextmanager
def _capturing(device):
    # A queue left mid-capture fails every later call against the device, so the capture closes even when the
    # captured op raises.
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        yield trace_id
    finally:
        _best_effort(lambda: ttnn.end_trace_capture(device, trace_id, cq_id=0))


@pytest.mark.parametrize(
    "hog_before_warmup",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                reason="https://github.com/tenstorrent/tt-metal/issues/46533: a plan chosen from live L1 misses the "
                "cache when free L1 shrinks between warm-up and capture, and a capture cannot compile the plan it "
                "then needs",
                raises=RuntimeError,
                strict=True,
            ),
        ),
    ],
    ids=["warm_under_pressure", "warm_then_pressure"],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 4 * 1024 * 1024}], indirect=True)
@skip_for_slow_dispatch("trace capture requires fast dispatch")
def test_reshape_codegen_trace_capture_under_l1_pressure(device, hog_before_warmup):
    # Trace capture compiles nothing, so it needs the plan warm-up cached. With 240 KiB free this head split takes
    # its single-buffered plan instead of the full one. warm_under_pressure warms up under that pressure, as
    # TraceCorrectness.md asks of a model, and the capture reuses the warm entry. warm_then_pressure warms up on
    # free L1 and captures under pressure: the capture needs a plan warm-up never compiled and aborts. Strict, so the
    # case fails if the pressure stops moving the plan.
    in_shape, out_shape, dtype = [1, 256, 4096], [1, 256, 32, 128], ttnn.float32
    source = _make(in_shape, dtype)
    x = _to_device(device, source, dtype, ttnn.TILE_LAYOUT)
    trace_id = None
    hog = None
    try:
        if hog_before_warmup:
            hog = _l1_hog_leaving(device, 240 * 1024)
        warm = ttnn.reshape(x, out_shape, memory_config=_DRAM)
        _assert_bits(source.reshape(out_shape), ttnn.to_torch(warm), dtype)
        ttnn.deallocate(warm)
        entries = device.num_program_cache_entries()
        if not hog_before_warmup:
            hog = _l1_hog_leaving(device, 240 * 1024)

        with _capturing(device) as trace_id:
            out = ttnn.reshape(x, out_shape, memory_config=_DRAM)
        ttnn.synchronize_device(device)
        assert device.num_program_cache_entries() == entries, "trace capture compiled a program"

        # The capture already left a correct result in out, so only a swapped input shows the replay ran.
        fresh = torch.flip(source, dims=[-1])
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(fresh, dtype=dtype, layout=ttnn.TILE_LAYOUT), x)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        _assert_bits(fresh.reshape(out_shape), ttnn.to_torch(out), dtype)
    finally:
        if trace_id is not None:
            _best_effort(lambda: ttnn.release_trace(device, trace_id))
        if hog is not None:
            ttnn.deallocate(hog)
