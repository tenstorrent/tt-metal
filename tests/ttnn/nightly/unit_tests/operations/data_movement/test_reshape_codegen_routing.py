# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Contracts: (1) every case the codegen scope or call contract rejects falls back to native and matches
# reshape_force_native; (2) a demoted case runs native; (3) a routed tile-rows call traces and replays new data;
# (4) a small row-major call keeps its program key when the L1 frontier moves; (5) the row-major cases the merge
# gate's test_reshape_codegen.py leaves out are bit-exact, route as fitted, and key the CB depth live L1 picks; (6) so
# are the segment-map TILE cases it leaves out; (7) a routed call runs with the least free L1 native runs with; (8) a
# segment map over its size caps runs native; (9) a plan chosen under L1 pressure traces when warmed up under that
# pressure.

import contextlib

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.unit_tests.operations.data_movement.test_reshape_codegen import (
    _PATHS,
    _assert_bits,
    _check_bit_exact,
    _check_cache_hit_new_addresses,
    _check_edge_shape,
    _check_program_key_stable,
    _check_routing,
    _check_sub_core_grids,
    _l1_hog,
    _l1_hog_leaving,
    _make,
    _rm_aligned_pair,
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


@pytest.mark.parametrize("placement", [ttnn.L1_MEMORY_CONFIG, _DRAM], ids=["l1", "dram"])
def test_reshape_codegen_rm_program_key_ignores_frontier_when_plan_fits(device, placement):
    # The full plan fits at either frontier; under L1 pressure the key steps with the CB depth by design.
    shape, out_shape, dtype = [12, 100], [12, 10, 10], ttnn.bfloat16
    xt = _to_device(device, _make_input(shape, dtype), dtype, ttnn.ROW_MAJOR_LAYOUT, placement)
    golden = ttnn.to_torch(_force_native(xt, out_shape))
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, out_shape)))
    entries_before = device.num_program_cache_entries()
    # Trace capture compiles nothing it has not already cached, so a call whose key moves with the L1 frontier
    # cannot be traced. This tensor moves the frontier by many pages per bank.
    resident = ttnn.from_torch(
        torch.zeros([1, 1, 256, 4096], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    try:
        assert_equal(golden, ttnn.to_torch(_force_codegen(xt, out_shape)))
        msg = "the program key moved with the L1 frontier, so the op cannot be traced"
        assert device.num_program_cache_entries() == entries_before, msg
    finally:
        ttnn.deallocate(resident)


# (id, in shape, out shape, dtype): row-major programs beyond the gate's stick, slab and dividing-slab cases.
_RM_PATHS = [
    ("rm_merge_uint32", [256, 2, 64], [256, 128], ttnn.uint32),
    ("rm_arbitrary_fp32", [64, 97], [97, 64], ttnn.float32),
]


@pytest.mark.parametrize("mem", [_DRAM, _L1], ids=["dram", "l1"])
@pytest.mark.parametrize("case,in_shape,out_shape,dtype", _RM_PATHS, ids=[p[0] for p in _RM_PATHS])
def test_reshape_codegen_rm_bit_exact(device, case, in_shape, out_shape, dtype, mem):
    _check_bit_exact(device, in_shape, out_shape, dtype, ttnn.ROW_MAJOR_LAYOUT, mem, mem)


_RM_SLAB = next(p[1:] for p in _PATHS if p[0] == "rm_arbitrary_bf16")


def test_reshape_codegen_rm_slab_sub_core_grids(device):
    _check_sub_core_grids(device, *_RM_SLAB)


def test_reshape_codegen_rm_slab_cache_hit_new_addresses(device):
    _check_cache_hit_new_addresses(device, *_RM_SLAB)


# (id, in shape, out shape, dtype, layout, expected route), as in the gate's _ROUTES.
_RM_ROUTES = [
    # The gate's demoted 256 B merge in uint16, which native rejects.
    ("rm_merge_uint16", *_rm_aligned_pair(256, merge=True), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, "codegen"),
    # A demoted merge whose stick program does not fit L1 runs the slab program, which beats native.
    ("rm_merge_slab", [64, 50001], [32, 100002], ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, "codegen"),
    # Widths that do not divide are demoted only when the slab program needs more L1 than native.
    ("rm_arbitrary", [64, 97], [97, 64], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "codegen"),
]


@pytest.mark.parametrize("case,in_shape,out_shape,dtype,layout,route", _RM_ROUTES, ids=[r[0] for r in _RM_ROUTES])
def test_reshape_codegen_rm_routing(device, case, in_shape, out_shape, dtype, layout, route):
    _check_routing(device, in_shape, out_shape, dtype, layout, route)


# (id, source sticks as a multiple of the grid plus an offset, pieces per source stick, expected route). An L1 split of
# 3-element pieces (6 or 12 B), a size where codegen moves one source stick faster than native.
_PAIRED_ROUTES = [
    # One source stick per core: codegen's single pass beats native's single kernel.
    ("one_per_core", 1, 0, 8, "codegen"),
    # Two per core: native moves one on each of its kernels, codegen both through one pipeline.
    ("two_per_core", 1, 3, 8, "native"),
    ("two_per_core_full", 2, 0, 8, "native"),
    # Three per core: native's busier kernel moves two and codegen catches up.
    ("three_per_core", 2, 1, 8, "codegen"),
    # Enough pieces per stick that codegen's cheaper repack pays for the second stick.
    ("two_per_core_long", 1, 3, 43, "codegen"),
]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("case,grids,extra,ratio,route", _PAIRED_ROUTES, ids=[r[0] for r in _PAIRED_ROUTES])
def test_reshape_codegen_routing_paired_l1_split(device, case, grids, extra, ratio, route, dtype):
    grid = device.compute_with_storage_grid_size()
    sticks = grids * grid.x * grid.y + extra
    in_shape, out_shape = [sticks, ratio * 3], [sticks, ratio, 3]
    source = _make(in_shape, dtype)
    x = _to_device(device, source, dtype, ttnn.ROW_MAJOR_LAYOUT, _L1)
    expected_leg = _force_codegen if route == "codegen" else _force_native
    _assert_bits(source.reshape(out_shape), ttnn.to_torch(expected_leg(x, out_shape, memory_config=_L1)), dtype)
    entries = device.num_program_cache_entries()
    out = ttnn.reshape(x, out_shape, memory_config=_L1)
    _assert_bits(source.reshape(out_shape), ttnn.to_torch(out), dtype)
    assert device.num_program_cache_entries() == entries, f"auto did not take {route}"


@pytest.mark.parametrize("mem", [_DRAM, _L1], ids=["dram", "l1"])
@pytest.mark.parametrize(
    "path,in_shape,out_shape,dtype,layout",
    [p for p in _PATHS if p[0] in ("rm_split", "rm_arbitrary_bf16")],
    ids=["rm_split", "rm_arbitrary_bf16"],
)
def test_reshape_codegen_rm_cache_hit_changed_free_l1(device, path, in_shape, out_shape, dtype, layout, mem):
    # Row-major buffers are sized from the L1 free at program creation, so the same call under less free L1 must not
    # reuse a program planned for more.
    source = _make(in_shape, dtype)
    expected = source.reshape(out_shape)
    x = _to_device(device, source, dtype, layout, mem)
    out = _force_codegen(x, out_shape, memory_config=mem)
    _assert_bits(expected, ttnn.to_torch(out), dtype)
    ttnn.deallocate(out)
    for bytes_per_core in (256 * 1024, 512 * 1024):
        hog = _l1_hog(device, bytes_per_core)
        try:
            out = _force_codegen(x, out_shape, memory_config=mem)
            _assert_bits(expected, ttnn.to_torch(out), dtype)
            ttnn.deallocate(out)
        finally:
            ttnn.deallocate(hog)
    # With the hogs freed the first plan comes back and hits its program.
    grown = device.num_program_cache_entries()
    out = _force_codegen(x, out_shape, memory_config=mem)
    _assert_bits(expected, ttnn.to_torch(out), dtype)
    assert device.num_program_cache_entries() == grown


# (id, shapes for a grid of N cores, L1 left free per bank in KiB at each step down). Each step must build a smaller
# program than the one before.
_RM_SHRINK = [
    # Coalesced 2 KB -> 16 KB merge, four new sticks per core: the full ring takes 64 KiB, the reduced ring of one
    # pushed stick 32 KiB, and the slab program at batch 1 under 13 KiB.
    ("coalesced_merge", lambda n: ([n * 32, 1024], [n * 4, 8192]), [48, 20]),
    # Non-aligned 80 B -> 40 B split, 64 source sticks per core: the full ring takes 12 KiB beside up to 1.5 KiB of
    # scratch, so 5 KiB free holds a 16-push ring.
    ("nonaligned_split", lambda n: ([n * 64, 40], [n * 64, 2, 20]), [5]),
]


@pytest.mark.parametrize("case,shapes,free_kib", _RM_SHRINK, ids=[c[0] for c in _RM_SHRINK])
def test_reshape_codegen_rm_shrinks_plan_under_l1_pressure(device, case, shapes, free_kib):
    grid = device.compute_with_storage_grid_size()
    in_shape, out_shape = shapes(grid.x * grid.y)
    dtype = ttnn.bfloat16
    source = _make(in_shape, dtype)
    expected = source.reshape(out_shape)
    x = _to_device(device, source, dtype, ttnn.ROW_MAJOR_LAYOUT, _DRAM)
    _assert_bits(expected, ttnn.to_torch(_force_codegen(x, out_shape, memory_config=_DRAM)), dtype)
    for free in free_kib:
        hog = _l1_hog_leaving(device, free * 1024)
        try:
            before = device.num_program_cache_entries()
            _assert_bits(expected, ttnn.to_torch(_force_codegen(x, out_shape, memory_config=_DRAM)), dtype)
            assert device.num_program_cache_entries() == before + 1, f"{free} KiB free did not build a smaller program"
        finally:
            ttnn.deallocate(hog)
    # With the hog freed the full plan comes back and hits its program.
    before = device.num_program_cache_entries()
    _assert_bits(expected, ttnn.to_torch(_force_codegen(x, out_shape, memory_config=_DRAM)), dtype)
    assert device.num_program_cache_entries() == before


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
# Native's row-major factory raises this when free L1 holds not even a one-slot destination ring.
_NO_STAGING = "RM reshape dest staging does not fit in L1"


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
    logger.info(f'[EXPECTED_ERROR BEGIN] RuntimeError message="{_CLASH}" or "{_NO_STAGING}"')
    try:
        # Native's program hash omits the ring depth it picks from free L1, so a cached program would hold the depth
        # of an earlier, larger probe.
        device.clear_program_cache()
        ttnn.deallocate(_force_native(x, out_shape, memory_config=_DRAM))
        return True
    except RuntimeError as error:
        if _CLASH not in str(error) and _NO_STAGING not in str(error):
            raise
        return False
    finally:
        logger.info(f'[EXPECTED_ERROR END] RuntimeError message="{_CLASH}" or "{_NO_STAGING}"')
        ttnn.deallocate(hog)


# (id, in shape, out shape, dtype, layout, free L1 per bank native is known to run with). Routed to codegen. TILE
# falls back to the segment map's smallest plan whenever no larger plan fits; row-major routes only shapes whose
# smallest plan fits native's smallest scratch.
_L1_FLOOR = [
    ("tile_rows_bf16", [1, 256, 4096], [1, 256, 32, 128], ttnn.bfloat16, ttnn.TILE_LAYOUT, 32 * 1024),
    ("mapped_fp32", [1, 1, 32, 100], [1, 1, 100, 32], ttnn.float32, ttnn.TILE_LAYOUT, 32 * 1024),
    ("rm_arbitrary_bf16", [64, 97], [97, 64], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, 32 * 1024),
]


@pytest.mark.parametrize("case,in_shape,out_shape,dtype,layout,ceiling", _L1_FLOOR, ids=[c[0] for c in _L1_FLOOR])
def test_reshape_codegen_runs_at_native_l1_floor(device, case, in_shape, out_shape, dtype, layout, ceiling):
    source = _make(in_shape, dtype)
    x = _to_device(device, source, dtype, layout)
    assert _native_runs_with(device, x, out_shape, ceiling), f"native does not run with {ceiling} B free"
    # The edge of native's L1 fit, to the hog's granularity: it fails at `low` and runs at `high`. Native's row-major
    # ring depth follows free L1, so its fit need not be monotone; any edge it runs at must hold the routed program.
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
