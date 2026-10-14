# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Reshape codegen port: routing, bit-exactness of the program, edge shapes, sub_core_grids and program-cache keys.

import pytest
import torch

import ttnn

_dm = ttnn._ttnn.operations.data_movement
_force_native = _dm.reshape_force_native
_force_codegen = _dm.reshape_force_codegen

_DRAM = ttnn.DRAM_MEMORY_CONFIG
_L1 = ttnn.L1_MEMORY_CONFIG
_SUB_CORE_GRIDS = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6))])

# NaN payloads, +-Inf, -0 and subnormals: values a compute-engine round trip would change.
_SPECIAL16 = [0x7FC0, 0x7FC1, 0xFFC0, 0x7F81, 0x7F80, 0xFF80, 0x8000, 0x0001, 0x007F, 0x8001, 0x0080]
_SPECIAL32 = [
    0x7FC00000,
    0x7FC00001,
    0xFFC00000,
    0x7F800001,
    0x7F800000,
    0xFF800000,
    0x80000000,
    0x00000001,
    0x007FFFFF,
    0x80000001,
    0x00800000,
]

# Float dtypes: (torch dtype handed to from_torch, integer view of its bits).
_FLOAT = {ttnn.bfloat16: (torch.bfloat16, torch.int16), ttnn.float32: (torch.float32, torch.int32)}
# Integer dtypes: full value range, carried in int64 so uint32 keeps its top bit.
_INT_RANGE = {ttnn.int32: (-(1 << 31), (1 << 31) - 1), ttnn.uint32: (0, (1 << 32) - 1), ttnn.uint16: (0, (1 << 16) - 1)}


def _make(shape, dtype):
    torch.manual_seed(0)
    if dtype in _INT_RANGE:
        low, high = _INT_RANGE[dtype]
        return torch.randint(low, high, shape, dtype=torch.int64).to(
            torch.int32 if dtype == ttnn.int32 else torch.int64
        )
    host, bits = _FLOAT[dtype]
    data = torch.randn(shape, dtype=torch.float32).to(host)
    width = 16 if bits == torch.int16 else 32
    special = [v - (1 << width) if v >= 1 << (width - 1) else v for v in (_SPECIAL16 if width == 16 else _SPECIAL32)]
    flat = data.view(bits).flatten()
    idx = torch.arange(0, flat.numel(), max(1, flat.numel() // (8 * len(special))))[: 8 * len(special)]
    flat[idx] = torch.tensor(special, dtype=bits).repeat(8)[: len(idx)]
    return flat.view(host).reshape(shape)


def _assert_bits(expected, actual, dtype):
    actual = actual.reshape(expected.shape)
    if dtype in _FLOAT:
        _, bits = _FLOAT[dtype]
        expected, actual = expected.contiguous().view(bits), actual.to(expected.dtype).contiguous().view(bits)
    else:
        expected, actual = expected.to(torch.int64), actual.to(torch.int64)
    bad = (expected != actual).sum().item()
    assert bad == 0, f"{bad} of {expected.numel()} datums differ"


def _to_device(device, source, dtype, layout, mem):
    return ttnn.from_torch(source, dtype=dtype, layout=layout, device=device, memory_config=mem)


# One case per program. Which program runs is the plan's choice for these shapes on the full grid.
_PATHS = [
    # Tile-row permutation: both widths multiples of 32.
    ("tile_rows", [1, 64, 1024], [1, 64, 8, 128], ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ("tile_rows_fp32", [1, 64, 1024], [1, 64, 8, 128], ttnn.float32, ttnn.TILE_LAYOUT),
    ("tile_rows_int32", [1, 64, 1024], [1, 64, 8, 128], ttnn.int32, ttnn.TILE_LAYOUT),
    ("tile_rows_uint32", [1, 64, 1024], [1, 64, 8, 128], ttnn.uint32, ttnn.TILE_LAYOUT),
    ("tile_rows_uint16", [1, 64, 1024], [1, 64, 8, 128], ttnn.uint16, ttnn.TILE_LAYOUT),
    # Row-major stick program: dividing widths.
    ("rm_split", [1, 256, 4096], [1, 256, 32, 128], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    # Row-major slab program: widths that do not divide, and dividing widths whose stick program does not fit L1.
    ("rm_arbitrary_bf16", [64, 97], [97, 64], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ("rm_slab_dividing", [2, 409600], [4, 204800], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
]


@pytest.mark.parametrize("mem", [_DRAM, _L1], ids=["dram", "l1"])
@pytest.mark.parametrize("path,in_shape,out_shape,dtype,layout", _PATHS, ids=[p[0] for p in _PATHS])
def test_reshape_codegen_bit_exact(device, path, in_shape, out_shape, dtype, layout, mem):
    source = _make(in_shape, dtype)
    x = _to_device(device, source, dtype, layout, mem)
    out = _force_codegen(x, out_shape, memory_config=mem)
    assert list(out.shape) == out_shape
    _assert_bits(source.reshape(out_shape), ttnn.to_torch(out), dtype)


@pytest.mark.parametrize(
    "path,in_shape,out_shape,dtype,layout",
    [p for p in _PATHS if p[0] in ("tile_rows", "rm_split")],
    ids=["tile_rows", "rm_split"],
)
def test_reshape_codegen_sub_core_grids(device, path, in_shape, out_shape, dtype, layout):
    source = _make(in_shape, dtype)
    x = _to_device(device, source, dtype, layout, _DRAM)
    out = _force_codegen(x, out_shape, memory_config=_DRAM, sub_core_grids=_SUB_CORE_GRIDS)
    _assert_bits(source.reshape(out_shape), ttnn.to_torch(out), dtype)
    # Auto keeps the caller's grid on the codegen program it just compiled.
    entries = device.num_program_cache_entries()
    out = ttnn.reshape(x, out_shape, memory_config=_DRAM, sub_core_grids=_SUB_CORE_GRIDS)
    _assert_bits(source.reshape(out_shape), ttnn.to_torch(out), dtype)
    assert device.num_program_cache_entries() == entries


# Forced tile rows on shapes whose tile padding, rank or outer dims differ from the common head split.
_EDGE = [
    ("h50", [3, 50, 64], [3, 100, 32]),
    ("h33", [7, 33, 64], [7, 66, 32]),
    ("h1", [5, 1, 128], [5, 4, 32]),
    ("rank2", [50, 64], [100, 32]),
    ("rank5", [2, 3, 1, 50, 64], [2, 3, 50, 2, 32]),
    ("odd_outer", [7, 32, 96], [3, 7, 32, 32]),
    ("outer_to_h", [3, 50, 64], [50, 3, 64]),
]
_EDGE_CASES = [(*e, ttnn.bfloat16) for e in _EDGE] + [
    (*e, dtype) for e in _EDGE if e[0] in ("h50", "rank5", "outer_to_h") for dtype in (ttnn.float32, ttnn.int32)
]


@pytest.mark.parametrize(
    "case,in_shape,out_shape,dtype", _EDGE_CASES, ids=[f"{c[0]}-{c[3].name.lower()}" for c in _EDGE_CASES]
)
def test_reshape_codegen_edge_shapes(device, case, in_shape, out_shape, dtype):
    source = _make(in_shape, dtype)
    x = _to_device(device, source, dtype, ttnn.TILE_LAYOUT, _DRAM)
    codegen = _force_codegen(x, out_shape, memory_config=_DRAM)
    native = _force_native(x, out_shape, memory_config=_DRAM)
    assert list(codegen.shape) == out_shape
    assert list(codegen.padded_shape) == list(native.padded_shape)
    _assert_bits(source.reshape(out_shape), ttnn.to_torch(codegen), dtype)
    # Native leaves tile padding unspecified unless pad_value is passed; codegen writes zeros there.
    padded = torch.zeros(list(codegen.padded_shape), dtype=source.dtype)
    padded[tuple(slice(0, d) for d in out_shape)] = source.reshape(out_shape)
    _assert_bits(padded, codegen.cpu().to_torch_with_padded_shape(), dtype)


# (id, in shape, out shape, dtype, layout, expected route). Row-major demotion is decided by stick sizes
# (rm_native_faster) and taken only where native runs and the stick program fits (demotion_applies). TILE demotion is
# decided by the tile plan (is_demoted); shapes the tile-rows program does not take run native.
_ROUTES = [
    # 512 B -> 256 B split: codegen keeps every short write in flight.
    ("rm_split", [512, 256], [1024, 128], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "codegen"),
    # 256 B -> 512 B DRAM merge: from 256 B native's two kernels read faster.
    ("rm_merge_demoted", [512, 128], [256, 256], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "native"),
    # Head split: batching cuts source reads 32x.
    ("tile_rows", [1, 256, 4096], [1, 256, 32, 128], ttnn.bfloat16, ttnn.TILE_LAYOUT, "codegen"),
    # 4-head merge: the cut is only 4x, but each output tile reads 24 fewer padded source tiles.
    ("tile_head_merge", [1, 512, 4, 256], [1, 512, 1024], ttnn.bfloat16, ttnn.TILE_LAYOUT, "codegen"),
    # Width change: tile rows saves only 2x the reads on far fewer cores.
    ("tile_width_native", [1, 1, 64, 96], [1, 1, 32, 192], ttnn.bfloat16, ttnn.TILE_LAYOUT, "native"),
    ("tile_width_uint16", [1, 1, 64, 96], [1, 1, 32, 192], ttnn.uint16, ttnn.TILE_LAYOUT, "codegen"),
    # A width that is not a multiple of 32 is not a tile-row permutation.
    ("tile_unaligned_width", [1, 1, 32, 100], [1, 1, 100, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, "native"),
]


@pytest.mark.parametrize("case,in_shape,out_shape,dtype,layout,route", _ROUTES, ids=[r[0] for r in _ROUTES])
def test_reshape_codegen_routing(device, case, in_shape, out_shape, dtype, layout, route):
    source = _make(in_shape, dtype)
    x = _to_device(device, source, dtype, layout, _DRAM)
    expected_leg = _force_codegen if route == "codegen" else _force_native
    _assert_bits(source.reshape(out_shape), ttnn.to_torch(expected_leg(x, out_shape, memory_config=_DRAM)), dtype)
    # The expected leg compiled its program; auto taking the other leg would compile a new one.
    entries = device.num_program_cache_entries()
    out = ttnn.reshape(x, out_shape, memory_config=_DRAM)
    _assert_bits(source.reshape(out_shape), ttnn.to_torch(out), dtype)
    assert device.num_program_cache_entries() == entries, f"auto did not take {route}"


@pytest.mark.parametrize(
    "path,in_shape,out_shape,dtype,layout",
    [p for p in _PATHS if p[0] in ("tile_rows", "rm_split")],
    ids=["tile_rows", "rm_split"],
)
def test_reshape_codegen_cache_hit_new_addresses(device, path, in_shape, out_shape, dtype, layout):
    source = _make(in_shape, dtype)
    x = _to_device(device, source, dtype, layout, _DRAM)
    first = _force_codegen(x, out_shape, memory_config=_DRAM)
    entries = device.num_program_cache_entries()
    # The first input and output stay allocated, so the second call runs on new addresses; a cache hit must
    # rebind them.
    other = torch.flip(source, dims=[0]) if source.shape[0] > 1 else torch.flip(source, dims=[-1])
    y = _to_device(device, other, dtype, layout, _DRAM)
    assert y.buffer_address() != x.buffer_address()
    second = _force_codegen(y, out_shape, memory_config=_DRAM)
    assert second.buffer_address() != first.buffer_address()
    assert device.num_program_cache_entries() == entries
    _assert_bits(source.reshape(out_shape), ttnn.to_torch(first), dtype)
    _assert_bits(other.reshape(out_shape), ttnn.to_torch(second), dtype)


def _l1_hog(device, bytes_per_core):
    grid = device.compute_with_storage_grid_size()
    rows = grid.x * grid.y * bytes_per_core // 2048
    return ttnn.from_torch(
        torch.zeros([rows, 1024], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_L1,
    )


@pytest.mark.parametrize("mem", [_DRAM, _L1], ids=["dram", "l1"])
@pytest.mark.parametrize(
    "path,in_shape,out_shape,dtype,layout",
    [p for p in _PATHS if p[0] == "tile_rows"],
    ids=["tile_rows"],
)
def test_reshape_codegen_program_key_stable_while_full_plan_fits(device, path, in_shape, out_shape, dtype, layout, mem):
    # The hog leaves room for the full plan, so the program and its key stay the same.
    source = _make(in_shape, dtype)
    expected = source.reshape(out_shape)
    x = _to_device(device, source, dtype, layout, mem)
    out = _force_codegen(x, out_shape, memory_config=mem)
    _assert_bits(expected, ttnn.to_torch(out), dtype)
    ttnn.deallocate(out)
    entries = device.num_program_cache_entries()
    hog = _l1_hog(device, 256 * 1024)
    out = _force_codegen(x, out_shape, memory_config=mem)
    _assert_bits(expected, ttnn.to_torch(out), dtype)
    assert device.num_program_cache_entries() == entries
    ttnn.deallocate(out)
    ttnn.deallocate(hog)


def _l1_hog_leaving(device, free_bytes_per_bank):
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    tiles_per_bank = (view.largest_contiguous_bytes_free_per_bank - free_bytes_per_bank) // (32 * 32 * 2)
    if tiles_per_bank <= 0:
        pytest.skip("L1 already has less free space than the test needs")
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, 1, 32, tiles_per_bank * view.num_banks * 32)),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        _L1,
    )


def test_reshape_codegen_shrinks_plan_under_l1_pressure(device):
    # Every output tile of this fp32 head split needs 32 source tiles, so no source cap shrinks the staged CB: the
    # full plan's buffers take 301 KiB per core double-buffered and 173 KiB single-buffered, and 240 KiB free fits only
    # the latter.
    in_shape, out_shape, dtype = [1, 256, 4096], [1, 256, 32, 128], ttnn.float32
    source = _make(in_shape, dtype)
    expected = source.reshape(out_shape)
    x = _to_device(device, source, dtype, ttnn.TILE_LAYOUT, _DRAM)
    _assert_bits(expected, ttnn.to_torch(_force_codegen(x, out_shape, memory_config=_DRAM)), dtype)
    entries = device.num_program_cache_entries()
    hog = _l1_hog_leaving(device, 240 * 1024)
    # The smaller plan is a new program, so it cannot reuse the full plan's cache entry.
    _assert_bits(expected, ttnn.to_torch(_force_codegen(x, out_shape, memory_config=_DRAM)), dtype)
    assert device.num_program_cache_entries() == entries + 1
    # Auto routes to the codegen program it just compiled; native would compile a new one.
    _assert_bits(expected, ttnn.to_torch(ttnn.reshape(x, out_shape, memory_config=_DRAM)), dtype)
    assert device.num_program_cache_entries() == entries + 1
    ttnn.deallocate(hog)
    # With the L1 free again the full plan fits, so the call hits the full plan's entry.
    _assert_bits(expected, ttnn.to_torch(_force_codegen(x, out_shape, memory_config=_DRAM)), dtype)
    assert device.num_program_cache_entries() == entries + 1


def test_reshape_codegen_routes_native_when_no_plan_fits_l1(device, expect_error):
    # The same head split's smallest plan takes 173 KiB per core, so with 128 KiB free no plan fits: auto must run
    # native, while the forced entry still refuses.
    in_shape, out_shape, dtype = [1, 256, 4096], [1, 256, 32, 128], ttnn.float32
    source = _make(in_shape, dtype)
    expected = source.reshape(out_shape)
    x = _to_device(device, source, dtype, ttnn.TILE_LAYOUT, _DRAM)
    hog = _l1_hog_leaving(device, 128 * 1024)
    _assert_bits(expected, ttnn.to_torch(ttnn.reshape(x, out_shape, memory_config=_DRAM)), dtype)
    with expect_error(RuntimeError, "no tile-row plan fits"):
        _force_codegen(x, out_shape, memory_config=_DRAM)
    ttnn.deallocate(hog)


_OUT_OF_SCOPE = [
    ("bfloat8_b", [1, 1, 64, 128], [1, 1, 128, 64], ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ("tile_unaligned_width", [1, 1, 32, 100], [1, 1, 100, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT),
]


@pytest.mark.parametrize("case,in_shape,out_shape,dtype,layout", _OUT_OF_SCOPE, ids=[c[0] for c in _OUT_OF_SCOPE])
def test_forced_codegen_refuses_out_of_scope_case(device, expect_error, case, in_shape, out_shape, dtype, layout):
    # The forced leg is compared against native, so outside its scope it must fail rather than fall back.
    x = ttnn.from_torch(torch.rand(in_shape), dtype=dtype, layout=layout, device=device)
    with expect_error(RuntimeError, "outside the codegen scope"):
        _force_codegen(x, out_shape)
