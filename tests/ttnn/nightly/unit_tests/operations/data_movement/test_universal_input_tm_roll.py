# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Universal I/O tests for ttnn.roll — comprehensive sharded matrix.

Covers:
  - ROW_MAJOR interleaved (DRAM/L1) — backward compatibility
  - TILE interleaved (DRAM/L1)
  - HEIGHT / WIDTH / BLOCK sharded, ROW_MAJOR (native gather kernel)
  - HEIGHT / WIDTH / BLOCK sharded, TILE tile-aligned (native whole-tile gather)
  - HEIGHT / WIDTH / BLOCK sharded, TILE non-tile-aligned (sharded untilize→roll→tilize)
  - Multi-dim rolls, higher-dim (batch/channel) rolls, last-dim within-row rolls
  - DRAM-sharded ROW_MAJOR: full-shard L1 staging (read DRAM→L1, assemble, write L1→DRAM)
  - DRAM-sharded TILE: per-tile NOC read/write (tile-size naturally DRAM-aligned)
  - Program-cache hash correctness: distinct shifts produce distinct programs
  - Optional output memory_config parameter (sharded → interleaved and vice versa)
  - COL_MAJOR shard orientation (HEIGHT / WIDTH / BLOCK, including genuinely 2D core grids)
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

_PCC = 0.9999

# Largest N for which arange(N) is exact in each dtype, i.e. how big a distinct-per-element fill can
# get before values start aliasing. bfloat16 has 8 significand bits, float32 has 24.
_EXACT_ARANGE_LIMIT = {torch.bfloat16: 256, torch.float32: 2**24}
_TTNN_DTYPE = {torch.bfloat16: ttnn.bfloat16, torch.float32: ttnn.float32}


# ─── shard-config helpers ─────────────────────────────────────────────────────


def _explicit_height_shard(device, ncores, sh, sw):
    compute_grid = device.compute_with_storage_grid_size()
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has {compute_grid.x * compute_grid.y} cores, test needs {ncores}")
    spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


def _explicit_width_shard(device, ncores, sh, sw):
    compute_grid = device.compute_with_storage_grid_size()
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has {compute_grid.x * compute_grid.y} cores, test needs {ncores}")
    spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)


def _explicit_block_shard(device, grid_y, grid_x, sh, sw):
    compute_grid = device.compute_with_storage_grid_size()
    if grid_y > compute_grid.y or grid_x > compute_grid.x:
        pytest.skip(f"Device grid ({compute_grid.y}x{compute_grid.x}) too small for {grid_y}x{grid_x}")
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)


# ─── runner ───────────────────────────────────────────────────────────────────


def run_roll(device, torch_input, layout, mem_config, shifts, dims):
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_config
    )
    ttnn_output = ttnn.roll(ttnn_input, list(shifts), list(dims))
    got = ttnn.to_torch(ttnn_output.cpu())
    ref = torch.roll(torch_input, list(shifts), list(dims))
    assert_with_pcc(ref.float(), got.float(), _PCC)


def run_roll_exact(
    device, shape, sh, sw, grid_x, grid_y, tensor_layout, orientation, shifts, dims, layout, dtype=torch.bfloat16
):
    """Roll on an explicit grid_x-by-grid_y core grid, compared bit-exactly.

    Every element gets a distinct value, so a gather that reads the right offset from the *wrong*
    core cannot coincide with the expected result. The fill has to stay exact in `dtype` for that
    to hold, which caps the element count (see `_EXACT_ARANGE_LIMIT`) — bfloat16 suffices for the
    small row-major shapes and keeps the 2-byte `cell_size` / `row_pitch_bytes` pitch under test,
    while a tilized shape needs float32 simply to fit.

    Also asserts the native sharded factory really ran. `native_ok` in `roll.cpp` is
    shape-dependent, and every sharded fallback reshards back to the input's config, so the output
    memory config alone would not reveal a config that had stopped reaching the factory under test.
    """
    compute_grid = device.compute_with_storage_grid_size()
    if grid_x > compute_grid.x or grid_y > compute_grid.y:
        pytest.skip(f"Device grid ({compute_grid.x}x{compute_grid.y}) too small for {grid_x}x{grid_y}")
    numel = int(torch.tensor(shape).prod())
    limit = _EXACT_ARANGE_LIMIT[dtype]
    assert numel <= limit, f"{numel} elements would alias in {dtype} (limit {limit}); fill must stay exact"
    torch_input = torch.arange(numel, dtype=torch.float32).reshape(shape).to(dtype)
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
        (sh, sw),
        orientation,
    )
    mem_config = ttnn.MemoryConfig(tensor_layout, ttnn.BufferType.L1, spec)
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=_TTNN_DTYPE[dtype], layout=layout, device=device, memory_config=mem_config
    )

    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        ttnn_output = ttnn.roll(ttnn_input, list(shifts), list(dims))
    finally:
        captured = ttnn.graph.end_graph_capture()
    # `roll.cpp` dispatches one roll_sharded per non-zero shift, so don't pin the count — the point
    # is that nothing *else* ran. Both sharded fallbacks would add device ops here: the TILE one an
    # untilize/tilize pair, the RM one an interleaved slice/concat round-trip.
    device_ops = [n for n in ttnn.graph.extract_calltrace(captured) if n.endswith("DeviceOperation")]
    assert device_ops and set(device_ops) == {"RollDeviceOperation"}, (
        f"expected only the native sharded roll factory, got {device_ops} — this config stopped "
        f"reaching the code path this test covers"
    )
    assert ttnn_output.memory_config() == mem_config, (
        f"native sharded roll must preserve the input memory config, "
        f"got {ttnn_output.memory_config()} vs {mem_config}"
    )

    got = ttnn.to_torch(ttnn_output.cpu())
    ref = torch.roll(torch_input, list(shifts), list(dims))
    if not torch.equal(ref.float(), got.float()):
        # Name the configuration and a few offending elements: a wrong shard -> core map fails on
        # some grid shapes and not others, so the grid and orientation are the diagnostic.
        bad = (ref.float() != got.float()).nonzero()
        sample = "; ".join(
            f"{tuple(idx.tolist())} expected {ref[tuple(idx.tolist())].item():g}, "
            f"got {got[tuple(idx.tolist())].item():g}"
            for idx in bad[:4]
        )
        pytest.fail(
            f"roll differs from torch.roll in {len(bad)}/{ref.numel()} elements\n"
            f"  shape={list(shape)} shard=({sh}, {sw}) grid={grid_x}x{grid_y} dtype={dtype}\n"
            f"  {tensor_layout} {orientation} {layout}\n"
            f"  shifts={list(shifts)} dims={list(dims)}\n"
            f"  first mismatches: {sample}"
        )


# ─── DRAM / L1 interleaved — backward compatibility ──────────────────────────


@pytest.mark.parametrize(
    "shape,shifts,dims,layout,mem_config",
    [
        ([1, 1, 4, 8], [2], [3], ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ([6, 4, 5, 1], [1, -2], [0, 2], ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ([4, 4, 8, 8], [3], [1], ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 32, 64], [16], [3], ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ([2, 1, 64, 64], [32, -32], [2, 3], ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 4, 8], [2], [3], ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG),
        ([1, 2, 64, 64], [32, -16], [2, 3], ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG),
    ],
)
def test_roll_interleaved(device, shape, shifts, dims, layout, mem_config):
    torch.manual_seed(0)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), layout, mem_config, shifts, dims)


# ─── HEIGHT_SHARDED + ROW_MAJOR (native gather kernel) ───────────────────────


@pytest.mark.parametrize(
    "shape,ncores,sh,sw,shifts,dims",
    [
        # last-dim within-row rotation
        ([1, 1, 4, 8], 4, 1, 8, [2], [3]),
        ([1, 1, 8, 16], 4, 2, 16, [4], [3]),
        ([1, 1, 8, 16], 4, 2, 16, [7], [3]),  # non-power-of-2 shift
        # height-dim page permutation
        ([1, 1, 8, 16], 4, 2, 16, [2], [2]),
        ([1, 1, 16, 8], 4, 4, 8, [3], [2]),
        # batch-dim roll
        ([2, 2, 8, 8], 4, 8, 8, [1], [0]),  # total_rows=32, sh=32//4=8
        ([4, 1, 4, 8], 4, 4, 8, [2], [0]),
        # multi-dim roll
        ([1, 1, 8, 16], 4, 2, 16, [2, 4], [2, 3]),
        # negative shift
        ([1, 1, 8, 16], 4, 2, 16, [-3], [3]),
    ],
)
def test_roll_height_sharded_row_major(device, shape, ncores, sh, sw, shifts, dims):
    torch.manual_seed(1)
    mem_config = _explicit_height_shard(device, ncores, sh, sw)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT, mem_config, shifts, dims)


# ─── HEIGHT_SHARDED + TILE, tile-aligned shifts (native whole-tile gather) ───


@pytest.mark.parametrize(
    "shape,ncores,sh,sw,shifts,dims",
    [
        ([1, 1, 64, 32], 2, 32, 32, [32], [3]),
        ([1, 1, 64, 64], 2, 32, 64, [32], [2]),
        ([1, 1, 128, 64], 4, 32, 64, [64], [2]),
        ([1, 1, 64, 128], 2, 32, 128, [64], [3]),
        ([1, 1, 64, 64], 2, 32, 64, [32, 32], [2, 3]),  # multi-dim tile-aligned
    ],
)
def test_roll_height_sharded_tile_aligned(device, shape, ncores, sh, sw, shifts, dims):
    torch.manual_seed(2)
    mem_config = _explicit_height_shard(device, ncores, sh, sw)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.TILE_LAYOUT, mem_config, shifts, dims)


# ─── HEIGHT_SHARDED + TILE, non-tile-aligned (sharded untilize→roll→tilize) ──


@pytest.mark.parametrize(
    "shape,ncores,sh,sw,shifts,dims",
    [
        ([1, 1, 64, 64], 2, 32, 64, [5], [3]),
        ([1, 1, 64, 64], 2, 32, 64, [7], [2]),
        ([1, 1, 128, 64], 4, 32, 64, [13], [3]),
    ],
)
def test_roll_height_sharded_tile_non_aligned(device, shape, ncores, sh, sw, shifts, dims):
    torch.manual_seed(3)
    mem_config = _explicit_height_shard(device, ncores, sh, sw)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.TILE_LAYOUT, mem_config, shifts, dims)


# ─── WIDTH_SHARDED + ROW_MAJOR (native gather kernel, cross-shard boundary) ──


@pytest.mark.parametrize(
    "shape,ncores,sh,sw,shifts,dims",
    [
        # cross-shard last-dim roll
        ([1, 1, 1, 32], 2, 1, 16, [8], [3]),
        ([1, 1, 1, 64], 4, 1, 16, [16], [3]),
        ([1, 1, 1, 32], 2, 1, 16, [7], [3]),  # non-power-of-2
        # height-dim page permutation
        ([1, 1, 4, 32], 2, 4, 16, [2], [2]),
        ([1, 1, 8, 32], 2, 8, 16, [3], [2]),
        # negative shift
        ([1, 1, 1, 64], 4, 1, 16, [-5], [3]),
    ],
)
def test_roll_width_sharded_row_major(device, shape, ncores, sh, sw, shifts, dims):
    torch.manual_seed(4)
    mem_config = _explicit_width_shard(device, ncores, sh, sw)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT, mem_config, shifts, dims)


# ─── WIDTH_SHARDED + TILE, tile-aligned (native whole-tile gather) ────────────


@pytest.mark.parametrize(
    "shape,ncores,sh,sw,shifts,dims",
    [
        ([1, 1, 32, 64], 2, 32, 32, [32], [3]),
        ([1, 1, 32, 128], 4, 32, 32, [32], [3]),
        ([1, 1, 64, 128], 4, 64, 32, [64], [3]),
    ],
)
def test_roll_width_sharded_tile_aligned(device, shape, ncores, sh, sw, shifts, dims):
    torch.manual_seed(5)
    mem_config = _explicit_width_shard(device, ncores, sh, sw)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.TILE_LAYOUT, mem_config, shifts, dims)


# ─── WIDTH_SHARDED + TILE, non-tile-aligned (sharded untilize→roll→tilize) ───


@pytest.mark.parametrize(
    "shape,ncores,sh,sw,shifts,dims",
    [
        ([1, 1, 32, 128], 4, 32, 32, [13], [3]),
        ([1, 1, 32, 128], 4, 32, 32, [7], [3]),
    ],
)
def test_roll_width_sharded_tile_non_aligned(device, shape, ncores, sh, sw, shifts, dims):
    torch.manual_seed(6)
    mem_config = _explicit_width_shard(device, ncores, sh, sw)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.TILE_LAYOUT, mem_config, shifts, dims)


# ─── BLOCK_SHARDED + ROW_MAJOR (native gather kernel) ────────────────────────


@pytest.mark.parametrize(
    "shape,grid_y,grid_x,sh,sw,shifts,dims",
    [
        # last-dim within-row rotation
        ([1, 1, 4, 8], 2, 2, 2, 4, [2], [3]),
        ([1, 1, 8, 16], 2, 2, 4, 8, [4], [3]),
        ([1, 1, 8, 16], 2, 2, 4, 8, [3], [3]),  # non-power-of-2
        # height-dim roll
        ([1, 1, 8, 16], 2, 2, 4, 8, [2], [2]),
        ([1, 1, 16, 16], 2, 2, 8, 8, [4], [2]),
        # negative shift
        ([1, 1, 8, 16], 2, 2, 4, 8, [-3], [3]),
    ],
)
def test_roll_block_sharded_row_major(device, shape, grid_y, grid_x, sh, sw, shifts, dims):
    torch.manual_seed(7)
    mem_config = _explicit_block_shard(device, grid_y, grid_x, sh, sw)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT, mem_config, shifts, dims)


# ─── BLOCK_SHARDED + TILE, tile-aligned (native whole-tile gather) ────────────


@pytest.mark.parametrize(
    "shape,grid_y,grid_x,sh,sw,shifts,dims",
    [
        ([1, 1, 64, 64], 2, 2, 32, 32, [32], [3]),
        ([1, 1, 64, 128], 2, 2, 32, 64, [32], [2]),
        ([1, 1, 128, 64], 2, 2, 64, 32, [64], [2]),
        ([1, 1, 64, 128], 2, 2, 32, 64, [64], [3]),
        ([1, 1, 64, 64], 2, 2, 32, 32, [32, 32], [2, 3]),  # multi-dim tile-aligned
    ],
)
def test_roll_block_sharded_tile_aligned(device, shape, grid_y, grid_x, sh, sw, shifts, dims):
    torch.manual_seed(8)
    mem_config = _explicit_block_shard(device, grid_y, grid_x, sh, sw)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.TILE_LAYOUT, mem_config, shifts, dims)


# ─── BLOCK_SHARDED + TILE, non-tile-aligned (sharded untilize→roll→tilize) ───


@pytest.mark.parametrize(
    "shape,grid_y,grid_x,sh,sw,shifts,dims",
    [
        ([1, 1, 64, 64], 2, 2, 32, 32, [5], [3]),
        ([1, 1, 64, 64], 2, 2, 32, 32, [7], [2]),
        ([1, 1, 64, 128], 2, 2, 32, 64, [13], [3]),
    ],
)
def test_roll_block_sharded_tile_non_aligned(device, shape, grid_y, grid_x, sh, sw, shifts, dims):
    torch.manual_seed(9)
    mem_config = _explicit_block_shard(device, grid_y, grid_x, sh, sw)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.TILE_LAYOUT, mem_config, shifts, dims)


# ─── DRAM-sharded inputs: native support via DRAM bank NOC transfers ─────────


@pytest.mark.parametrize(
    "shape,sh,sw,shifts,dims",
    [
        ([1, 1, 8, 16], 2, 16, [4], [3]),  # last-dim within-row rotation
        ([1, 1, 8, 16], 2, 16, [2], [2]),  # height-dim page permutation
        ([1, 1, 16, 32], 4, 32, [8], [3]),  # larger shard
    ],
)
def test_roll_dram_sharded_row_major_native(device, shape, sh, sw, shifts, dims):
    """DRAM-sharded ROW_MAJOR roll — mode 2: full-shard L1 staging.
    Reads entire source shard from DRAM into L1, assembles the rolled result in L1 via
    element-level local copies, then writes the complete shard from L1 back to DRAM.
    All DRAM NOC transfers are shard-sized (32-byte aligned) — no sub-alignment issue.
    """
    torch.manual_seed(10)
    compute_grid = device.compute_with_storage_grid_size()
    total_h = shape[0] * shape[1] * shape[2]
    ncores = total_h // sh
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has insufficient cores ({ncores} needed)")
    shard_spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        [sh, sw],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, shard_spec)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT, mem_config, shifts, dims)


@pytest.mark.parametrize(
    "shape,sh,sw,shifts,dims",
    [
        ([1, 1, 64, 64], 32, 64, [32], [2]),  # height-dim tile permutation
        ([1, 1, 64, 128], 32, 128, [64], [3]),  # last-dim tile rotation
    ],
)
def test_roll_dram_sharded_tile_native(device, shape, sh, sw, shifts, dims):
    """DRAM-sharded TILE roll — mode 1: per-tile NOC read+write.
    Tiles are 2048 bytes (bf16), naturally satisfying the 32-byte DRAM NOC write
    alignment requirement — no full-shard staging needed.
    """
    torch.manual_seed(10)
    compute_grid = device.compute_with_storage_grid_size()
    total_h = shape[0] * shape[1] * shape[2]
    ncores = total_h // sh
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has insufficient cores ({ncores} needed)")
    shard_spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        [sh, sw],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, shard_spec)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.TILE_LAYOUT, mem_config, shifts, dims)


# ─── Program-cache hash: distinct shifts must produce distinct programs ──────


@pytest.mark.parametrize("shift", [2, 5, 7])
def test_roll_program_cache_distinct_shifts(device, shift):
    """Rolls with different shifts on the same shape must produce correct results."""
    torch.manual_seed(11)
    shape = [1, 1, 8, 16]
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        [2, 16],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT, mem_config, [shift], [3])


# ─── Optional output memory_config ──────────────────────────────────────────


def test_roll_output_memory_config_dram(device):
    """Roll a sharded input and request DRAM interleaved output."""
    torch.manual_seed(12)
    shape = [1, 1, 8, 16]
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        [2, 16],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    x = torch.randn(shape, dtype=torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_mc)
    out = ttnn.roll(t, [3], [3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert out.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert_with_pcc(torch.roll(x, [3], [3]).float(), ttnn.to_torch(out.cpu()).float(), _PCC)


def test_roll_output_memory_config_l1_interleaved(device):
    """Roll an interleaved input and request L1 interleaved output."""
    torch.manual_seed(13)
    shape = [1, 1, 32, 64]
    x = torch.randn(shape, dtype=torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.roll(t, [16], [3], memory_config=ttnn.L1_MEMORY_CONFIG)
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    assert_with_pcc(torch.roll(x, [16], [3]).float(), ttnn.to_torch(out.cpu()).float(), _PCC)


# ─── COL_MAJOR shard orientation ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape,sh,sw,shifts,dims,layout",
    [
        ([1, 1, 8, 16], 2, 16, [2], [3], ttnn.ROW_MAJOR_LAYOUT),  # last-dim rotation
        ([1, 1, 8, 16], 2, 16, [2], [2], ttnn.ROW_MAJOR_LAYOUT),  # height-dim permutation
        ([1, 1, 64, 64], 32, 64, [32], [3], ttnn.TILE_LAYOUT),  # tile-aligned, last-dim; sh=32 for tile alignment
        ([1, 1, 64, 64], 32, 64, [32], [2], ttnn.TILE_LAYOUT),  # tile-aligned, height-dim
    ],
)
def test_roll_col_major_height_sharded(device, shape, sh, sw, shifts, dims, layout):
    """HEIGHT_SHARDED with COL_MAJOR orientation — native kernel.

    `num_cores_to_corerangeset` packs these shard counts onto a single row of cores, where the
    ROW_MAJOR and COL_MAJOR core enumerations coincide. The multi-row grids that actually
    distinguish them are covered by test_roll_col_major_height_sharded_multi_row_grid below.
    """
    torch.manual_seed(14)
    compute_grid = device.compute_with_storage_grid_size()
    total_rows = shape[0] * shape[1] * shape[2]
    ncores = total_rows // sh
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has insufficient cores for this test ({ncores} needed)")
    shard_spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        [sh, sw],
        ttnn.ShardOrientation.COL_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    run_roll(device, torch.randn(shape, dtype=torch.bfloat16), layout, mem_config, shifts, dims)


# On a single row (or single column) of cores the ROW_MAJOR and COL_MAJOR enumerations agree, so
# HEIGHT/WIDTH + COL_MAJOR only diverge once the grid is genuinely 2D. These tests pin the
# shard -> core map for each sharded layout: HEIGHT on a multi-row grid used to gather whole shards
# from the wrong core, WIDTH used to index past the end of the per-core transfer table and segfault
# inside the program factory, and BLOCK — the one layout whose shard grid matches its core grid —
# was and stays correct. The map itself is layout-independent, but each case is run untilized and
# tilized so a regression in the surrounding cell_h/cell_w and row-pitch arithmetic is caught too.


@pytest.mark.parametrize("grid_x,grid_y", [(8, 2), (2, 8), (4, 4)])
@pytest.mark.parametrize(
    "shifts,dims",
    [
        ([2], [2]),  # height dim: rows cross shard boundaries -> src core != dst core
        ([1], [1]),  # higher dim: whole cell-rows permute across shards
        ([3], [3]),  # last dim: rotates within each shard, src core == dst core
    ],
)
def test_roll_col_major_height_sharded_multi_row_grid(device, grid_x, grid_y, shifts, dims):
    """HEIGHT_SHARDED + COL_MAJOR on a 2D core grid: 16 shards of 2 rows over 16 cores."""
    run_roll_exact(
        device,
        [1, 2, 16, 8],
        sh=2,
        sw=8,
        grid_x=grid_x,
        grid_y=grid_y,
        tensor_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        orientation=ttnn.ShardOrientation.COL_MAJOR,
        shifts=shifts,
        dims=dims,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


@pytest.mark.parametrize("grid_x,grid_y", [(4, 2), (2, 4)])
@pytest.mark.parametrize("shifts,dims", [([32], [2]), ([32], [3])])
def test_roll_col_major_height_sharded_multi_row_grid_tile(device, grid_x, grid_y, shifts, dims):
    """Same shard -> core map, TILE layout: 8 shards of one tile-row over 8 cores.

    The mapping code is layout-independent, so this is the tilized twin of the test above rather
    than a separate code path — it guards against a tile-only regression in cell_h/cell_w handling.
    """
    run_roll_exact(
        device,
        [1, 1, 256, 64],
        sh=32,
        sw=64,
        grid_x=grid_x,
        grid_y=grid_y,
        tensor_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        orientation=ttnn.ShardOrientation.COL_MAJOR,
        shifts=shifts,
        dims=dims,
        layout=ttnn.TILE_LAYOUT,
        dtype=torch.float32,  # 8 tile-rows of shards cannot fit inside bfloat16's exact range
    )


@pytest.mark.parametrize("grid_x,grid_y", [(8, 1), (4, 2), (2, 4), (1, 8)])
@pytest.mark.parametrize(
    "layout,shape,sh,sw,dtype,shifts,dims",
    [
        # Row-major: 2-byte cells, shard row padded from 4B up to the 16B L1 alignment.
        (ttnn.ROW_MAJOR_LAYOUT, [1, 1, 8, 16], 8, 2, torch.bfloat16, [2], [3]),
        (ttnn.ROW_MAJOR_LAYOUT, [1, 1, 8, 16], 8, 2, torch.bfloat16, [1], [2]),
        # Tilized twin: whole-tile cells, and the width split puts shard columns on other cores.
        (ttnn.TILE_LAYOUT, [1, 1, 64, 256], 64, 32, torch.float32, [32], [3]),
        (ttnn.TILE_LAYOUT, [1, 1, 64, 256], 64, 32, torch.float32, [32], [2]),
    ],
)
def test_roll_col_major_width_sharded(device, grid_x, grid_y, layout, shape, sh, sw, dtype, shifts, dims):
    """WIDTH_SHARDED + COL_MAJOR: 8 shards spread across the width, over 8 cores."""
    run_roll_exact(
        device,
        shape,
        sh=sh,
        sw=sw,
        grid_x=grid_x,
        grid_y=grid_y,
        tensor_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        orientation=ttnn.ShardOrientation.COL_MAJOR,
        shifts=shifts,
        dims=dims,
        layout=layout,
        dtype=dtype,
    )


@pytest.mark.parametrize(
    "orientation,grid_x,grid_y",
    [
        # 2 shard-rows x 4 shard-cols. ROW_MAJOR sends height->y and width->x; COL_MAJOR swaps them.
        (ttnn.ShardOrientation.ROW_MAJOR, 4, 2),
        (ttnn.ShardOrientation.COL_MAJOR, 2, 4),
    ],
)
@pytest.mark.parametrize(
    "layout,shape,sh,sw,dtype,shifts,dims",
    [
        (ttnn.ROW_MAJOR_LAYOUT, [1, 1, 16, 16], 8, 4, torch.bfloat16, [2], [3]),
        (ttnn.ROW_MAJOR_LAYOUT, [1, 1, 16, 16], 8, 4, torch.bfloat16, [4], [2]),
        # Tilized twin: both shard dimensions are split, in whole-tile cells.
        (ttnn.TILE_LAYOUT, [1, 1, 64, 128], 32, 32, torch.float32, [32], [3]),
        (ttnn.TILE_LAYOUT, [1, 1, 64, 128], 32, 32, torch.float32, [32], [2]),
    ],
)
def test_roll_block_sharded_orientations(
    device, orientation, grid_x, grid_y, layout, shape, sh, sw, dtype, shifts, dims
):
    """BLOCK_SHARDED in both orientations still matches torch.roll exactly."""
    run_roll_exact(
        device,
        shape,
        sh=sh,
        sw=sw,
        grid_x=grid_x,
        grid_y=grid_y,
        tensor_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        orientation=orientation,
        shifts=shifts,
        dims=dims,
        layout=layout,
        dtype=dtype,
    )


# rd=(2,6) shard_h=4 shift=1 dim=2: 3-source case (past reader's 2-slot `src_base`) routed via interleaved round-trip.
def test_roll_dram_sharded_row_major_ge3_source_shards_routes_via_interleaved(device):
    torch.manual_seed(51213)
    shape = [1, 2, 6, 16]
    sh, sw = 4, 16
    total_h = shape[0] * shape[1] * shape[2]
    ncores = total_h // sh
    compute_grid = device.compute_with_storage_grid_size()
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has insufficient cores ({ncores} needed)")
    shard_spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        [sh, sw],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, shard_spec)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )
    out = ttnn.roll(tt_in, [1], [2])
    assert_with_pcc(torch.roll(x, [1], [2]).float(), ttnn.to_torch(out.cpu()).float(), _PCC)
