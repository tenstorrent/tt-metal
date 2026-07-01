# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Universal I/O tests for ttnn.repeat — edge-case matrix (transpose-style)."""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

_TTNN_TO_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.bfloat8_b: torch.bfloat16,
}

_TILE_HEIGHT = 32
_TILE_WIDTH = 32


def _round_up(x, mult):
    return ((x + mult - 1) // mult) * mult


def _padded_hw(shape, layout):
    """Return (total_height, width). For TILE layout, last-two dims tile-padded."""
    width = shape[-1]
    height = 1
    for d in shape[:-1]:
        height *= d
    if layout == ttnn.TILE_LAYOUT:
        # Pad H/W to tile multiples for TILE layout.
        height = (height // shape[-2]) * _round_up(shape[-2], _TILE_HEIGHT)
        width = _round_up(width, _TILE_WIDTH)
    return height, width


def _tile_align(shard_shape, layout):
    h, w = shard_shape
    if layout == ttnn.TILE_LAYOUT:
        return (_round_up(h, _TILE_HEIGHT), _round_up(w, _TILE_WIDTH))
    return (h, w)


def _height_shard_config(shape, device, num_cores=4, layout=ttnn.TILE_LAYOUT):
    """Height-sharded MemoryConfig."""
    compute_grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, compute_grid.x * compute_grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
    total_h, w = _padded_hw(shape, layout)
    shard_shape = _tile_align(((total_h + num_cores - 1) // num_cores, w), layout)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _width_shard_config(shape, device, num_cores=4, layout=ttnn.TILE_LAYOUT):
    """Width-sharded MemoryConfig."""
    compute_grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, compute_grid.x * compute_grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
    total_h, w = _padded_hw(shape, layout)
    shard_shape = _tile_align((total_h, (w + num_cores - 1) // num_cores), layout)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


def _block_shard_config(shape, device, layout=ttnn.TILE_LAYOUT):
    """Block-sharded MemoryConfig (2x2 grid)."""
    compute_grid = device.compute_with_storage_grid_size()
    grid_x = min(2, compute_grid.x)
    grid_y = min(2, compute_grid.y)
    total_h, w = _padded_hw(shape, layout)
    shard_shape = _tile_align(((total_h + grid_y - 1) // grid_y, (w + grid_x - 1) // grid_x), layout)
    # RM block-sharding requires width divisible by grid_x; skip if invalid.
    if layout == ttnn.ROW_MAJOR_LAYOUT and w % grid_x != 0:
        pytest.skip(f"RM block-sharding requires width {w} divisible by grid_x {grid_x}")
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)


def _explicit_block_shard_config(device, grid_y, grid_x, sh, sw):
    """Explicit block-shard grid; skips if device grid is too small."""
    compute_grid = device.compute_with_storage_grid_size()
    if grid_y > compute_grid.y or grid_x > compute_grid.x:
        pytest.skip(f"Device grid ({compute_grid.y}x{compute_grid.x}) too small for {grid_y}x{grid_x}")
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)


def _explicit_height_shard_config(device, ncores, sh, sw):
    """Explicit height-shard grid; skips if device has fewer cores."""
    compute_grid = device.compute_with_storage_grid_size()
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has {compute_grid.x * compute_grid.y} cores, test needs {ncores}")
    spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


def _explicit_width_shard_config(device, ncores, sh, sw):
    """Explicit width-shard grid; skips if device has fewer cores."""
    compute_grid = device.compute_with_storage_grid_size()
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has {compute_grid.x * compute_grid.y} cores, test needs {ncores}")
    spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)


L1_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
DRAM_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


def _interleaved(_d):
    return L1_INTERLEAVED


def _sharded_no_spec(memory_layout):
    """Sharded output MemoryConfig without explicit shard_spec."""
    return lambda _d: ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1)


def run_repeat_test(
    shape,
    repeat_shape,
    device,
    input_layout=ttnn.TILE_LAYOUT,
    input_mem_config=None,
    output_mem_config=None,
    dtype=ttnn.bfloat16,
    pcc=0.9999,
):
    """Run repeat and assert PCC + output memory_layout."""
    torch.manual_seed(12345)
    torch_dtype = _TTNN_TO_TORCH_DTYPE[dtype]
    x = torch.rand(shape, dtype=torch_dtype)

    if input_mem_config is None:
        input_mem_config = L1_INTERLEAVED

    ttnn_input = ttnn.from_torch(x, layout=input_layout, dtype=dtype, device=device, memory_config=input_mem_config)
    result = ttnn.repeat(ttnn_input, list(repeat_shape), memory_config=output_mem_config)

    actual = result.memory_config()
    if output_mem_config is not None:
        assert (
            actual.memory_layout == output_mem_config.memory_layout
        ), f"Expected output memory layout {output_mem_config.memory_layout}, got {actual.memory_layout}"
        if output_mem_config.memory_layout in (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ):
            assert (
                actual.shard_spec is not None
            ), "Sharded output requested but result has no shard_spec (silent fallback?)"
    elif input_mem_config.is_sharded():
        # Sharded input → inherit layout or fallback to interleaved (non-alignable shapes).
        assert actual.memory_layout in (
            input_mem_config.memory_layout,
            ttnn.TensorMemoryLayout.INTERLEAVED,
        ), f"Expected inherited {input_mem_config.memory_layout} or interleaved fallback, got {actual.memory_layout}"
        if actual.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            assert actual.shard_spec is not None, "Sharded output must carry a shard_spec"

    ref_repeat = list(repeat_shape)
    if len(ref_repeat) < x.dim():
        ref_repeat = [1] * (x.dim() - len(ref_repeat)) + ref_repeat
    ref = x.repeat(*ref_repeat)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    assert_with_pcc(ref.float(), got.float(), pcc)


# Native TILE sharded paths.
@pytest.mark.parametrize(
    "shape, repeat_shape, input_factory, output_factory, id_suffix",
    [
        # TILE HEIGHT, last-dim
        (
            (1, 1, 128, 64),
            (1, 1, 1, 2),
            lambda d: _height_shard_config((1, 1, 128, 64), d),
            _interleaved,
            "tile_height_last_dim",
        ),
        # TILE WIDTH, upper-dim
        (
            (1, 1, 64, 128),
            (2, 1, 1, 1),
            lambda d: _width_shard_config((1, 1, 64, 128), d),
            _interleaved,
            "tile_width_upper_dim",
        ),
        # TILE BLOCK, H-dim
        (
            (1, 1, 64, 64),
            (1, 1, 2, 1),
            lambda d: _block_shard_config((1, 1, 64, 64), d),
            _interleaved,
            "tile_block_h_dim",
        ),
        # TILE HEIGHT -> explicit sharded out
        (
            (1, 1, 128, 64),
            (1, 1, 1, 2),
            lambda d: _height_shard_config((1, 1, 128, 64), d),
            lambda d: _height_shard_config((1, 1, 128, 128), d),
            "tile_height_to_height_explicit",
        ),
    ],
)
def test_repeat_native_tile_sharded(shape, repeat_shape, input_factory, output_factory, id_suffix, device):
    in_mc = input_factory(device)
    out_mc = output_factory(device)
    run_repeat_test(
        shape,
        repeat_shape,
        device,
        input_layout=ttnn.TILE_LAYOUT,
        input_mem_config=in_mc,
        output_mem_config=out_mc,
        dtype=ttnn.bfloat16,
    )


# Native RM HEIGHT sharded.
@pytest.mark.parametrize(
    "shape, repeat_shape, id_suffix",
    [
        ((4, 32), (1, 2), "rm_height_last_dim"),
        ((4, 32), (2, 1), "rm_height_upper_dim"),
    ],
)
def test_repeat_native_rm_height_sharded(shape, repeat_shape, id_suffix, device):
    in_mc = _height_shard_config(shape if len(shape) == 4 else (1, 1, *shape), device, layout=ttnn.ROW_MAJOR_LAYOUT)
    run_repeat_test(
        shape,
        repeat_shape,
        device,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        input_mem_config=in_mc,
        output_mem_config=L1_INTERLEAVED,
        dtype=ttnn.bfloat16,
    )


# RM BLOCK/WIDTH + last-dim repeat: predicate forces composite (Fix 2 guard).
@pytest.mark.parametrize(
    "shape, repeat_shape, input_factory, output_factory, id_suffix",
    [
        # RM BLOCK -> interleaved
        (
            (4, 64),
            (1, 2),
            lambda d: _block_shard_config((1, 1, 4, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            _interleaved,
            "rm_block_last_dim_to_interleaved",
        ),
        # RM WIDTH -> interleaved
        (
            (4, 64),
            (1, 2),
            lambda d: _width_shard_config((1, 1, 4, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            _interleaved,
            "rm_width_last_dim_to_interleaved",
        ),
    ],
)
def test_repeat_rm_block_width_last_dim_composite(
    shape, repeat_shape, input_factory, output_factory, id_suffix, device
):
    in_mc = input_factory(device)
    out_mc = output_factory(device)
    run_repeat_test(
        shape,
        repeat_shape,
        device,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        input_mem_config=in_mc,
        output_mem_config=out_mc,
        dtype=ttnn.bfloat16,
    )


# Composite fallback (predicate rejects native path).
@pytest.mark.parametrize(
    "shape, repeat_shape, input_factory, output_factory, id_suffix",
    [
        # RM BLOCK upper-dim: not locally contained -> composite
        (
            (4, 64),
            (2, 1),
            lambda d: _block_shard_config((1, 1, 4, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            _interleaved,
            "rm_block_upper_dim_composite",
        ),
        # Multi-axis repeat on sharded input -> composite
        (
            (1, 1, 64, 64),
            (2, 1, 2, 1),
            lambda d: _block_shard_config((1, 1, 64, 64), d),
            _interleaved,
            "tile_multi_dim_composite",
        ),
    ],
)
def test_repeat_composite_fallback(shape, repeat_shape, input_factory, output_factory, id_suffix, device):
    in_mc = input_factory(device)
    out_mc = output_factory(device)
    is_tile = "tile" in id_suffix
    run_repeat_test(
        shape,
        repeat_shape,
        device,
        input_layout=ttnn.TILE_LAYOUT if is_tile else ttnn.ROW_MAJOR_LAYOUT,
        input_mem_config=in_mc,
        output_mem_config=out_mc,
        dtype=ttnn.bfloat16,
    )


# Sharded output synthesis: derived spec from sharded/interleaved input.
@pytest.mark.parametrize(
    "shape, repeat_shape, input_layout, in_mc_fn, out_layout",
    [
        pytest.param(
            (1, 1, 64, 128),
            (1, 1, 1, 2),
            ttnn.TILE_LAYOUT,
            lambda d: _width_shard_config((1, 1, 64, 128), d),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            id="tile_width_to_width",
        ),
        pytest.param(
            (1, 1, 128, 64),
            (1, 1, 1, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d: _height_shard_config((1, 1, 128, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            id="rm_height_to_height",
        ),
        pytest.param(
            (1, 1, 64, 32),
            (1, 1, 1, 2),
            ttnn.TILE_LAYOUT,
            lambda d: _explicit_height_shard_config(d, 2, 32, 32),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            id="tile_subtile_height_to_height",
        ),
    ],
)
def test_repeat_sharded_to_derived_spec(shape, repeat_shape, input_layout, in_mc_fn, out_layout, device):
    out_mc = _sharded_no_spec(out_layout)(device)
    run_repeat_test(
        shape,
        repeat_shape,
        device,
        input_layout=input_layout,
        input_mem_config=in_mc_fn(device),
        output_mem_config=out_mc,
    )


# Interleaved input → sharded output without shard_spec (TILE and RM, all layouts).
@pytest.mark.parametrize(
    "shape, repeat_shape, input_layout, memory_layout",
    [
        pytest.param(
            (1, 1, 64, 64), (1, 1, 1, 2), ttnn.TILE_LAYOUT, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, id="tile_H"
        ),
        pytest.param(
            (1, 1, 64, 64), (1, 1, 1, 2), ttnn.TILE_LAYOUT, ttnn.TensorMemoryLayout.WIDTH_SHARDED, id="tile_W"
        ),
        pytest.param(
            (1, 1, 64, 64), (1, 1, 1, 2), ttnn.TILE_LAYOUT, ttnn.TensorMemoryLayout.BLOCK_SHARDED, id="tile_B"
        ),
        pytest.param(
            (1, 1, 64, 128), (1, 1, 1, 2), ttnn.ROW_MAJOR_LAYOUT, ttnn.TensorMemoryLayout.BLOCK_SHARDED, id="rm_B"
        ),
        pytest.param(
            (1, 1, 64, 128), (1, 1, 1, 2), ttnn.ROW_MAJOR_LAYOUT, ttnn.TensorMemoryLayout.WIDTH_SHARDED, id="rm_W"
        ),
        pytest.param(
            (1, 4, 32, 64),
            (1, 2, 1, 1),
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            id="rm_B_C_repeat",
        ),
    ],
)
def test_repeat_interleaved_to_sharded_nospec(shape, repeat_shape, input_layout, memory_layout, device):
    out_mc = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1)
    run_repeat_test(
        shape,
        repeat_shape,
        device,
        input_layout=input_layout,
        input_mem_config=L1_INTERLEAVED,
        output_mem_config=out_mc,
    )


# RM sharded → sharded round-trip (BLOCK and WIDTH).
@pytest.mark.parametrize(
    "shape, input_factory, output_layout",
    [
        pytest.param(
            (1, 1, 4, 64),
            lambda d: _block_shard_config((1, 1, 4, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            id="block_to_block",
        ),
        pytest.param(
            (1, 1, 4, 64),
            lambda d: _width_shard_config((1, 1, 4, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            id="width_to_width",
        ),
    ],
)
def test_repeat_rm_sharded_to_sharded(shape, input_factory, output_layout, device):
    out_mc = ttnn.MemoryConfig(output_layout, ttnn.BufferType.L1)
    run_repeat_test(
        shape,
        (1, 1, 1, 2),
        device,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        input_mem_config=input_factory(device),
        output_mem_config=out_mc,
    )


# DRAM-sharded input -> L1 interleaved (to_memory_config fallback path).
def test_repeat_dram_sharded_to_interleaved(device):
    shape = (1, 1, 128, 64)
    compute_grid = device.compute_with_storage_grid_size()
    num_cores = min(4, compute_grid.x * compute_grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
    total_h = shape[0] * shape[1] * _round_up(shape[2], _TILE_HEIGHT)
    w = _round_up(shape[3], _TILE_WIDTH)
    shard_shape = (_round_up((total_h + num_cores - 1) // num_cores, _TILE_HEIGHT), w)
    dram_sharded = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    torch.manual_seed(12345)
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=dram_sharded
    )
    result = ttnn.repeat(ttnn_input, [1, 1, 1, 2], memory_config=L1_INTERLEAVED)
    assert (
        result.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    ), f"Expected L1 INTERLEAVED, got {result.memory_config().memory_layout}"
    ref = x.repeat(1, 1, 1, 2)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    assert_with_pcc(ref.float(), got.float(), 0.9999)


# DRAM interleaved baseline (existing fast path).
@pytest.mark.parametrize(
    "input_layout, dtype",
    [
        pytest.param(ttnn.TILE_LAYOUT, ttnn.bfloat16, id="tile_bf16"),
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, id="rm_bf16"),
    ],
)
def test_repeat_dram_interleaved(input_layout, dtype, device):
    shape = (1, 1, 64, 64)
    run_repeat_test(
        shape,
        (1, 1, 1, 2),
        device,
        input_layout=input_layout,
        input_mem_config=DRAM_INTERLEAVED,
        output_mem_config=DRAM_INTERLEAVED,
        dtype=dtype,
    )


# RM L1-interleaved native path (last-dim + higher-dim, odd page width).
@pytest.mark.parametrize(
    "shape, repeat_shape, id_suffix",
    [
        pytest.param((2, 32), (1, 2), "rm_il_last_dim"),
        pytest.param((2, 32), (3, 1), "rm_il_higher_dim"),
        pytest.param((1, 1, 4, 96), (1, 1, 2, 1), "rm_il_higher_dim_4d"),
        pytest.param((3, 5), (1, 4), "rm_il_last_dim_odd_width"),
    ],
)
def test_repeat_rm_interleaved(shape, repeat_shape, id_suffix, device):
    run_repeat_test(
        shape,
        repeat_shape,
        device,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        input_mem_config=L1_INTERLEAVED,
        output_mem_config=L1_INTERLEAVED,
        dtype=ttnn.bfloat16,
    )


# All-1 repeat vector: host returns input unchanged.
@pytest.mark.parametrize(
    "input_layout, mc_factory",
    [
        pytest.param(ttnn.TILE_LAYOUT, lambda d: L1_INTERLEAVED, id="tile_interleaved"),
        pytest.param(ttnn.TILE_LAYOUT, lambda d: _height_shard_config((1, 1, 64, 64), d), id="tile_height_sharded"),
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, lambda d: L1_INTERLEAVED, id="rm_interleaved"),
        pytest.param(
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d: _height_shard_config((1, 1, 64, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="rm_height_sharded",
        ),
    ],
)
def test_repeat_all_ones_shortcut(input_layout, mc_factory, device):
    run_repeat_test(
        (1, 1, 64, 64),
        (1, 1, 1, 1),
        device,
        input_layout=input_layout,
        input_mem_config=mc_factory(device),
        output_mem_config=None,
        dtype=ttnn.bfloat16,
    )


# TILE universal-I/O matrix: essential input × output routing paths.
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="f32"),
    ],
)
@pytest.mark.parametrize(
    "shape, repeat_shape, input_factory, output_factory",
    [
        pytest.param(
            (1, 1, 64, 64),
            (1, 1, 1, 2),
            lambda d: _block_shard_config((1, 1, 64, 64), d),
            _interleaved,
            id="block_to_interleaved",
        ),
        pytest.param(
            (1, 1, 64, 128),
            (1, 1, 1, 2),
            _interleaved,
            lambda d: _height_shard_config((1, 1, 64, 256), d),
            id="interleaved_to_height",
        ),
        pytest.param(
            (1, 1, 64, 128),
            (1, 1, 1, 2),
            lambda d: _width_shard_config((1, 1, 64, 128), d),
            lambda d: _height_shard_config((1, 1, 64, 256), d),
            id="width_to_height",
        ),
        pytest.param(
            (1, 1, 64, 64),
            (1, 1, 1, 2),
            lambda d: _block_shard_config((1, 1, 64, 64), d),
            None,
            id="block_default_output",
        ),
        pytest.param(
            (1, 1, 128, 64),
            (1, 1, 1, 2),
            lambda d: _height_shard_config((1, 1, 128, 64), d),
            _sharded_no_spec(ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            id="height_to_height_nospec",
        ),
        pytest.param(
            (2, 3, 64, 96),
            (1, 1, 1, 2),
            _interleaved,
            None,
            id="interleaved_baseline",
        ),
    ],
)
def test_repeat_tile_universal_io_matrix(shape, repeat_shape, input_factory, output_factory, dtype, device):
    out_mc = output_factory(device) if output_factory is not None else None
    run_repeat_test(
        shape,
        repeat_shape,
        device,
        input_layout=ttnn.TILE_LAYOUT,
        input_mem_config=input_factory(device),
        output_mem_config=out_mc,
        dtype=dtype,
    )


# RM universal-I/O matrix: essential composite + native paths.
_RM = ttnn.ROW_MAJOR_LAYOUT


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="f32"),
    ],
)
@pytest.mark.parametrize(
    "shape, repeat_shape, input_factory, output_factory",
    [
        pytest.param(
            (1, 1, 64, 64),
            (1, 1, 1, 2),
            lambda d: _block_shard_config((1, 1, 64, 64), d, layout=_RM),
            _interleaved,
            id="block_to_interleaved",
        ),
        pytest.param(
            (1, 1, 64, 128),
            (1, 1, 1, 2),
            _interleaved,
            lambda d: _width_shard_config((1, 1, 64, 256), d, layout=_RM),
            id="interleaved_to_width",
        ),
        pytest.param(
            (1, 1, 64, 32),
            (1, 1, 1, 2),
            lambda d: _height_shard_config((1, 1, 64, 32), d, layout=_RM),
            _interleaved,
            id="height_to_interleaved",
        ),
    ],
)
def test_repeat_rm_universal_io_matrix(shape, repeat_shape, input_factory, output_factory, dtype, device):
    run_repeat_test(
        shape,
        repeat_shape,
        device,
        input_layout=_RM,
        input_mem_config=input_factory(device),
        output_mem_config=output_factory(device),
        dtype=dtype,
    )


# Irregular logical shapes (not tile-aligned) — interleaved only (sharded variants in universal matrices).
_IRREGULAR_SHAPES = [
    ((1, 1, 65, 97), (1, 1, 1, 2)),
    ((1, 13, 47, 64), (1, 2, 1, 1)),
    ((3, 5, 32, 64), (2, 1, 1, 1)),
]


@pytest.mark.parametrize("shape, repeat_shape", _IRREGULAR_SHAPES)
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_repeat_irregular_shapes_interleaved(shape, repeat_shape, input_layout, device):
    run_repeat_test(shape, repeat_shape, device, input_layout=input_layout)


# Explicit-grid edge cases: irregular shapes with uneven sharding (dynamic helpers can't express).
@pytest.mark.parametrize(
    "shape, repeat_shape, mc_factory",
    [
        pytest.param(
            (1, 1, 65, 64),
            (1, 1, 1, 2),
            lambda d: _explicit_block_shard_config(d, 3, 2, 32, 32),
            id="tile_block_3x2_irregular_H",
        ),
        pytest.param(
            (1, 1, 96, 64),
            (1, 1, 1, 2),
            lambda d: _explicit_block_shard_config(d, 2, 2, 64, 32),
            id="tile_block_uneven_h96",
        ),
    ],
)
def test_repeat_explicit_grid_edge_cases(shape, repeat_shape, mc_factory, device):
    run_repeat_test(shape, repeat_shape, device, input_layout=ttnn.TILE_LAYOUT, input_mem_config=mc_factory(device))


# Higher-rank inputs (rank 5 and 6).
@pytest.mark.parametrize(
    "shape, repeat_shape, layout, input_mem_config, dtype",
    [
        pytest.param(
            (2, 1, 1, 64, 64),
            (1, 1, 1, 1, 2),
            ttnn.TILE_LAYOUT,
            "height_sharded",
            ttnn.bfloat16,
            id="rank5_height_sharded",
        ),
        pytest.param(
            (1, 2, 1, 1, 32, 64),
            (1, 1, 1, 1, 1),
            ttnn.TILE_LAYOUT,
            "interleaved",
            ttnn.bfloat16,
            id="rank6_interleaved_noop",
        ),
        pytest.param(
            (1, 2, 1, 1, 32, 64),
            (1, 1, 1, 1, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            "interleaved",
            ttnn.bfloat16,
            id="rank6_interleaved_last_dim",
        ),
    ],
)
def test_repeat_higher_rank(shape, repeat_shape, layout, input_mem_config, dtype, device):
    if input_mem_config == "height_sharded":
        in_mc = _height_shard_config(shape, device, layout=layout)
    else:
        in_mc = L1_INTERLEAVED
    run_repeat_test(shape, repeat_shape, device, input_layout=layout, input_mem_config=in_mc, dtype=dtype)


# Default output memory_config inherits and rescales input shard_spec.
@pytest.mark.parametrize(
    "shape, repeat_shape, input_factory",
    [
        pytest.param(
            (1, 1, 128, 64),
            (1, 1, 1, 2),
            lambda d: _height_shard_config((1, 1, 128, 64), d),
            id="height_default_mc",
        ),
        pytest.param(
            (1, 1, 64, 128),
            (1, 1, 1, 2),
            lambda d: _width_shard_config((1, 1, 64, 128), d),
            id="width_default_mc",
        ),
    ],
)
def test_repeat_default_memory_config(shape, repeat_shape, input_factory, device):
    run_repeat_test(
        shape,
        repeat_shape,
        device,
        input_layout=ttnn.TILE_LAYOUT,
        input_mem_config=input_factory(device),
        output_mem_config=None,
    )


# Sharded input → sharded output without shard_spec (native + composite paths).
@pytest.mark.parametrize(
    "shape, repeat_shape, input_factory, output_layout",
    [
        pytest.param(
            (1, 1, 128, 64),
            (1, 1, 1, 2),
            lambda d: _height_shard_config((1, 1, 128, 64), d),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            id="height_in_height_out_nospec",
        ),
        pytest.param(
            (1, 1, 64, 64),
            (1, 1, 1, 2),
            lambda d: _block_shard_config((1, 1, 64, 64), d),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            id="block_in_block_out_nospec",
        ),
        pytest.param(
            (1, 1, 96, 64),
            (1, 1, 1, 2),
            lambda d: _explicit_block_shard_config(d, 2, 2, 64, 32),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            id="uneven_block_in_height_out_nospec",
        ),
    ],
)
def test_repeat_sharded_in_sharded_out_nospec(shape, repeat_shape, input_factory, output_layout, device):
    out_mc = ttnn.MemoryConfig(output_layout, ttnn.BufferType.L1)
    run_repeat_test(
        shape,
        repeat_shape,
        device,
        input_layout=ttnn.TILE_LAYOUT,
        input_mem_config=input_factory(device),
        output_mem_config=out_mc,
    )


# Focused dtype spot-checks on representative native paths.
@pytest.mark.parametrize(
    "shape, repeat_shape, layout, in_mc_fn, dtype",
    [
        pytest.param(
            (1, 1, 64, 64),
            (1, 1, 1, 2),
            ttnn.TILE_LAYOUT,
            _interleaved,
            ttnn.float32,
            id="f32_tile_interleaved",
        ),
        pytest.param(
            (1, 1, 128, 64),
            (1, 1, 1, 2),
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((1, 1, 128, 64), d),
            ttnn.float32,
            id="f32_tile_height_sharded",
        ),
        pytest.param(
            (1, 1, 64, 64),
            (1, 1, 1, 2),
            ttnn.TILE_LAYOUT,
            _interleaved,
            ttnn.bfloat8_b,
            id="bf8b_tile_interleaved",
        ),
        pytest.param(
            (4, 32),
            (1, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            _interleaved,
            ttnn.float32,
            id="f32_rm_interleaved_2d",
        ),
    ],
)
def test_repeat_dtype_coverage(shape, repeat_shape, layout, in_mc_fn, dtype, device):
    run_repeat_test(
        shape,
        repeat_shape,
        device,
        input_layout=layout,
        input_mem_config=in_mc_fn(device),
        dtype=dtype,
        pcc=0.99 if dtype == ttnn.bfloat8_b else 0.9999,
    )


# DRAM interleaved across repeat dims (N/C/H/W).
@pytest.mark.parametrize(
    "repeat_shape",
    [
        pytest.param((2, 1, 1, 1), id="N"),
        pytest.param((1, 2, 1, 1), id="C"),
        pytest.param((1, 1, 2, 1), id="H"),
        pytest.param((1, 1, 1, 2), id="W"),
    ],
)
def test_repeat_dram_interleaved_dims(repeat_shape, device):
    run_repeat_test(
        (1, 4, 64, 128),
        repeat_shape,
        device,
        input_layout=ttnn.TILE_LAYOUT,
        input_mem_config=DRAM_INTERLEAVED,
        output_mem_config=DRAM_INTERLEAVED,
    )


# ─── Path-coverage additions ─────────────────────────────────────────────────
# Targeted tests for specific dispatch branches not fully exercised by the matrices above.


# Rank mismatch handling: reps vs tensor rank.
@pytest.mark.parametrize(
    "shape, repeat_shape, id_str",
    [
        pytest.param((64, 64), (1, 1, 1, 2), "reps_longer_than_rank"),
        pytest.param((1, 1, 64, 64), (1, 2), "reps_shorter_than_rank"),
    ],
)
def test_repeat_rank_mismatch(shape, repeat_shape, id_str, device):
    run_repeat_test(shape, repeat_shape, device, input_layout=ttnn.TILE_LAYOUT, input_mem_config=L1_INTERLEAVED)


# Zero-rep across various inputs (ttnn::zeros path, Fix 4).
@pytest.mark.parametrize(
    "shape, repeat_shape, input_layout, in_mc_fn, expected_shape",
    [
        pytest.param(
            (1, 1, 64, 64),
            (1, 1, 0, 2),
            ttnn.TILE_LAYOUT,
            lambda d: L1_INTERLEAVED,
            (1, 1, 0, 128),
            id="tile_interleaved_zero_H",
        ),
        pytest.param(
            (1, 1, 64, 64),
            (1, 0, 1, 1),
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((1, 1, 64, 64), d),
            (1, 0, 64, 64),
            id="tile_height_sharded_zero_C",
        ),
        pytest.param(
            (64, 64),
            (0, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d: L1_INTERLEAVED,
            (0, 128),
            id="rm_interleaved_zero_row",
        ),
    ],
)
def test_repeat_zero_reps(shape, repeat_shape, input_layout, in_mc_fn, expected_shape, device):
    torch.manual_seed(12345)
    x = torch.rand(shape, dtype=torch.bfloat16)
    in_mc = in_mc_fn(device)
    ttnn_input = ttnn.from_torch(x, layout=input_layout, dtype=ttnn.bfloat16, device=device, memory_config=in_mc)
    result = ttnn.repeat(ttnn_input, list(repeat_shape))
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    assert got.numel() == 0
    assert tuple(got.shape) == expected_shape


# Predicate guards forcing composite fallback (specific edge cases).
@pytest.mark.parametrize(
    "shape, repeat_shape, in_mc_fn, out_mc_fn, path_id",
    [
        pytest.param(
            (1, 1, 128, 64),
            (1, 1, 1, 2),
            lambda d: _explicit_height_shard_config(d, 4, 32, 64),
            lambda d: _explicit_height_shard_config(d, 2, 64, 128),
            "sharded_mismatched_grids",
        ),
        pytest.param(
            (1, 3, 32, 32),
            (1, 2, 1, 1),
            lambda d: _explicit_height_shard_config(d, 3, 32, 32),
            lambda d: L1_INTERLEAVED,
            "higher_dim_not_locally_contained",
        ),
        pytest.param(
            (1, 1, 64, 64),
            (2, 2, 1, 1),
            lambda d: _block_shard_config((1, 1, 64, 64), d),
            lambda d: _height_shard_config((2, 2, 64, 64), d),
            "composite_with_user_spec",
        ),
        pytest.param(
            (32, 64),
            (1, 2),
            lambda d: _height_shard_config((32, 64), d, num_cores=1, layout=ttnn.TILE_LAYOUT),
            lambda d: L1_INTERLEAVED,
            "tile_rank2_sharded",
        ),
    ],
)
def test_repeat_composite_edge_cases(shape, repeat_shape, in_mc_fn, out_mc_fn, path_id, device):
    run_repeat_test(
        shape,
        repeat_shape,
        device,
        input_layout=ttnn.TILE_LAYOUT,
        input_mem_config=in_mc_fn(device),
        output_mem_config=out_mc_fn(device),
    )
