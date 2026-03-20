# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CompressedTensor class."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device({"allocator_mode": ttnn.device.AllocatorMode.HYBRID})
from models.demos.deepseek_v3_b1.compressed_tensor import (
    COMPRESSED_FORMATS,
    CompressedTensor,
    CompressedTensorAssigner,
    ttnn_quantize_fn,
)
from models.demos.deepseek_v3_b1.compressed_tensor.metrics import metric_value
from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import (
    pack_bfp_tile,
    quantize_dequantize_bfp,
    unpack_bfp_tile,
)


def _check_bfp2_cpp_vs_python(xn: np.ndarray, label: str):
    """Helper: verify C++ bfp2 pack→unpack matches Python for every tile."""
    tile_hw = 32
    tiles_h, tiles_w = xn.shape[0] // tile_hw, xn.shape[1] // tile_hw
    num_mismatches = 0
    for tr in range(tiles_h):
        for tc in range(tiles_w):
            tile_np = xn[tr * tile_hw : (tr + 1) * tile_hw, tc * tile_hw : (tc + 1) * tile_hw].copy()
            packed = pack_bfp_tile(tile_np, mant_bits=1)
            cpp_result = unpack_bfp_tile(packed, mant_bits=1)
            py_result = quantize_dequantize_bfp(tile_np, mant_bits=1)
            if not np.array_equal(cpp_result, py_result):
                max_diff = np.max(np.abs(cpp_result - py_result))
                print(f"  MISMATCH {label} tile ({tr},{tc}): max diff = {max_diff}")
                num_mismatches += 1
    print(f"bfp2 C++ vs Python [{label}]: {tiles_h * tiles_w} tiles, {num_mismatches} mismatches")
    assert num_mismatches == 0, f"{label}: {num_mismatches} tiles had C++ vs Python mismatch"


def test_bfp2_cpp_matches_python():
    """Validate C++ bfp2 pack→unpack matches Python with varied data patterns."""
    torch.manual_seed(0)
    x = torch.randn(256, 256)
    x[:64, :64] *= 100.0
    x[64:128, 64:128] *= 1e-4
    x[128:160, 128:160] = 0.0
    _check_bfp2_cpp_vs_python(x.numpy(), "basic_256x256")


def test_bfp2_cpp_matches_python_large():
    """bfp2 C++ vs Python on a large tensor (7168x32, DeepSeek-like shape)."""
    torch.manual_seed(1)
    x = torch.randn(7168, 32)
    _check_bfp2_cpp_vs_python(x.numpy(), "large_7168x32")


def test_bfp2_cpp_matches_python_edge_cases():
    """bfp2 C++ vs Python with edge-case data: inf, denormals, alternating signs, constants."""
    tile_hw = 32
    tiles = []

    # All zeros
    tiles.append(np.zeros((tile_hw, tile_hw), dtype=np.float32))

    # All same positive value
    tiles.append(np.full((tile_hw, tile_hw), 42.0, dtype=np.float32))

    # All same negative value
    tiles.append(np.full((tile_hw, tile_hw), -0.001, dtype=np.float32))

    # Alternating +1/-1 checkerboard
    checker = np.ones((tile_hw, tile_hw), dtype=np.float32)
    checker[::2, ::2] = -1.0
    checker[1::2, 1::2] = -1.0
    tiles.append(checker)

    # Very large values (near float32 max exponent)
    tiles.append(np.full((tile_hw, tile_hw), 1e38, dtype=np.float32))

    # Denormals (very small but nonzero)
    tiles.append(np.full((tile_hw, tile_hw), 1e-40, dtype=np.float32))

    # Mixed: one element huge, rest tiny
    mixed = np.full((tile_hw, tile_hw), 1e-6, dtype=np.float32)
    mixed[0, 0] = 1e6
    tiles.append(mixed)

    # Random with high dynamic range per row
    rng = np.random.RandomState(99)
    dynamic = rng.randn(tile_hw, tile_hw).astype(np.float32)
    for row in range(tile_hw):
        dynamic[row, :] *= 10.0 ** (row - 16)
    tiles.append(dynamic)

    # Stack into a single tensor: 8 tiles in a column
    x = np.concatenate(tiles, axis=0)  # (8*32, 32)
    _check_bfp2_cpp_vs_python(x, "edge_cases")


def _div_up(a, b):
    return (a + b - 1) // b


def _make_sharded_mem_config(tensor_shape, layout, buffer_type, grid, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    """Build a sharded MemoryConfig with proper tile-aligned shard shape.

    Computes shard shape using div_up on the tensor dimensions, aligned to tile size (32).
    """
    h, w = tensor_shape[-2], tensor_shape[-1]
    tile = 32
    num_cores = grid.num_cores()
    grid_size = grid.bounding_box().grid_size()
    grid_h, grid_w = grid_size.y, grid_size.x

    if layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        # Each core gets some rows, all columns
        shard_h = _div_up(h, num_cores)
        shard_h = _div_up(shard_h, tile) * tile  # align to tile
        shard_shape = [shard_h, w]
    elif layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        # Each core gets all rows, some columns
        shard_w = _div_up(w, num_cores)
        shard_w = _div_up(shard_w, tile) * tile
        shard_shape = [h, shard_w]
    elif layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        is_row_major = orientation == ttnn.ShardOrientation.ROW_MAJOR
        h_cores = grid_h if is_row_major else grid_w
        w_cores = grid_w if is_row_major else grid_h
        shard_h = _div_up(h, h_cores)
        shard_h = _div_up(shard_h, tile) * tile
        shard_w = _div_up(w, w_cores)
        shard_w = _div_up(shard_w, tile) * tile
        shard_shape = [shard_h, shard_w]
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    shard_spec = ttnn.ShardSpec(grid, shard_shape, orientation)
    return ttnn.MemoryConfig(layout, buffer_type, shard_spec)


def test_device_height_sharded(device):
    """Data tensor height-sharded in L1, tile rows split across 4 cores."""
    torch.manual_seed(42)
    x = torch.randn(128, 128)  # 4x4 tile grid

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    # 4 tile rows → 4 cores, 1 tile row per core
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))])
    data_mem = _make_sharded_mem_config(x.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, grid)

    ct = CompressedTensor.from_torch(x, assigner, device=device, memory_config=data_mem)

    print(f"{ct}")
    assert all(ttnn.is_tensor_storage_on_device(t) for t in ct.get_data_tensors())
    assert ct.max_shard_size > 0, "Expected sharded layout"

    recovered = ct.to_torch()
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"Height-sharded PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low after height-sharded round-trip"


def test_device_width_sharded(device):
    """Data tensor width-sharded in DRAM, tile columns split across 4 banks."""
    torch.manual_seed(42)
    x = torch.randn(128, 128)  # 4x4 tile grid

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))])
    data_mem = _make_sharded_mem_config(x.shape, ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, grid)

    ct = CompressedTensor.from_torch(x, assigner, device=device, memory_config=data_mem)

    print(f"{ct}")
    assert all(ttnn.is_tensor_storage_on_device(t) for t in ct.get_data_tensors())
    assert ct.max_shard_size > 0, "Expected sharded layout"

    recovered = ct.to_torch()
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"Width-sharded PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low after width-sharded round-trip"


def test_device_block_sharded(device):
    """Data tensor block-sharded in L1, 2×2 core grid for a 4×4 tile grid."""
    torch.manual_seed(42)
    x = torch.randn(128, 128)  # 4x4 tile grid

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    # 2x2 grid: each core gets a 2x2 block of tiles
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])
    data_mem = _make_sharded_mem_config(x.shape, ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, grid)

    ct = CompressedTensor.from_torch(x, assigner, device=device, memory_config=data_mem)

    print(f"{ct}")
    assert all(ttnn.is_tensor_storage_on_device(t) for t in ct.get_data_tensors())
    assert ct.max_shard_size > 0, "Expected sharded layout"

    recovered = ct.to_torch()
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"Block-sharded PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low after block-sharded round-trip"


# ---------------------------------------------------------------------------
# Uneven shard tests (tile grid doesn't divide evenly across cores)
# ---------------------------------------------------------------------------


def _print_shard_distribution(ct, memory_config):
    """Print how many tiles each core gets, with core coordinates."""
    if not hasattr(ct, "_shard_tile_coords"):
        return
    # Enumerate cores from the shard spec grid in row-major order
    grid = memory_config.shard_spec.grid
    cores = []
    for cr in grid.ranges():
        for y in range(cr.start.y, cr.end.y + 1):
            for x in range(cr.start.x, cr.end.x + 1):
                cores.append((x, y))
    for i, coords in enumerate(ct._shard_tile_coords):
        core_xy = cores[i] if i < len(cores) else ("?", "?")
        print(f"  core ({core_xy[0]},{core_xy[1]}): {len(coords)} tiles {coords}")
    print(f"  max_shard_size: {ct.max_shard_size} bytes")


def _validate_shard_distribution(ct, memory_config):
    """Validate that our tile-to-core mapping matches the memory config's shard shape.

    Converts the shard shape from elements to tiles and checks our max tiles per shard matches.
    """
    if not hasattr(ct, "_shard_tile_coords"):
        return

    tile_hw = ct.tile_hw
    shard_shape = memory_config.shard_spec.shape
    shard_h_tiles = shard_shape[0] // tile_hw
    shard_w_tiles = shard_shape[1] // tile_hw
    expected_max = shard_h_tiles * shard_w_tiles

    our_max = max(len(coords) for coords in ct._shard_tile_coords)

    print(f"  shard shape: {shard_shape} = {shard_h_tiles}x{shard_w_tiles} tiles = {expected_max} tiles/shard")
    print(f"  our max tiles/shard: {our_max}")
    print(f"  our tiles/core: {[len(c) for c in ct._shard_tile_coords]}")

    assert our_max == expected_max, f"Max tiles per shard mismatch: expected={expected_max}, ours={our_max}"


def test_device_uneven_height_shard(device):
    """3 tile rows across 2 cores — first core gets 2 rows, second gets 1."""
    torch.manual_seed(42)
    x = torch.randn(96, 128)  # 3x4 tile grid

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    data_mem = _make_sharded_mem_config(x.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, grid)

    ct = CompressedTensor.from_torch(x, assigner, device=device, memory_config=data_mem)

    print(f"{ct}")
    _print_shard_distribution(ct, data_mem)
    _validate_shard_distribution(ct, data_mem)
    assert all(ttnn.is_tensor_storage_on_device(t) for t in ct.get_data_tensors())

    # Verify assignment round-trips correctly despite uneven shards
    result = assigner.assign(x, ttnn_quantize_fn)
    recovered_assignment = ct.get_assignment()
    assert (recovered_assignment == result.assignment).all(), "Assignment mismatch on uneven shard"

    recovered = ct.to_torch()
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"Uneven height-shard PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low"


def test_device_uneven_width_shard(device):
    """4 tile columns across 3 banks — first bank gets 2 cols, others get 1."""
    torch.manual_seed(42)
    x = torch.randn(96, 128)  # 3x4 tile grid

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))])
    data_mem = _make_sharded_mem_config(x.shape, ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, grid)

    ct = CompressedTensor.from_torch(x, assigner, device=device, memory_config=data_mem)

    print(f"{ct}")
    _print_shard_distribution(ct, data_mem)
    _validate_shard_distribution(ct, data_mem)
    assert all(ttnn.is_tensor_storage_on_device(t) for t in ct.get_data_tensors())

    result = assigner.assign(x, ttnn_quantize_fn)
    recovered_assignment = ct.get_assignment()
    assert (recovered_assignment == result.assignment).all(), "Assignment mismatch on uneven shard"

    recovered = ct.to_torch()
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"Uneven width-shard PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low"


def test_device_uneven_block_shard(device):
    """3x4 tile grid on a 2x3 core grid — uneven in both dims.

    Height: 3 rows / 2 grid_h → row 0 gets 2 tile rows, row 1 gets 1.
    Width: 4 cols / 3 grid_w → col 0 gets 2 tile cols, cols 1-2 get 1.
    Core (0,0) gets 2x2=4 tiles, core (1,2) gets 1x1=1 tile.
    """
    torch.manual_seed(42)
    x = torch.randn(96, 128)  # 3x4 tile grid

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 1))])
    data_mem = _make_sharded_mem_config(x.shape, ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, grid)

    ct = CompressedTensor.from_torch(x, assigner, device=device, memory_config=data_mem)

    print(f"{ct}")
    _print_shard_distribution(ct, data_mem)
    _validate_shard_distribution(ct, data_mem)
    assert all(ttnn.is_tensor_storage_on_device(t) for t in ct.get_data_tensors())

    result = assigner.assign(x, ttnn_quantize_fn)
    recovered_assignment = ct.get_assignment()
    assert (recovered_assignment == result.assignment).all(), "Assignment mismatch on uneven block shard"

    recovered = ct.to_torch()
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"Uneven block-shard PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low"


def test_device_4d_uneven_height_shard(device):
    """4D tensor (2,3,96,128) height-sharded across 4 cores.

    tiles_h = (2*3*96)/32 = 18 rows, div_up(18,4) = 5 rows/shard.
    First 2 cores get 5 tile rows, last 2 get 4 tile rows.
    """
    torch.manual_seed(42)
    x = torch.randn(2, 3, 96, 128)  # tiles_h=18, tiles_w=4

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))])
    data_mem = _make_sharded_mem_config(
        (2 * 3 * 96, 128),  # flattened 2D shape for shard computation
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        grid,
    )

    ct = CompressedTensor.from_torch(x, assigner, device=device, memory_config=data_mem)

    print(f"{ct}")
    print(f"tiles_h={ct.tiles_h}, tiles_w={ct.tiles_w}")
    _print_shard_distribution(ct, data_mem)
    _validate_shard_distribution(ct, data_mem)
    assert all(ttnn.is_tensor_storage_on_device(t) for t in ct.get_data_tensors())
    assert ct.tiles_h == 18
    assert ct.tiles_w == 4

    recovered = ct.to_torch()
    assert recovered.shape == x.shape, f"Shape mismatch: {recovered.shape} != {x.shape}"
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"4D uneven height-shard PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low"


def test_device_4d_uneven_width_shard(device):
    """4D tensor (2,3,64,160) width-sharded across 3 banks.

    tiles_w = 160/32 = 5 cols, div_up(5,3) = 2 cols/shard.
    First 2 banks get 2 tile cols, last bank gets 1.
    """
    torch.manual_seed(42)
    x = torch.randn(2, 3, 64, 160)  # tiles_h=12, tiles_w=5

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))])
    data_mem = _make_sharded_mem_config(
        (2 * 3 * 64, 160),
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        grid,
    )

    ct = CompressedTensor.from_torch(x, assigner, device=device, memory_config=data_mem)

    print(f"{ct}")
    print(f"tiles_h={ct.tiles_h}, tiles_w={ct.tiles_w}")
    _print_shard_distribution(ct, data_mem)
    _validate_shard_distribution(ct, data_mem)
    assert all(ttnn.is_tensor_storage_on_device(t) for t in ct.get_data_tensors())
    assert ct.tiles_h == 12
    assert ct.tiles_w == 5

    recovered = ct.to_torch()
    assert recovered.shape == x.shape, f"Shape mismatch: {recovered.shape} != {x.shape}"
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"4D uneven width-shard PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low"


def test_device_4d_uneven_block_shard(device):
    """4D tensor (2,3,96,160) block-sharded on a 2x3 grid.

    tiles_h = (2*3*96)/32 = 18, tiles_w = 160/32 = 5.
    Height: div_up(18,2) = 9 rows/shard. Width: div_up(5,3) = 2 cols/shard.
    Core (0,0) gets 9x2=18 tiles, core (2,1) gets 0x1=0 tiles.
    """
    torch.manual_seed(42)
    x = torch.randn(2, 3, 96, 160)  # tiles_h=18, tiles_w=5

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 1))])
    data_mem = _make_sharded_mem_config(
        (2 * 3 * 96, 160),
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        grid,
    )

    ct = CompressedTensor.from_torch(x, assigner, device=device, memory_config=data_mem)

    print(f"{ct}")
    print(f"tiles_h={ct.tiles_h}, tiles_w={ct.tiles_w}")
    _print_shard_distribution(ct, data_mem)
    _validate_shard_distribution(ct, data_mem)
    assert all(ttnn.is_tensor_storage_on_device(t) for t in ct.get_data_tensors())
    assert ct.tiles_h == 18
    assert ct.tiles_w == 5

    recovered = ct.to_torch()
    assert recovered.shape == x.shape, f"Shape mismatch: {recovered.shape} != {x.shape}"
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"4D uneven block-shard PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low"


# ---------------------------------------------------------------------------
# Mixed format with bfp0 on device
# ---------------------------------------------------------------------------


def test_device_bfp0_bfp2_bfp4_uneven_height_shard(device):
    """All 3 compressed formats (bfp4/bfp2/bfp0) on device with uneven height sharding.

    Tensor has 3 distinct regions:
      - Rows 0-63: large values → bfp4
      - Rows 64-127: moderate values → bfp2
      - Rows 128-159: near-zero values → bfp0

    5 tile rows across 3 cores (uneven: 2, 2, 1).
    """
    torch.manual_seed(42)
    x = torch.randn(160, 128)  # 5x4 tile grid
    x[:64, :] *= 50.0  # rows 0-63: large → bfp4
    x[64:128, :] *= 1.0  # rows 64-127: moderate → bfp2
    x[128:, :] *= 1e-6  # rows 128-159: tiny → bfp0

    assigner = CompressedTensorAssigner(
        metric="pcc", threshold=0.89, formats=["bfp4", "bfp2", "bfp0"], bfp0_mae_threshold=1e-4
    )
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))])
    data_mem = _make_sharded_mem_config(x.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, grid)

    ct = CompressedTensor.from_torch(x, assigner, device=device, memory_config=data_mem)

    print(f"{ct}")
    _print_shard_distribution(ct, data_mem)
    assert all(ttnn.is_tensor_storage_on_device(t) for t in ct.get_data_tensors())

    # Should have a mix of all 3 formats
    assert ct.tile_counts["bfp4"] > 0, f"Expected bfp4 tiles: {ct.tile_counts}"
    assert ct.tile_counts["bfp2"] > 0, f"Expected bfp2 tiles: {ct.tile_counts}"
    assert ct.tile_counts["bfp0"] > 0, f"Expected bfp0 tiles: {ct.tile_counts}"

    recovered = ct.to_torch()
    tile_hw = 32
    assign = ct.get_assignment()

    # Log per-tile format assignment as a grid
    print("Format assignment grid:")
    for tr in range(ct.tiles_h):
        row_fmts = [COMPRESSED_FORMATS[assign[tr, tc]] for tc in range(ct.tiles_w)]
        print(f"  row {tr}: {row_fmts}")

    # Verify per-tile: bfp0 tiles are zeros, others are non-zero with reasonable quality
    for tr in range(ct.tiles_h):
        for tc in range(ct.tiles_w):
            fmt = COMPRESSED_FORMATS[assign[tr, tc]]
            ref_tile = x[tr * tile_hw : (tr + 1) * tile_hw, tc * tile_hw : (tc + 1) * tile_hw]
            rec_tile = recovered[tr * tile_hw : (tr + 1) * tile_hw, tc * tile_hw : (tc + 1) * tile_hw]
            if fmt == "bfp0":
                assert (rec_tile == 0).all(), f"bfp0 tile ({tr},{tc}) should be all zeros"
            else:
                assert rec_tile.abs().max() > 0, f"{fmt} tile ({tr},{tc}) should be non-zero"
                pcc = metric_value(ref_tile.numpy(), rec_tile.numpy(), "pcc")
                print(f"  tile ({tr},{tc}) [{fmt}]: PCC={pcc:.6f}")

    # Overall PCC including bfp0 tiles (zeroed regions lower the PCC, that's the tradeoff)
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"Overall PCC (including bfp0): {pcc:.6f}")
    assert pcc > 0.93, f"Overall PCC {pcc:.6f} too low"
