# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CompressedTensor class."""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_b1.compressed_tensor import (
    CompressedTensor,
    CompressedTensorAssigner,
    bfp_tile_packed_size,
    ttnn_quantize_fn,
)
from models.demos.deepseek_v3_b1.compressed_tensor.metrics import metric_value


def test_from_torch_round_trip():
    """Create CompressedTensor via from_torch, unpack, and verify PCC."""
    torch.manual_seed(42)
    x = torch.randn(128, 128)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    ct = CompressedTensor.from_torch(x, assigner)

    print(f"{ct}")
    print(f"Tile counts: {ct.tile_counts}")

    assert ct.tile_counts["bfp8"] > 0 and ct.tile_counts["bfp4"] > 0, f"Expected mix: {ct.tile_counts}"

    recovered = ct.to_torch()
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"Round-trip PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low"


def test_assignment_stored_correctly():
    """The assignment tensor should round-trip through ttnn uint8 storage."""
    torch.manual_seed(0)
    x = torch.randn(64, 64)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    result = assigner.assign(x, ttnn_quantize_fn)
    ct = CompressedTensor(x, result.assignment)

    recovered_assignment = ct.get_assignment_numpy()
    assert (recovered_assignment == result.assignment).all(), "Assignment round-trip mismatch"


def test_data_bytes_matches_packed_size():
    """data_bytes property should match actual packed tensor size."""
    torch.manual_seed(42)
    x = torch.randn(128, 128)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    ct = CompressedTensor.from_torch(x, assigner)

    actual_size = ttnn.to_torch(ct.data).numel()
    print(f"data_bytes={ct.data_bytes}, actual tensor size={actual_size}")
    assert ct.data_bytes == actual_size, f"data_bytes {ct.data_bytes} != actual {actual_size}"


def test_packed_size_savings():
    """CompressedTensor with mixed formats should use fewer bytes than uniform bfp8."""
    torch.manual_seed(99)
    x = torch.randn(128, 128)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    ct = CompressedTensor.from_torch(x, assigner)

    uniform_bfp8_bytes = ct.num_tiles * bfp_tile_packed_size(7)

    print(f"{ct}")
    print(
        f"Savings: {uniform_bfp8_bytes - ct.data_bytes} bytes "
        f"({100 * (1 - ct.data_bytes / uniform_bfp8_bytes):.1f}%)"
    )
    assert (
        ct.data_bytes <= uniform_bfp8_bytes
    ), f"Mixed ({ct.data_bytes}) should be <= uniform bfp8 ({uniform_bfp8_bytes})"


# ---------------------------------------------------------------------------
# On-device tests
# ---------------------------------------------------------------------------


def test_device_round_trip(device):
    """Pack on host, place on device via from_torch(device=), unpack after read-back."""
    torch.manual_seed(42)
    x = torch.randn(128, 128)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    ct = CompressedTensor.from_torch(x, assigner, device=device)

    print(f"{ct}")
    print(f"Tile counts: {ct.tile_counts}")

    assert ct.tile_counts["bfp8"] > 0 and ct.tile_counts["bfp4"] > 0, f"Expected mix: {ct.tile_counts}"

    # unpack() handles from_device internally
    recovered = ct.to_torch()
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"Device round-trip PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low after device round-trip"


def test_device_with_memory_config(device):
    """Place data in DRAM and assignment in L1 using separate memory configs."""
    torch.manual_seed(42)
    x = torch.randn(128, 128)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    ct = CompressedTensor.from_torch(
        x,
        assigner,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        assignment_memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    print(f"{ct}")

    # Verify tensors are on device with correct memory configs
    assert ttnn.is_tensor_storage_on_device(ct.data)
    assert ttnn.is_tensor_storage_on_device(ct.assignment)

    # Unpack and verify PCC
    recovered = ct.to_torch()
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"DRAM data + L1 assignment round-trip PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low"


def test_device_shared_memory_config(device):
    """Single memory_config applies to both data and assignment."""
    torch.manual_seed(42)
    x = torch.randn(128, 128)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    ct = CompressedTensor.from_torch(
        x,
        assigner,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    assert ttnn.is_tensor_storage_on_device(ct.data)
    assert ttnn.is_tensor_storage_on_device(ct.assignment)

    recovered = ct.to_torch()
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"Shared DRAM memory config PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low"


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
    assert ttnn.is_tensor_storage_on_device(ct.data)
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
    assert ttnn.is_tensor_storage_on_device(ct.data)
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
    assert ttnn.is_tensor_storage_on_device(ct.data)
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
    assert ttnn.is_tensor_storage_on_device(ct.data)

    # Verify assignment round-trips correctly despite uneven shards
    result = assigner.assign(x, ttnn_quantize_fn)
    recovered_assignment = ct.get_assignment_numpy()
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
    assert ttnn.is_tensor_storage_on_device(ct.data)

    result = assigner.assign(x, ttnn_quantize_fn)
    recovered_assignment = ct.get_assignment_numpy()
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
    assert ttnn.is_tensor_storage_on_device(ct.data)

    result = assigner.assign(x, ttnn_quantize_fn)
    recovered_assignment = ct.get_assignment_numpy()
    assert (recovered_assignment == result.assignment).all(), "Assignment mismatch on uneven block shard"

    recovered = ct.to_torch()
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"Uneven block-shard PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low"
