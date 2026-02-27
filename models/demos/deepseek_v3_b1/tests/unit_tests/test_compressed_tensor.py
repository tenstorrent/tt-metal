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


def _make_sharded_config(layout, buffer_type, num_cores):
    """Helper to build a sharded MemoryConfig with a dummy shard shape.

    CompressedTensor will recompute the shard shape based on packed tile sizes.
    """
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))])
    # Dummy shard shape — CompressedTensor overrides this
    shard_spec = ttnn.ShardSpec(grid, [1, 1], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(layout, buffer_type, shard_spec)


def test_device_height_sharded(device):
    """Data tensor height-sharded in L1, tile rows split across 4 cores."""
    torch.manual_seed(42)
    x = torch.randn(128, 128)  # 4x4 tile grid

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    # 4 tile rows → 4 cores, 1 tile row per core
    data_mem = _make_sharded_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, num_cores=4)

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
    # 4 tile cols → 4 banks, 1 tile col per bank
    data_mem = _make_sharded_config(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, num_cores=4)

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
    shard_spec = ttnn.ShardSpec(grid, [1, 1], ttnn.ShardOrientation.ROW_MAJOR)
    data_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)

    ct = CompressedTensor.from_torch(x, assigner, device=device, memory_config=data_mem)

    print(f"{ct}")
    assert ttnn.is_tensor_storage_on_device(ct.data)
    assert ct.max_shard_size > 0, "Expected sharded layout"

    recovered = ct.to_torch()
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"Block-sharded PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low after block-sharded round-trip"
