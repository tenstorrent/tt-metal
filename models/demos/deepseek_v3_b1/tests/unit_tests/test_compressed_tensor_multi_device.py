# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for CompressedTensor multi-device (mesh) support."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ttnn
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor, CompressedTensorAssigner
from models.demos.deepseek_v3_b1.compressed_tensor.metrics import metric_value


@pytest.fixture(scope="function")
def mesh_device():
    """Create a mesh device with HYBRID allocator mode for multi-device compressed tensor tests."""
    num_devices = ttnn.get_num_devices()
    if num_devices < 2:
        pytest.skip("Multi-device compressed tensor tests require at least 2 devices")

    mesh_shape = ttnn.MeshShape(1, min(num_devices, 2))
    mesh = ttnn.open_mesh_device(
        mesh_shape=mesh_shape,
        allocator_mode=ttnn.device.AllocatorMode.HYBRID,
    )
    yield mesh
    ttnn.close_mesh_device(mesh)


def _print_per_device_per_core_addresses(ct):
    """Print L1 addresses for all per-device per-core tensors."""
    for dev_idx in range(ct._num_devices):
        per_core = ct._multi_device_data_per_core[dev_idx]
        addrs = []
        for (cx, cy), tensor in per_core.items():
            core = ttnn.CoreCoord(cx, cy)
            addr = ct.get_data_l1_address_per_core(core, device_idx=dev_idx)
            addrs.append(f"({cx},{cy})={addr:#x}")
        print(f"  Device {dev_idx}: {', '.join(addrs)}")


def _div_up(a, b):
    return (a + b - 1) // b


def _make_sharded_mem_config(tensor_shape, layout, buffer_type, grid, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    """Build a sharded MemoryConfig with proper tile-aligned shard shape."""
    h, w = tensor_shape[-2], tensor_shape[-1]
    tile = 32
    num_cores = grid.num_cores()
    grid_size = grid.bounding_box().grid_size()
    grid_h, grid_w = grid_size.y, grid_size.x

    if layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        shard_h = _div_up(h, num_cores)
        shard_h = _div_up(shard_h, tile) * tile
        shard_shape = [shard_h, w]
    elif layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
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


# ---------------------------------------------------------------------------
# Lockstep multi-device tests
# ---------------------------------------------------------------------------


def test_lockstep_round_trip_height_sharded(mesh_device):
    """Lockstep multi-device: height-sharded round-trip with from_torch → to_torch."""
    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(42)
    # Each device gets 64 rows → 2 tile rows, 4 tile cols
    x = torch.randn(64 * num_devices, 128)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    # Shard config is per-device: (64, 128) height-sharded across 2 cores
    data_mem = _make_sharded_mem_config((64, 128), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, grid)
    ct = CompressedTensor.from_torch(
        x,
        assigner,
        device=mesh_device,
        memory_config=data_mem,
        per_core_allocation=False,
        mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0)]),
    )

    print(f"{ct}")
    assert ct._num_devices == num_devices
    assert ct.data is not None

    recovered = ct.to_torch()
    assert recovered.shape == x.shape, f"Shape mismatch: {recovered.shape} != {x.shape}"
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"Lockstep height-sharded multi-device PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low"


def test_lockstep_per_device_data_independence(mesh_device):
    """Verify each device gets different data in lockstep mode."""
    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(42)
    # Create tensor with distinct data per device-slice
    slices = []
    for i in range(num_devices):
        slices.append(torch.randn(64, 128) * (10.0**i))
    x = torch.cat(slices, dim=0)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    data_mem = _make_sharded_mem_config((64, 128), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, grid)
    ct = CompressedTensor.from_torch(
        x,
        assigner,
        device=mesh_device,
        memory_config=data_mem,
        per_core_allocation=False,
        mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0)]),
    )

    recovered = ct.to_torch()
    # Verify per-device slices are different
    for i in range(num_devices):
        dev_recovered = recovered[i * 64 : (i + 1) * 64, :]
        dev_original = x[i * 64 : (i + 1) * 64, :]
        pcc = metric_value(dev_original.numpy(), dev_recovered.numpy(), "pcc")
        print(f"Device {i} PCC: {pcc:.6f}")
        assert pcc > 0.98, f"Device {i} PCC {pcc:.6f} too low"


def test_lockstep_different_assignment_per_device(mesh_device):
    """Different data per device should result in different assignments."""
    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(42)
    # Device 0: large values (bfp8), Device 1: moderate values (bfp4 or bfp2)
    dev0_data = torch.randn(128, 128) * 100.0
    dev1_data = torch.randn(128, 128) * 0.01
    x = torch.cat([dev0_data, dev1_data], dim=0)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.99, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))])
    data_mem = _make_sharded_mem_config((128, 128), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, grid)
    ct = CompressedTensor.from_torch(
        x,
        assigner,
        device=mesh_device,
        memory_config=data_mem,
        per_core_allocation=False,
        mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0)]),
    )

    # Verify the full assignment covers all devices
    full_assign = ct.get_assignment()
    tiles_h_per_dev = ct._per_device_tiles_h
    tiles_w = ct.tiles_w
    dev0_assign = full_assign[:tiles_h_per_dev, :]
    dev1_assign = full_assign[tiles_h_per_dev:, :]
    # Assignments should differ because data magnitudes differ
    print(f"Dev0 assignment unique: {np.unique(dev0_assign)}")
    print(f"Dev1 assignment unique: {np.unique(dev1_assign)}")

    recovered = ct.to_torch()
    assert recovered.shape == x.shape


# ---------------------------------------------------------------------------
# Per-core multi-device tests
# ---------------------------------------------------------------------------


def test_per_core_round_trip_height_sharded(mesh_device):
    """Per-core multi-device: height-sharded round-trip."""
    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(42)
    x = torch.randn(64 * num_devices, 128)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    data_mem = _make_sharded_mem_config((64, 128), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, grid)
    ct = CompressedTensor.from_torch(
        x,
        assigner,
        device=mesh_device,
        memory_config=data_mem,
        per_core_allocation=True,
        mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0)]),
    )

    print(f"{ct}")
    assert ct._num_devices == num_devices
    assert len(ct._multi_device_data_per_core) == num_devices
    _print_per_device_per_core_addresses(ct)

    recovered = ct.to_torch()
    assert recovered.shape == x.shape, f"Shape mismatch: {recovered.shape} != {x.shape}"
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"Per-core height-sharded multi-device PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low"


def test_per_core_per_device_data_independence(mesh_device):
    """Verify each device gets independent per-core tensors."""
    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(42)
    slices = []
    for i in range(num_devices):
        slices.append(torch.randn(64, 128) * (10.0**i))
    x = torch.cat(slices, dim=0)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    data_mem = _make_sharded_mem_config((64, 128), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, grid)
    ct = CompressedTensor.from_torch(
        x,
        assigner,
        device=mesh_device,
        memory_config=data_mem,
        per_core_allocation=True,
        mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0)]),
    )

    # Each device should have its own set of per-core tensors
    for dev_idx in range(num_devices):
        per_core = ct._multi_device_data_per_core[dev_idx]
        assert len(per_core) > 0, f"Device {dev_idx} has no per-core tensors"
        print(f"Device {dev_idx}: {len(per_core)} cores with data")
    _print_per_device_per_core_addresses(ct)

    recovered = ct.to_torch()
    for i in range(num_devices):
        dev_recovered = recovered[i * 64 : (i + 1) * 64, :]
        dev_original = x[i * 64 : (i + 1) * 64, :]
        pcc = metric_value(dev_original.numpy(), dev_recovered.numpy(), "pcc")
        print(f"Device {i} PCC: {pcc:.6f}")
        assert pcc > 0.98, f"Device {i} PCC {pcc:.6f} too low"


def test_per_core_get_data_tensors_flattens_all_devices(mesh_device):
    """get_data_tensors() should return all per-core tensors from all devices."""
    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(42)
    x = torch.randn(64 * num_devices, 128)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    data_mem = _make_sharded_mem_config((64, 128), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, grid)
    ct = CompressedTensor.from_torch(
        x,
        assigner,
        device=mesh_device,
        memory_config=data_mem,
        per_core_allocation=True,
        mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0)]),
    )

    all_tensors = ct.get_data_tensors()
    # Should have tensors from all devices
    total_cores = sum(len(ct._multi_device_data_per_core[d]) for d in range(num_devices))
    assert len(all_tensors) == total_cores, f"Expected {total_cores} tensors, got {len(all_tensors)}"
    _print_per_device_per_core_addresses(ct)


def test_per_core_tensors_allocated_per_device(mesh_device):
    """Verify per-device per-core tensors are allocated on device."""
    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(42)
    x = torch.randn(64 * num_devices, 128)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    data_mem = _make_sharded_mem_config((64, 128), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, grid)
    ct = CompressedTensor.from_torch(
        x,
        assigner,
        device=mesh_device,
        memory_config=data_mem,
        per_core_allocation=True,
        mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0)]),
    )

    for dev_idx in range(num_devices):
        per_core = ct._multi_device_data_per_core[dev_idx]
        assert len(per_core) > 0, f"Device {dev_idx} has no per-core tensors"
        for (cx, cy), tensor in per_core.items():
            assert ttnn.is_tensor_storage_on_device(tensor), f"Device {dev_idx} core ({cx},{cy}) tensor not on device"
    _print_per_device_per_core_addresses(ct)


# ---------------------------------------------------------------------------
# Mixed format multi-device tests
# ---------------------------------------------------------------------------


def test_lockstep_mixed_formats_per_device(mesh_device):
    """Each device gets different data patterns → different format mixes."""
    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(42)
    # Device 0: mostly bfp4 (moderate values)
    # Device 1: mix of bfp4 and bfp2 (varying values)
    dev0_data = torch.randn(128, 128) * 10.0
    dev1_data = torch.randn(128, 128) * 0.1
    x = torch.cat([dev0_data, dev1_data], dim=0)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.85, formats=["bfp4", "bfp2"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))])
    data_mem = _make_sharded_mem_config((128, 128), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, grid)
    ct = CompressedTensor.from_torch(
        x,
        assigner,
        device=mesh_device,
        memory_config=data_mem,
        per_core_allocation=False,
        mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0)]),
    )

    recovered = ct.to_torch()
    assert recovered.shape == x.shape
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"Mixed format lockstep multi-device PCC: {pcc:.6f}")
    assert pcc > 0.80, f"PCC {pcc:.6f} too low"


# ---------------------------------------------------------------------------
# Core range set multi-device tests
# ---------------------------------------------------------------------------


def test_get_data_core_range_set_per_device(mesh_device):
    """get_data_core_range_set works per-device for multi-device per-core mode."""
    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(42)
    x = torch.randn(64 * num_devices, 128)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    data_mem = _make_sharded_mem_config((64, 128), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, grid)
    ct = CompressedTensor.from_torch(
        x,
        assigner,
        device=mesh_device,
        memory_config=data_mem,
        per_core_allocation=True,
        mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0)]),
    )

    for dev_idx in range(num_devices):
        crs = ct.get_data_core_range_set(device_idx=dev_idx)
        print(f"Device {dev_idx} core range set: {crs}")
        # Should have cores with data
        assert crs.num_cores() > 0, f"Device {dev_idx} has no cores in range set"
    _print_per_device_per_core_addresses(ct)


# ---------------------------------------------------------------------------
# 2D mesh sharding tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def mesh_device_2d():
    """Create a 2D mesh device for 2D sharding tests."""
    num_devices = ttnn.get_num_devices()
    if num_devices < 4:
        pytest.skip("2D mesh sharding tests require at least 4 devices")

    mesh_shape = ttnn.MeshShape(2, 2)
    mesh = ttnn.open_mesh_device(
        mesh_shape=mesh_shape,
        allocator_mode=ttnn.device.AllocatorMode.HYBRID,
    )
    yield mesh
    ttnn.close_mesh_device(mesh)


def test_2d_lockstep_round_trip(mesh_device_2d):
    """2D mesh sharding: split height across rows, width across cols."""
    mesh_rows, mesh_cols = mesh_device_2d.shape[0], mesh_device_2d.shape[1]
    num_devices = mesh_device_2d.get_num_devices()
    torch.manual_seed(42)
    # Each device gets (128, 64) → 4x2 tile grid
    x = torch.randn(128 * mesh_rows, 64 * mesh_cols)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    # Shard config is per-device: (128, 64) height-sharded across 2 cores
    data_mem = _make_sharded_mem_config((128, 64), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, grid)

    ct = CompressedTensor.from_torch(
        x,
        assigner,
        device=mesh_device_2d,
        memory_config=data_mem,
        per_core_allocation=False,
        mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(-1)]),
    )

    print(f"{ct}")
    assert ct._num_devices == num_devices

    recovered = ct.to_torch()
    assert recovered.shape == x.shape, f"Shape mismatch: {recovered.shape} != {x.shape}"
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"2D lockstep round-trip PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} too low"


def test_2d_per_core_round_trip(mesh_device_2d):
    """2D mesh per-core: split height across rows, width across cols."""
    mesh_rows, mesh_cols = mesh_device_2d.shape[0], mesh_device_2d.shape[1]
    num_devices = mesh_device_2d.get_num_devices()
    per_dev_h, per_dev_w = 128, 64
    torch.manual_seed(42)
    x = torch.randn(per_dev_h * mesh_rows, per_dev_w * mesh_cols)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    data_mem = _make_sharded_mem_config(
        (per_dev_h, per_dev_w), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, grid
    )

    ct = CompressedTensor.from_torch(
        x,
        assigner,
        device=mesh_device_2d,
        memory_config=data_mem,
        per_core_allocation=True,
        mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(-1)]),
    )

    print(f"{ct}")
    assert ct._num_devices == num_devices
    assert len(ct._multi_device_data_per_core) == num_devices
    _print_per_device_per_core_addresses(ct)

    recovered = ct.to_torch()
    assert recovered.shape == x.shape, f"Shape mismatch: {recovered.shape} != {x.shape}"
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"2D per-core round-trip PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} too low"


def test_2d_replicate_one_axis(mesh_device_2d):
    """2D mesh with None on one axis: replicate across rows, shard cols."""
    mesh_rows, mesh_cols = mesh_device_2d.shape[0], mesh_device_2d.shape[1]
    num_devices = mesh_device_2d.get_num_devices()
    torch.manual_seed(42)
    # Only shard width across cols, replicate across rows
    # Each device gets (128, 64)
    x = torch.randn(128, 64 * mesh_cols)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    data_mem = _make_sharded_mem_config((128, 64), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, grid)

    ct = CompressedTensor.from_torch(
        x,
        assigner,
        device=mesh_device_2d,
        memory_config=data_mem,
        per_core_allocation=False,
        mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)]),
    )

    print(f"{ct}")
    assert ct._num_devices == num_devices

    recovered = ct.to_torch()
    assert recovered.shape == x.shape, f"Shape mismatch: {recovered.shape} != {x.shape}"
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"2D replicate-rows shard-cols PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} too low"
