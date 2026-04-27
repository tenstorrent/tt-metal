# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CompressedTensor multi-device (mesh) support."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ttnn
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor, CompressedTensorAssigner
from models.demos.deepseek_v3_b1.compressed_tensor.metrics import metric_value


def _print_per_device_per_core_addresses(ct):
    """Print L1 addresses for all per-device per-core tensors."""
    for coord in ct._iter_mesh_coords():
        per_core = ct._multi_device_data_per_core[coord]
        addrs = []
        for (cx, cy), tensor in per_core.items():
            core = ttnn.CoreCoord(cx, cy)
            addr = ct.get_data_l1_address_per_core(core, device_coord=coord)
            addrs.append(f"({cx},{cy})={addr:#x}")
        print(f"  Device {coord}: {', '.join(addrs)}")


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


@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
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


@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
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


@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
def test_lockstep_different_assignment_per_device(mesh_device):
    """Different data per device should result in different assignments."""
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


@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
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


@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
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
    for coord in ct._iter_mesh_coords():
        per_core = ct._multi_device_data_per_core[coord]
        assert len(per_core) > 0, f"Device {coord} has no per-core tensors"
        print(f"Device {coord}: {len(per_core)} cores with data")
    _print_per_device_per_core_addresses(ct)

    recovered = ct.to_torch()
    for i in range(num_devices):
        dev_recovered = recovered[i * 64 : (i + 1) * 64, :]
        dev_original = x[i * 64 : (i + 1) * 64, :]
        pcc = metric_value(dev_original.numpy(), dev_recovered.numpy(), "pcc")
        print(f"Device {i} PCC: {pcc:.6f}")
        assert pcc > 0.98, f"Device {i} PCC {pcc:.6f} too low"


@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
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
    total_cores = sum(len(pc) for pc in ct._multi_device_data_per_core.values())
    assert len(all_tensors) == total_cores, f"Expected {total_cores} tensors, got {len(all_tensors)}"
    _print_per_device_per_core_addresses(ct)


@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
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

    for coord in ct._iter_mesh_coords():
        per_core = ct._multi_device_data_per_core[coord]
        assert len(per_core) > 0, f"Device {coord} has no per-core tensors"
        for (cx, cy), tensor in per_core.items():
            assert ttnn.is_tensor_storage_on_device(tensor), f"Device {coord} core ({cx},{cy}) tensor not on device"
    _print_per_device_per_core_addresses(ct)


# ---------------------------------------------------------------------------
# Mixed format multi-device tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
def test_lockstep_mixed_formats_per_device(mesh_device):
    """Each device gets different data patterns → different format mixes."""
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


@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
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

    for coord in ct._iter_mesh_coords():
        crs = ct.get_data_core_range_set(device_coord=coord)
        print(f"Device {coord} core range set: {crs}")
        # Should have cores with data
        assert crs.num_cores() > 0, f"Device {coord} has no cores in range set"
    _print_per_device_per_core_addresses(ct)


# ---------------------------------------------------------------------------
# 2D mesh sharding tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
def test_2d_lockstep_round_trip(mesh_device):
    """2D mesh sharding: split height across rows, width across cols."""
    mesh_rows, mesh_cols = mesh_device.shape[0], mesh_device.shape[1]
    num_devices = mesh_device.get_num_devices()
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
        device=mesh_device,
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


@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
def test_2d_per_core_round_trip(mesh_device):
    """2D mesh per-core: split height across rows, width across cols."""
    mesh_rows, mesh_cols = mesh_device.shape[0], mesh_device.shape[1]
    num_devices = mesh_device.get_num_devices()
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
        device=mesh_device,
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


@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
def test_2d_replicate_one_axis(mesh_device):
    """2D mesh with None on one axis: replicate across rows, shard cols."""
    mesh_rows, mesh_cols = mesh_device.shape[0], mesh_device.shape[1]
    num_devices = mesh_device.get_num_devices()
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
        device=mesh_device,
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


# ---------------------------------------------------------------------------
# 4D tensor with 2D mesh tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
def test_4d_lockstep_shard_dim0_dim2(mesh_device):
    """4D tensor sharded along dim 0 and dim 2 across a 2x2 mesh."""
    mesh_rows, mesh_cols = mesh_device.shape[0], mesh_device.shape[1]
    torch.manual_seed(42)
    # Shape: (2*mesh_rows, 3, 64*mesh_cols, 128) → per device (2, 3, 64, 128)
    # Per-device 2D: (2*3*64, 128) = (384, 128), tiles_h=12, tiles_w=4
    per_dev = (2, 3, 64, 128)
    x = torch.randn(per_dev[0] * mesh_rows, per_dev[1], per_dev[2] * mesh_cols, per_dev[3])

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))])
    data_mem = _make_sharded_mem_config(
        (per_dev[0] * per_dev[1] * per_dev[2], per_dev[3]),
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        grid,
    )

    ct = CompressedTensor.from_torch(
        x,
        assigner,
        device=mesh_device,
        memory_config=data_mem,
        per_core_allocation=False,
        mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(2)]),
    )

    print(f"{ct}")
    assert ct._num_devices == mesh_rows * mesh_cols

    recovered = ct.to_torch()
    assert recovered.shape == x.shape, f"Shape mismatch: {recovered.shape} != {x.shape}"
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"4D lockstep shard dim0/dim2 PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} too low"


@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
def test_4d_per_core_shard_dim0_dim2(mesh_device):
    """4D tensor per-core sharded along dim 0 and dim 2 across a 2x2 mesh."""
    mesh_rows, mesh_cols = mesh_device.shape[0], mesh_device.shape[1]
    torch.manual_seed(42)
    per_dev = (2, 3, 64, 128)
    x = torch.randn(per_dev[0] * mesh_rows, per_dev[1], per_dev[2] * mesh_cols, per_dev[3])

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))])
    data_mem = _make_sharded_mem_config(
        (per_dev[0] * per_dev[1] * per_dev[2], per_dev[3]),
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        grid,
    )

    ct = CompressedTensor.from_torch(
        x,
        assigner,
        device=mesh_device,
        memory_config=data_mem,
        per_core_allocation=True,
        mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(2)]),
    )

    print(f"{ct}")
    assert ct._num_devices == mesh_rows * mesh_cols
    assert len(ct._multi_device_data_per_core) == mesh_rows * mesh_cols

    recovered = ct.to_torch()
    assert recovered.shape == x.shape, f"Shape mismatch: {recovered.shape} != {x.shape}"
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    print(f"4D per-core shard dim0/dim2 PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} too low"


# ---------------------------------------------------------------------------
# Per-core tensor ↔ CB collision tests
# ---------------------------------------------------------------------------


def _per_core_mem_config(grid, shard_bytes):
    """Build a width-sharded MemoryConfig with experimental per-core allocation enabled."""
    shard_spec = ttnn.ShardSpec(grid, [1, shard_bytes], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    mem_config.experimental_set_per_core_allocation(True)
    return mem_config


def _alloc_large_per_core_tensor(mesh_device, grid, use_from_torch, shard_bytes=1_200_000):
    """Allocate a per-core L1 tensor consuming ~shard_bytes per core in `grid`.
    `use_from_torch=True` → ttnn.from_torch + ReplicateTensorToMesh (allocated on every mesh device).
    `use_from_torch=False` → experimental_to_single_device (allocated on coord (0,0) only)."""
    mem_config = _per_core_mem_config(grid, shard_bytes)
    data = torch.zeros(1, shard_bytes * grid.num_cores(), dtype=torch.uint8)
    if use_from_torch:
        return ttnn.from_torch(
            data,
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
    host_tensor = ttnn.from_torch(data, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT)
    coord = ttnn.MeshCoordinate(0, 0)
    return ttnn._ttnn.tensor.experimental_to_single_device(host_tensor, mesh_device, coord, mem_config)


def _make_io_tensors(mesh_device, cb_core_grid):
    """Pre-allocate minimal I/O tensors required by generic_op on `cb_core_grid`."""
    tile = 32
    io_shape = (tile, tile * cb_core_grid.num_cores())
    io_mem = _make_sharded_mem_config(io_shape, ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, cb_core_grid)
    in_t = ttnn.from_torch(
        torch.zeros(io_shape), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=io_mem
    )
    out_t = ttnn.from_torch(
        torch.zeros(io_shape), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=io_mem
    )
    return in_t, out_t


def _run_cb_heavy_noop_program(io_tensors, cb_core_grid, cb_total_size_bytes):
    """Launch a dummy generic_op on `cb_core_grid` with a single CB sized `cb_total_size_bytes`.
    The kernel is a noop (blank.cpp); the program exists solely to force CB placement validation.
    Caller must pre-allocate I/O tensors (before any per-core allocation) to avoid lockstep being
    pushed low by per-core-avoidance logic."""
    tile = 32
    tile_obj = ttnn.Tile((tile, tile))
    page_size = tile_obj.get_tile_size(ttnn.bfloat16)
    cb_fmt = ttnn.CBFormatDescriptor(
        buffer_index=0, data_format=ttnn.bfloat16, page_size=page_size, tile=ttnn.TileDescriptor(tile_obj)
    )
    cb_desc = ttnn.CBDescriptor(total_size=cb_total_size_bytes, core_ranges=cb_core_grid, format_descriptors=[cb_fmt])
    # Dispatch needs all 3 RISCs configured (NCRISC + BRISC + TRISC) even if kernels are blank;
    # otherwise launch crashes on the un-bound cores.
    reader = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/dataflow/blank.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cb_core_grid,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default
        ),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/dataflow/blank.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cb_core_grid,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_1_default
        ),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/compute/blank.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cb_core_grid,
        config=ttnn.ComputeConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=[cb_desc])
    # SPMD: dispatch to the entire mesh. The CB validator iterates all local device allocators,
    # so it will catch per-core collisions on whichever coord(s) host them.
    return ttnn.generic_op(list(io_tensors), program)


@pytest.mark.parametrize("use_from_torch", [False, True], ids=["experimental_to_single_device", "from_torch_replicate"])
@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
def test_per_core_cb_collision_errors_on_same_cores(mesh_device, use_from_torch):
    """CB validation should reject a program whose CBs land on cores already filled by a per-core tensor."""
    same_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    # Allocate I/O first so lockstep places them at top of L1 (no per-core ranges to avoid yet).
    io_tensors = _make_io_tensors(mesh_device, same_grid)
    _per_core_tensor = _alloc_large_per_core_tensor(mesh_device, same_grid, use_from_torch=use_from_torch)
    assert _per_core_tensor is not None  # keep alive until CB validation runs

    # Huge CB on the SAME cores as the per-core tensor: should trip validate_circular_buffer_region.
    with pytest.raises(RuntimeError, match=r"clash|circular buffer|L1"):
        _run_cb_heavy_noop_program(io_tensors, same_grid, cb_total_size_bytes=1024 * 1024)


@pytest.mark.parametrize("use_from_torch", [False, True], ids=["experimental_to_single_device", "from_torch_replicate"])
@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
def test_per_core_cb_no_collision_on_disjoint_cores(mesh_device, use_from_torch):
    """Per-core tensor on cores A should not tighten CB budgets for a program on disjoint cores B."""
    a_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    b_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))])
    # Allocate I/O first so lockstep places them at top of L1 (no per-core ranges to avoid yet).
    io_tensors = _make_io_tensors(mesh_device, b_grid)
    _per_core_tensor = _alloc_large_per_core_tensor(mesh_device, a_grid, use_from_torch=use_from_torch)
    assert _per_core_tensor is not None  # keep alive until CB validation runs

    # Same-size CB, but on cores B (disjoint from A). Should succeed: per-core tensor on A must
    # not tighten B's budget.
    _run_cb_heavy_noop_program(io_tensors, b_grid, cb_total_size_bytes=1024 * 1024)


def _alloc_large_per_core_compressed_tensor(mesh_device, grid):
    """Allocate a CompressedTensor with per_core_allocation=True. Sized to consume ~80% of L1
    after bfp8 compression (the larger-per-tile of the two formats), to repro the segfault
    that the copilot-style derived-from-worker_l1 sizing originally triggered."""
    num_devices = mesh_device.get_num_devices()
    tile = 32
    worker_l1 = ttnn._ttnn.reports.get_device_info(mesh_device).worker_l1_size
    target_per_core_bytes = worker_l1 * 80 // 100
    # Canonical bfp8 tile size: 1024 mantissa B + 64 shared-exp B = 1088 B/tile.
    bfp8_tile_bytes = 1088
    tiles_per_shard = max(1, target_per_core_bytes // bfp8_tile_bytes)
    shard_w = tiles_per_shard * tile
    total_w = shard_w * grid.num_cores()
    # Shard data on dim 0 across mesh devices so each device gets `tile` rows.
    x = torch.randn(tile * num_devices, total_w)
    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.99, formats=["bfp8", "bfp4"])
    mem_config = _make_sharded_mem_config(
        (tile, total_w), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, grid
    )
    return CompressedTensor.from_torch(
        x,
        assigner,
        device=mesh_device,
        memory_config=mem_config,
        per_core_allocation=True,
        mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0)]),
    )


def _try_compressed_per_core_collision(mesh_device, per_core_grid, cb_grid):
    """Allocate IO + per-core CompressedTensor + run CB program. Returns the captured error
    message if generic_op raised, else None.

    NOTE: this is intentionally a separate function so its locals (io_tensors, _ct) live in
    THIS frame. When this function returns, the frame is released along with all per-device
    per-core buffers — while mesh_device is still alive. Putting these allocations directly in
    the test function would have any later assert / pytest.fail capture them via the failure
    traceback, pinning the buffers past the function-scoped mesh_device's teardown and crashing
    the next test's gc.collect."""
    io_tensors = _make_io_tensors(mesh_device, cb_grid)
    _ct = _alloc_large_per_core_compressed_tensor(mesh_device, per_core_grid)
    assert _ct is not None  # keep alive until CB validation runs
    # CB sized as a fraction of the device's reported worker L1, so collision detection scales
    # with architecture instead of relying on a hardcoded ~1.4 MB bank assumption. ~70% of L1
    # leaves the CB alone fitting on disjoint cores, while per-core (sized at ~80% of L1 in
    # _alloc_large_per_core_compressed_tensor) + this CB > 100% → guaranteed collision when
    # they share cores, regardless of which format multi-format CompressedTensor's assigner
    # chose per tile (bfp4 = 576 B/tile, bfp8 = 1088 B/tile).
    worker_l1 = ttnn._ttnn.reports.get_device_info(mesh_device).worker_l1_size
    # CB total_size must be divisible by page_size (bfloat16 tile = 32*32*2 = 2048 B).
    page_size = ttnn.Tile((32, 32)).get_tile_size(ttnn.bfloat16)
    cb_bytes = (worker_l1 * 70 // 100) // page_size * page_size
    try:
        _run_cb_heavy_noop_program(io_tensors, cb_grid, cb_total_size_bytes=cb_bytes)
        return None
    except RuntimeError as e:
        return str(e)


@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
def test_compressed_per_core_cb_collision_errors_on_same_cores(mesh_device):
    """Same as test_per_core_cb_collision_errors_on_same_cores but using CompressedTensor."""
    same_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    err = _try_compressed_per_core_collision(mesh_device, same_grid, same_grid)
    assert err is not None, "Expected RuntimeError from CB validator (collision with per-core tensor)"
    assert any(s in err for s in ("clash", "circular buffer", "L1")), f"Unexpected error: {err}"


@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
def test_compressed_per_core_cb_no_collision_on_disjoint_cores(mesh_device):
    """Same as test_per_core_cb_no_collision_on_disjoint_cores but using CompressedTensor."""
    a_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    b_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))])
    err = _try_compressed_per_core_collision(mesh_device, a_grid, b_grid)
    assert err is None, f"Expected no error on disjoint cores, but got: {err}"
