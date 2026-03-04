# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Roundtrip tests for overlapped tensor serialization
(dump_overlapped_tensors / load_overlapped_tensors).

Creates OverlappedTensor views via overlap_tensors, dumps them to disk,
loads them back, and verifies that both the metadata (shapes, dtypes,
core ranges, byte offsets) and raw tensor data survive the roundtrip.
"""

import pathlib

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_b1.blitz_overlap_tensors import OverlappedShardSpec, OverlappedTensor, overlap_tensors


def _core_range_set_eq(a: ttnn.CoreRangeSet, b: ttnn.CoreRangeSet) -> bool:
    """Compare two CoreRangeSets by their sorted range tuples."""

    def _key(crs):
        return sorted((r.start.x, r.start.y, r.end.x, r.end.y) for r in crs.ranges())

    return _key(a) == _key(b)


def _assert_overlapped_metadata_eq(original: OverlappedTensor, loaded: OverlappedTensor, key: str | int):
    """Assert all metadata fields of two OverlappedTensor views match."""
    assert (
        original.tensor_shape == loaded.tensor_shape
    ), f"view[{key}] tensor_shape: {original.tensor_shape} != {loaded.tensor_shape}"
    assert (
        original.shard_shape == loaded.shard_shape
    ), f"view[{key}] shard_shape: {original.shard_shape} != {loaded.shard_shape}"
    assert _core_range_set_eq(original.core_range_set, loaded.core_range_set), f"view[{key}] core_range_set mismatch"
    assert original.dtype == loaded.dtype, f"view[{key}] dtype: {original.dtype} != {loaded.dtype}"
    assert (
        original.tile_shape == loaded.tile_shape
    ), f"view[{key}] tile_shape: {original.tile_shape} != {loaded.tile_shape}"
    assert (
        original.byte_offset == loaded.byte_offset
    ), f"view[{key}] byte_offset: {original.byte_offset} != {loaded.byte_offset}"


def test_overlapped_tensor_roundtrip_single_lane(tmp_path, device):
    """Roundtrip: two BFP8 sub-tensors on same cores, single lane."""
    torch.manual_seed(0)

    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    t1 = torch.randn(512, 512, dtype=torch.bfloat16)
    t2 = torch.randn(256, 512, dtype=torch.bfloat16)

    spec1 = OverlappedShardSpec(core_range_set=crs, raw_tensor_shape=(512, 512), dtype=ttnn.bfloat8_b)
    spec2 = OverlappedShardSpec(core_range_set=crs, raw_tensor_shape=(256, 512), dtype=ttnn.bfloat8_b)

    views = overlap_tensors([[("t1", t1, spec1), ("t2", t2, spec2)]], device=device)
    assert len(views) == 2

    file_name = str(tmp_path / "single_lane.overlappedtensorbin")
    ttnn._ttnn.tensor.dump_overlapped_tensors(file_name, views)

    loaded = ttnn._ttnn.tensor.load_overlapped_tensors(file_name)
    assert len(loaded) == len(views)
    assert set(loaded.keys()) == set(views.keys())

    for name in views:
        _assert_overlapped_metadata_eq(views[name], loaded[name], name)

    orig_torch = ttnn.to_torch(views["t1"].fused_tensor)
    loaded_torch = ttnn.to_torch(loaded["t1"].fused_tensor)
    assert orig_torch.shape == loaded_torch.shape
    assert torch.equal(orig_torch, loaded_torch)


def test_overlapped_tensor_roundtrip_multi_lane(tmp_path, device):
    """Roundtrip: two lanes with different core ranges and dtypes."""
    torch.manual_seed(42)

    crs_a = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    crs_b = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 0))})

    t_bfp8 = torch.randn(512, 512, dtype=torch.bfloat16)
    t_bf16 = torch.randn(128, 128, dtype=torch.bfloat16)

    spec_bfp8 = OverlappedShardSpec(core_range_set=crs_a, raw_tensor_shape=(512, 512), dtype=ttnn.bfloat8_b)
    spec_bf16 = OverlappedShardSpec(core_range_set=crs_b, raw_tensor_shape=(128, 128), dtype=ttnn.bfloat16)

    views = overlap_tensors(
        [[("bfp8", t_bfp8, spec_bfp8)], [("bf16", t_bf16, spec_bf16)]],
        device=device,
    )
    assert len(views) == 2

    file_name = str(tmp_path / "multi_lane.overlappedtensorbin")
    ttnn._ttnn.tensor.dump_overlapped_tensors(file_name, views)

    loaded = ttnn._ttnn.tensor.load_overlapped_tensors(file_name)
    assert len(loaded) == len(views)
    assert set(loaded.keys()) == set(views.keys())

    for name in views:
        _assert_overlapped_metadata_eq(views[name], loaded[name], name)

    orig_torch = ttnn.to_torch(views["bfp8"].fused_tensor)
    loaded_torch = ttnn.to_torch(loaded["bfp8"].fused_tensor)
    assert torch.equal(orig_torch, loaded_torch)


def test_overlapped_tensor_roundtrip_to_device(tmp_path, device):
    """Roundtrip with load-to-device: loads directly onto device."""
    torch.manual_seed(7)

    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    t1 = torch.randn(256, 512, dtype=torch.bfloat16)
    spec1 = OverlappedShardSpec(core_range_set=crs, raw_tensor_shape=(256, 512), dtype=ttnn.bfloat8_b)

    views = overlap_tensors([[("t1", t1, spec1)]], device=device)

    file_name = str(tmp_path / "to_device.overlappedtensorbin")
    ttnn._ttnn.tensor.dump_overlapped_tensors(file_name, views)

    loaded = ttnn._ttnn.tensor.load_overlapped_tensors(file_name, device=device)
    assert len(loaded) == 1
    assert loaded["t1"].fused_tensor.is_allocated()

    _assert_overlapped_metadata_eq(views["t1"], loaded["t1"], "t1")

    orig_torch = ttnn.to_torch(views["t1"].fused_tensor)
    loaded_torch = ttnn.to_torch(loaded["t1"].fused_tensor)
    assert torch.equal(orig_torch, loaded_torch)


def test_overlapped_tensor_roundtrip_height_sharded(tmp_path, device):
    """Roundtrip with HEIGHT_SHARDED sub-tensors."""
    torch.manual_seed(13)

    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    t1 = torch.randn(512, 256, dtype=torch.bfloat16)
    spec1 = OverlappedShardSpec(
        core_range_set=crs,
        raw_tensor_shape=(512, 256),
        dtype=ttnn.bfloat8_b,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )

    views = overlap_tensors([[("t1", t1, spec1)]], device=device)

    file_name = str(tmp_path / "height_sharded.overlappedtensorbin")
    ttnn._ttnn.tensor.dump_overlapped_tensors(file_name, views)

    loaded = ttnn._ttnn.tensor.load_overlapped_tensors(file_name)
    assert len(loaded) == 1
    _assert_overlapped_metadata_eq(views["t1"], loaded["t1"], "t1")

    orig_torch = ttnn.to_torch(views["t1"].fused_tensor)
    loaded_torch = ttnn.to_torch(loaded["t1"].fused_tensor)
    assert torch.equal(orig_torch, loaded_torch)


def test_overlapped_tensor_roundtrip_mixed_tiles(tmp_path, device):
    """Roundtrip with mixed tile sizes (32x32 BFP8 + 1x32 BF16 gamma)."""
    torch.manual_seed(99)

    crs_main = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    crs_gamma = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 0))})

    t_main = torch.randn(256, 512, dtype=torch.bfloat16)
    t_gamma = torch.randn(1, 32, dtype=torch.bfloat16)

    spec_main = OverlappedShardSpec(core_range_set=crs_main, raw_tensor_shape=(256, 512), dtype=ttnn.bfloat8_b)
    spec_gamma = OverlappedShardSpec(
        core_range_set=crs_gamma,
        raw_tensor_shape=(1, 32),
        dtype=ttnn.bfloat16,
        tile_h=1,
        tile_w=32,
    )

    views = overlap_tensors(
        [[("main", t_main, spec_main)], [("gamma", t_gamma, spec_gamma)]],
        device=device,
    )
    assert len(views) == 2

    file_name = str(tmp_path / "mixed_tiles.overlappedtensorbin")
    ttnn._ttnn.tensor.dump_overlapped_tensors(file_name, views)

    loaded = ttnn._ttnn.tensor.load_overlapped_tensors(file_name)
    assert len(loaded) == 2
    assert set(loaded.keys()) == set(views.keys())

    for name in views:
        _assert_overlapped_metadata_eq(views[name], loaded[name], name)

    orig_torch = ttnn.to_torch(views["main"].fused_tensor)
    loaded_torch = ttnn.to_torch(loaded["main"].fused_tensor)
    assert torch.equal(orig_torch, loaded_torch)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_overlapped_tensor_roundtrip_tp_4x2(tmp_path, bh_2d_mesh_device):
    """Roundtrip serialization with TP on a 4x2 mesh.

    Creates two sub-tensors on the same core range:
      - t1: replicated across all devices (no TP)
      - t2: TP-sharded along width across mesh columns (tp_dim=(None, 1))

    The fused tensor has different data on each mesh column. Verifies
    that after dump+load, each device gets back its original data.
    """
    mesh = bh_2d_mesh_device
    if mesh.shape[0] * mesh.shape[1] < 8:
        pytest.skip("Test requires 4x2 mesh (8 devices)")

    submesh = mesh.create_submesh(ttnn.MeshShape((4, 2)))
    mesh_rows, mesh_cols = 4, 2

    torch.manual_seed(42)

    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})

    t1 = torch.randn(512, 512, dtype=torch.bfloat16)
    t2 = torch.randn(256, 1024, dtype=torch.bfloat16)

    spec1 = OverlappedShardSpec(core_range_set=crs, raw_tensor_shape=(512, 512), dtype=ttnn.bfloat8_b)
    spec2 = OverlappedShardSpec(
        core_range_set=crs,
        raw_tensor_shape=(256, 1024),
        dtype=ttnn.bfloat8_b,
        tp_dim=(None, 1),
    )

    views = overlap_tensors([[("t1", t1, spec1), ("t2", t2, spec2)]], device=submesh)
    assert len(views) == 2

    file_name = str(tmp_path / "tp_4x2.overlappedtensorbin")
    ttnn._ttnn.tensor.dump_overlapped_tensors(file_name, views)

    loaded = ttnn._ttnn.tensor.load_overlapped_tensors(file_name, device=submesh)
    assert len(loaded) == len(views)
    assert set(loaded.keys()) == set(views.keys())

    for name in views:
        _assert_overlapped_metadata_eq(views[name], loaded[name], name)

    assert loaded["t1"].fused_tensor.is_allocated()

    orig_shards = ttnn.get_device_tensors(views["t1"].fused_tensor)
    loaded_shards = ttnn.get_device_tensors(loaded["t1"].fused_tensor)
    assert len(orig_shards) == len(loaded_shards) == mesh_rows * mesh_cols

    for dev_idx in range(len(orig_shards)):
        orig_cpu = ttnn.to_torch(orig_shards[dev_idx])
        loaded_cpu = ttnn.to_torch(loaded_shards[dev_idx])
        assert torch.equal(orig_cpu, loaded_cpu), f"Data mismatch on device {dev_idx}"

    ttnn.close_mesh_device(submesh)
