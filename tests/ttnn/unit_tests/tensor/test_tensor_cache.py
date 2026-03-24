# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for the content-addressed TensorCache (lazy cache).

Validates the get_or_create API:
  - cache miss: fuse callable is invoked, views are stored to disk
  - cache hit: fuse callable is NOT invoked, views are loaded from disk
  - metadata (shapes, dtypes, offsets) survives the round-trip
  - different fingerprints produce separate cache entries
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_b1.blitz_overlap_tensors import OverlappedShardSpec, OverlappedTensor, overlap_tensors
from models.demos.deepseek_v3_b1.tensor_cache import Fingerprint, TensorCache


def _core_range_set_eq(a: ttnn.CoreRangeSet, b: ttnn.CoreRangeSet) -> bool:
    def _key(crs):
        return sorted((r.start.x, r.start.y, r.end.x, r.end.y) for r in crs.ranges())

    return _key(a) == _key(b)


def _assert_overlapped_metadata_eq(original: OverlappedTensor, loaded: OverlappedTensor, key: str):
    assert original.tensor_shape == loaded.tensor_shape, f"[{key}] tensor_shape mismatch"
    assert original.shard_shape == loaded.shard_shape, f"[{key}] shard_shape mismatch"
    assert _core_range_set_eq(original.core_range_set, loaded.core_range_set), f"[{key}] core_range_set mismatch"
    assert original.dtype == loaded.dtype, f"[{key}] dtype mismatch"
    assert original.tile_shape == loaded.tile_shape, f"[{key}] tile_shape mismatch"
    assert original.byte_offset == loaded.byte_offset, f"[{key}] byte_offset mismatch"
    assert original.total_size == loaded.total_size, f"[{key}] total_size mismatch"


def _make_fingerprint(spec1: OverlappedShardSpec, spec2: OverlappedShardSpec, **overrides) -> Fingerprint:
    defaults = dict(
        schema_version=1,
        hf_model_id="test-model",
        hf_revision="abc123",
        transform_version=1,
        mesh_shape=(1, 1),
        group_name="test_group",
        layer_idx=0,
        spec_fingerprints=(spec1.fingerprint(), spec2.fingerprint()),
    )
    defaults.update(overrides)
    return Fingerprint(**defaults)


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_cache_miss_then_hit(tmp_path, device, dtype):
    """First get_or_create is a miss (fuse called), second is a hit (fuse NOT called)."""
    torch.manual_seed(42)

    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    t1 = torch.randn(512, 512, dtype=torch.bfloat16)
    t2 = torch.randn(256, 512, dtype=torch.bfloat16)
    spec1 = OverlappedShardSpec(core_range_set=crs, raw_tensor_shape=(512, 512), dtype=dtype)
    spec2 = OverlappedShardSpec(core_range_set=crs, raw_tensor_shape=(256, 512), dtype=dtype)

    fp = _make_fingerprint(spec1, spec2)
    cache = TensorCache(tmp_path / "cache")

    fuse_count = [0]

    def fuse():
        fuse_count[0] += 1
        return overlap_tensors([[("t1", t1, spec1), ("t2", t2, spec2)]], device=device)

    # Miss: fuse is called
    views_miss = cache.get_or_create(fp, fuse=fuse, device=device)
    assert fuse_count[0] == 1
    assert set(views_miss.keys()) == {"t1", "t2"}

    # Hit: fuse is NOT called
    views_hit = cache.get_or_create(fp, fuse=fuse, device=device)
    assert fuse_count[0] == 1, "fuse should not be called on cache hit"
    assert set(views_hit.keys()) == {"t1", "t2"}

    for name in views_miss:
        _assert_overlapped_metadata_eq(views_miss[name], views_hit[name], name)


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_cache_data_survives_roundtrip(tmp_path, device, dtype):
    """Tensor data loaded from cache matches the original."""
    torch.manual_seed(42)

    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    t1 = torch.randn(512, 512, dtype=torch.bfloat16)
    spec1 = OverlappedShardSpec(core_range_set=crs, raw_tensor_shape=(512, 512), dtype=dtype)
    spec2 = OverlappedShardSpec(core_range_set=crs, raw_tensor_shape=(512, 512), dtype=dtype)

    fp = _make_fingerprint(spec1, spec2)
    cache = TensorCache(tmp_path / "cache")

    views_miss = cache.get_or_create(
        fp,
        fuse=lambda: overlap_tensors([[("t1", t1, spec1)]], device=device),
        device=device,
    )
    orig_data = ttnn.to_torch(views_miss["t1"].fused_tensor)

    views_hit = cache.get_or_create(
        fp,
        fuse=lambda: (_ for _ in ()).throw(AssertionError("should not be called")),
        device=device,
    )
    loaded_data = ttnn.to_torch(views_hit["t1"].fused_tensor)

    assert orig_data.shape == loaded_data.shape
    assert torch.equal(orig_data, loaded_data)


def test_different_fingerprints_separate_entries(tmp_path, device):
    """Different fingerprints produce separate cache entries."""
    torch.manual_seed(42)

    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    t1 = torch.randn(512, 512, dtype=torch.bfloat16)
    t2 = torch.randn(512, 512, dtype=torch.bfloat16)
    spec = OverlappedShardSpec(core_range_set=crs, raw_tensor_shape=(512, 512), dtype=ttnn.bfloat8_b)

    cache = TensorCache(tmp_path / "cache")

    fp_v1 = _make_fingerprint(spec, spec, transform_version=1)
    fp_v2 = _make_fingerprint(spec, spec, transform_version=2)

    assert fp_v1.artifact_id() != fp_v2.artifact_id()

    fuse_count = [0]

    def fuse_v1():
        fuse_count[0] += 1
        return overlap_tensors([[("t1", t1, spec)]], device=device)

    def fuse_v2():
        fuse_count[0] += 1
        return overlap_tensors([[("t1", t2, spec)]], device=device)

    cache.get_or_create(fp_v1, fuse=fuse_v1, device=device)
    assert fuse_count[0] == 1

    cache.get_or_create(fp_v2, fuse=fuse_v2, device=device)
    assert fuse_count[0] == 2, "different fingerprint should cause a miss"

    # Both should now be cached
    cache.get_or_create(fp_v1, fuse=fuse_v1, device=device)
    cache.get_or_create(fp_v2, fuse=fuse_v2, device=device)
    assert fuse_count[0] == 2, "both entries should be cached"


def test_fingerprint_deterministic():
    """Same spec fields produce the same fingerprint hash."""
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    spec_a = OverlappedShardSpec(core_range_set=crs, raw_tensor_shape=(512, 512), dtype=ttnn.bfloat8_b)
    spec_b = OverlappedShardSpec(core_range_set=crs, raw_tensor_shape=(512, 512), dtype=ttnn.bfloat8_b)

    assert spec_a.fingerprint() == spec_b.fingerprint()

    fp_a = Fingerprint(
        schema_version=1,
        hf_model_id="test",
        hf_revision="abc",
        transform_version=1,
        mesh_shape=(4, 2),
        group_name="test",
        layer_idx=0,
        spec_fingerprints=(spec_a.fingerprint(),),
    )
    fp_b = Fingerprint(
        schema_version=1,
        hf_model_id="test",
        hf_revision="abc",
        transform_version=1,
        mesh_shape=(4, 2),
        group_name="test",
        layer_idx=0,
        spec_fingerprints=(spec_b.fingerprint(),),
    )
    assert fp_a.artifact_id() == fp_b.artifact_id()


def test_fingerprint_changes_with_dtype():
    """Different dtype produces different fingerprint."""
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    spec_bf8 = OverlappedShardSpec(core_range_set=crs, raw_tensor_shape=(512, 512), dtype=ttnn.bfloat8_b)
    spec_bf16 = OverlappedShardSpec(core_range_set=crs, raw_tensor_shape=(512, 512), dtype=ttnn.bfloat16)

    assert spec_bf8.fingerprint() != spec_bf16.fingerprint()


def test_standalone_tensor_cache_miss_then_hit(tmp_path, device):
    """get_or_create_tensor: miss calls create, hit loads from disk."""
    torch.manual_seed(42)

    t = torch.randn(32, 64, dtype=torch.bfloat16)
    cache = TensorCache(tmp_path / "cache")

    fp = Fingerprint(
        schema_version=1,
        hf_model_id="test",
        hf_revision="abc",
        transform_version=1,
        mesh_shape=(1, 1),
        group_name="standalone_test",
        layer_idx=0,
        spec_fingerprints=(),
    )

    create_count = [0]

    def create():
        create_count[0] += 1
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    result_miss = cache.get_or_create_tensor(fp, create=create, device=device)
    assert create_count[0] == 1

    result_hit = cache.get_or_create_tensor(fp, create=create, device=device)
    assert create_count[0] == 1, "create should not be called on cache hit"

    miss_torch = ttnn.to_torch(result_miss)
    hit_torch = ttnn.to_torch(result_hit)
    assert torch.equal(miss_torch, hit_torch)
