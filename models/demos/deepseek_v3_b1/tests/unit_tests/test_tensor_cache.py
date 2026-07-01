# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the content-addressed TensorCache.

Covers standalone tensors (`TensorTarget`) and fusion groups
(`FusionGroupSpec`): fingerprint determinism and sensitivity, CAS layout,
miss/hit round-trips (including fused artifacts and corrupt recovery),
content hash in metadata, and `CacheContext` helpers.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

import ttnn
from conftest import requires_hybrid_allocator
from models.demos.deepseek_v3_b1.weights.cache.cache import (
    AbsentCacheEntry,
    CorruptCacheEntry,
    EphemeralTensorCache,
    PresentCacheEntry,
    TensorCache,
)
from models.demos.deepseek_v3_b1.weights.cache.fingerprint import canonical, compute_artifact_id
from models.demos.deepseek_v3_b1.weights.cache.sram_compressed_cache import (
    assigner_fingerprint,
    get_or_create_sram_compressed_expert,
    to_canonical_mesh_mapper,
)
from models.demos.deepseek_v3_b1.weights.cache.types import (
    CacheContext,
    Fingerprint,
    FusionGroupSpec,
    RegionSpec,
    ReplicateMeshMapper,
    Shard2dMeshMapper,
    ShardMeshMapper,
    SourceTensorSelection,
    SramCompressedTensorTarget,
    TensorTarget,
)
from models.demos.deepseek_v3_b1.weights.overlap.packing import OverlappedTensor
from models.demos.deepseek_v3_b1.weights.overlap.spec import OverlappedTensorSpec


def _make_fingerprint(**overrides) -> Fingerprint:
    """Build a Fingerprint with sensible defaults; override any field via kwargs."""
    defaults = dict(
        schema_version=1,
        source=SourceTensorSelection(names=("weight_a", "weight_b")),
        hf_model_id="deepseek-ai/DeepSeek-V3",
        hf_revision="d1a891dd58e6bb0a671bfc6f3046e29e3478e924",
        mesh_shape=(4, 2),
        target=TensorTarget(
            name="test_tensor",
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tile_shape=(32, 32),
        ),
    )
    defaults.update(overrides)
    return Fingerprint(**defaults)


class TestFingerprint:
    def test_determinism(self):
        """Same inputs produce the same artifact_id across calls."""
        fingerprint1 = _make_fingerprint()
        fingerprint2 = _make_fingerprint()
        assert compute_artifact_id(fingerprint1) == compute_artifact_id(fingerprint2)

    def test_canonical_is_dict(self):
        fingerprint = _make_fingerprint()
        c = canonical(fingerprint)
        assert isinstance(c, dict)
        assert c["schema_version"] == 1
        assert c["source"] == ["weight_a", "weight_b"]
        assert c["target"]["kind"] == "tensor"
        assert c["target"]["tile_shape"] == [32, 32]
        assert c["target"]["mesh_mapper_config"] == {"strategy": "replicate", "dim": None, "dims": None}

    def test_canonical_source_sorted(self):
        """Source names are sorted in canonical form regardless of input order."""
        fingerprint = _make_fingerprint(source=SourceTensorSelection(names=("z_weight", "a_weight")))
        c = canonical(fingerprint)
        assert c["source"] == ["a_weight", "z_weight"]

    def test_sensitivity_source(self):
        fingerprint1 = _make_fingerprint(source=SourceTensorSelection(names=("a",)))
        fingerprint2 = _make_fingerprint(source=SourceTensorSelection(names=("b",)))
        assert compute_artifact_id(fingerprint1) != compute_artifact_id(fingerprint2)

    def test_sensitivity_tensor_target_transform_version(self):
        t1 = TensorTarget(name="t", transform_version=1)
        t2 = TensorTarget(name="t", transform_version=2)
        fp1 = _make_fingerprint(target=t1)
        fp2 = _make_fingerprint(target=t2)
        assert compute_artifact_id(fp1) != compute_artifact_id(fp2)

    def test_sensitivity_fusion_group_transform_version(self):
        fg1 = _sample_fusion_group_spec(transform_version=1)
        fg2 = _sample_fusion_group_spec(transform_version=2)
        fp1 = _make_fingerprint(target=fg1)
        fp2 = _make_fingerprint(target=fg2)
        assert compute_artifact_id(fp1) != compute_artifact_id(fp2)

    def test_sensitivity_mesh_shape(self):
        fingerprint1 = _make_fingerprint(mesh_shape=(4, 2))
        fingerprint2 = _make_fingerprint(mesh_shape=(8, 4))
        assert compute_artifact_id(fingerprint1) != compute_artifact_id(fingerprint2)

    def test_sensitivity_dtype(self):
        t1 = TensorTarget(name="t", dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        t2 = TensorTarget(name="t", dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        fingerprint1 = _make_fingerprint(target=t1)
        fingerprint2 = _make_fingerprint(target=t2)
        assert compute_artifact_id(fingerprint1) != compute_artifact_id(fingerprint2)

    def test_sensitivity_tile_shape(self):
        t1 = TensorTarget(name="t", tile_shape=(32, 32))
        t2 = TensorTarget(name="t", tile_shape=(16, 16))
        fingerprint1 = _make_fingerprint(target=t1)
        fingerprint2 = _make_fingerprint(target=t2)
        assert compute_artifact_id(fingerprint1) != compute_artifact_id(fingerprint2)

    def test_sensitivity_hf_revision(self):
        fingerprint1 = _make_fingerprint(hf_revision="aaa")
        fingerprint2 = _make_fingerprint(hf_revision="bbb")
        assert compute_artifact_id(fingerprint1) != compute_artifact_id(fingerprint2)

    def test_sensitivity_schema_version(self):
        fingerprint1 = _make_fingerprint(schema_version=1)
        fingerprint2 = _make_fingerprint(schema_version=2)
        assert compute_artifact_id(fingerprint1) != compute_artifact_id(fingerprint2)

    def test_sensitivity_name(self):
        t1 = TensorTarget(name="gate_bias")
        t2 = TensorTarget(name="shared_down_proj")
        fingerprint1 = _make_fingerprint(target=t1)
        fingerprint2 = _make_fingerprint(target=t2)
        assert compute_artifact_id(fingerprint1) != compute_artifact_id(fingerprint2)

    def test_sensitivity_mesh_mapper_config(self):
        t1 = TensorTarget(name="t", mesh_mapper_config=ReplicateMeshMapper())
        t2 = TensorTarget(name="t", mesh_mapper_config=ShardMeshMapper(dim=1))
        t3 = TensorTarget(name="t", mesh_mapper_config=Shard2dMeshMapper(dims=(0, 1)))
        t4 = TensorTarget(name="t", mesh_mapper_config=Shard2dMeshMapper(dims=(None, 1)))
        fingerprint1 = _make_fingerprint(target=t1)
        fingerprint2 = _make_fingerprint(target=t2)
        fingerprint3 = _make_fingerprint(target=t3)
        fingerprint4 = _make_fingerprint(target=t4)
        ids = {
            compute_artifact_id(fingerprint1),
            compute_artifact_id(fingerprint2),
            compute_artifact_id(fingerprint3),
            compute_artifact_id(fingerprint4),
        }
        assert len(ids) == 4

    def test_canonical_includes_mesh_mapper_config(self):
        t = TensorTarget(name="t", mesh_mapper_config=ShardMeshMapper(dim=1))
        fingerprint = _make_fingerprint(target=t)
        c = canonical(fingerprint)
        assert c["target"]["mesh_mapper_config"] == {"strategy": "shard", "dim": 1, "dims": None}

    def test_canonical_mesh_mapper_config_shard_2d(self):
        t = TensorTarget(name="t", mesh_mapper_config=Shard2dMeshMapper(dims=(0, 1)))
        fingerprint = _make_fingerprint(target=t)
        c = canonical(fingerprint)
        assert c["target"]["mesh_mapper_config"] == {"strategy": "shard_2d", "dim": None, "dims": [0, 1]}

    def test_canonical_mesh_mapper_config_shard_2d_with_none_dim(self):
        t = TensorTarget(name="t", mesh_mapper_config=Shard2dMeshMapper(dims=(None, 1)))
        fingerprint = _make_fingerprint(target=t)
        c = canonical(fingerprint)
        assert c["target"]["mesh_mapper_config"] == {"strategy": "shard_2d", "dim": None, "dims": [None, 1]}

    def test_artifact_id_is_hex_sha256(self):
        fingerprint = _make_fingerprint()
        artifact_id = compute_artifact_id(fingerprint)
        assert len(artifact_id) == 64
        int(artifact_id, 16)  # valid hex


class TestCasLayout:
    def test_store_uses_local_dump_mode(self, tmp_path, monkeypatch):
        """_store should call ttnn.dump_tensor with LOCAL mode."""
        cache = TensorCache(tmp_path)
        fingerprint = _make_fingerprint()
        artifact_id = compute_artifact_id(fingerprint)
        tensor_host = ttnn.from_torch(
            torch.randn(4, 4, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tile=ttnn.Tile((32, 32)),
        )

        called = {"mode": None}

        def _dump_tensor_spy(file_name, tensor, *, mode):
            called["mode"] = mode
            # Write placeholder bytes so _store can hash/stat the file.
            with open(file_name, "wb") as f:
                f.write(b"tensorbin-placeholder")

        monkeypatch.setattr(ttnn, "dump_tensor", _dump_tensor_spy)
        cache._store(artifact_id, fingerprint, tensor_host)
        assert called["mode"] == ttnn.DumpTensorMode.LOCAL

    def test_store_creates_expected_files(self, tmp_path):
        """After _store, the object dir contains data.tensorbin, manifest.json, metadata.json."""
        cache = TensorCache(tmp_path)
        fingerprint = _make_fingerprint()
        artifact_id = compute_artifact_id(fingerprint)

        tensor_host = ttnn.from_torch(
            torch.randn(4, 4, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tile=ttnn.Tile((32, 32)),
        )
        paths = cache._store(artifact_id, fingerprint, tensor_host)

        assert paths.object_dir.is_dir()
        assert paths.data_path.is_file()
        assert (paths.object_dir / "manifest.json").is_file()
        assert (paths.object_dir / "metadata.json").is_file()

    def test_store_directory_structure(self, tmp_path):
        """Object is stored under objects/{id[:2]}/{id}/."""
        cache = TensorCache(tmp_path)
        fingerprint = _make_fingerprint()
        artifact_id = compute_artifact_id(fingerprint)

        tensor_host = ttnn.from_torch(
            torch.randn(4, 4, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tile=ttnn.Tile((32, 32)),
        )
        cache._store(artifact_id, fingerprint, tensor_host)

        expected_dir = tmp_path / "objects" / artifact_id[:2] / artifact_id
        assert expected_dir.is_dir()

    def test_content_hash_matches(self, tmp_path):
        """metadata.json content_hash matches SHA-256 of data.tensorbin."""
        cache = TensorCache(tmp_path)
        fingerprint = _make_fingerprint()
        artifact_id = compute_artifact_id(fingerprint)

        tensor_host = ttnn.from_torch(
            torch.randn(4, 4, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tile=ttnn.Tile((32, 32)),
        )
        paths = cache._store(artifact_id, fingerprint, tensor_host)

        with open(paths.object_dir / "metadata.json") as f:
            meta = json.load(f)
        stored_hash = meta["content_hash"]

        h = hashlib.sha256()
        with open(paths.data_path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        assert stored_hash == h.hexdigest()

    def test_manifest_contains_fingerprint(self, tmp_path):
        """manifest.json contains the canonical fingerprint."""
        cache = TensorCache(tmp_path)
        fingerprint = _make_fingerprint()
        artifact_id = compute_artifact_id(fingerprint)

        tensor_host = ttnn.from_torch(
            torch.randn(4, 4, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tile=ttnn.Tile((32, 32)),
        )
        paths = cache._store(artifact_id, fingerprint, tensor_host)

        with open(paths.object_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["fingerprint"] == canonical(fingerprint)
        assert manifest["logical_name"] == "test_tensor"


class TestLookup:
    def test_absent(self, tmp_path):
        cache = TensorCache(tmp_path)
        entry = cache._lookup("deadbeef" * 8)
        assert isinstance(entry, AbsentCacheEntry)

    def test_present_after_store(self, tmp_path):
        cache = TensorCache(tmp_path)
        fingerprint = _make_fingerprint()
        artifact_id = compute_artifact_id(fingerprint)

        tensor_host = ttnn.from_torch(
            torch.randn(4, 4, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tile=ttnn.Tile((32, 32)),
        )
        cache._store(artifact_id, fingerprint, tensor_host)
        entry = cache._lookup(artifact_id)
        assert isinstance(entry, PresentCacheEntry)

    def test_corrupt_when_data_missing(self, tmp_path):
        """If the object dir exists but data.tensorbin is missing, entry is corrupt."""
        cache = TensorCache(tmp_path)
        fingerprint = _make_fingerprint()
        artifact_id = compute_artifact_id(fingerprint)

        tensor_host = ttnn.from_torch(
            torch.randn(4, 4, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tile=ttnn.Tile((32, 32)),
        )
        paths = cache._store(artifact_id, fingerprint, tensor_host)
        paths.data_path.unlink()

        entry = cache._lookup(artifact_id)
        assert isinstance(entry, CorruptCacheEntry)


class TestGetOrCreate:
    @pytest.fixture()
    def cache_dir(self, tmp_path):
        return tmp_path / "tensor_cache"

    def test_miss_then_hit(self, cache_dir, device):
        """First call is a miss (preprocess called), second is a hit (preprocess not called)."""
        cache = TensorCache(cache_dir)

        raw_data = torch.randn(16, 16, dtype=torch.bfloat16)
        preprocess_call_count = [0]

        def preprocess(tensors):
            preprocess_call_count[0] += 1
            return {"gate_bias": tensors["raw"].reshape(16, 16).T.contiguous()}

        fingerprint = _make_fingerprint(
            source=SourceTensorSelection(names=("mlp.gate.e_score_correction_bias",)),
            target=TensorTarget(
                name="gate_bias",
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                tile_shape=(32, 32),
            ),
        )

        result1 = cache.get_or_create(
            fingerprint,
            device,
            preprocess=preprocess,
            raw_tensors=lambda: {"raw": raw_data},
        )
        assert preprocess_call_count[0] == 1
        assert isinstance(result1, ttnn.Tensor)

        result2 = cache.get_or_create(
            fingerprint,
            device,
            preprocess=preprocess,
            raw_tensors=lambda: {"raw": raw_data},
        )
        assert preprocess_call_count[0] == 1  # preprocess NOT called again

        ttnn.deallocate(result1, force=True)
        ttnn.deallocate(result2, force=True)

    def test_raw_tensors_not_called_on_hit(self, cache_dir, device):
        """raw_tensors callable should not be invoked on a cache hit."""
        cache = TensorCache(cache_dir)
        raw_data = torch.randn(16, 16, dtype=torch.bfloat16)

        fingerprint = _make_fingerprint(
            source=SourceTensorSelection(names=("w",)),
            target=TensorTarget(name="t", dtype=ttnn.bfloat16, tile_shape=(32, 32)),
        )

        result1 = cache.get_or_create(
            fingerprint, device, preprocess=lambda d: {"t": d["w"]}, raw_tensors=lambda: {"w": raw_data}
        )

        raw_called = [False]

        def raw_tensors_spy():
            raw_called[0] = True
            return {"w": raw_data}

        result2 = cache.get_or_create(
            fingerprint, device, preprocess=lambda d: {"t": d["w"]}, raw_tensors=raw_tensors_spy
        )
        assert not raw_called[0]

        ttnn.deallocate(result1, force=True)
        ttnn.deallocate(result2, force=True)

    def test_corrupt_recovery(self, cache_dir, device):
        """Deleting data.tensorbin should cause a transparent rebuild on next access."""
        cache = TensorCache(cache_dir)
        raw_data = torch.randn(16, 16, dtype=torch.bfloat16)

        fingerprint = _make_fingerprint(
            source=SourceTensorSelection(names=("w",)),
            target=TensorTarget(name="t", dtype=ttnn.bfloat16, tile_shape=(32, 32)),
        )

        result1 = cache.get_or_create(
            fingerprint, device, preprocess=lambda d: {"t": d["w"]}, raw_tensors=lambda: {"w": raw_data}
        )
        ttnn.deallocate(result1, force=True)

        artifact_id = compute_artifact_id(fingerprint)
        data_path = cache._content_addressed_paths(artifact_id).data_path
        assert data_path.is_file()
        data_path.unlink()

        preprocess_calls = [0]

        def counting_preprocess(d):
            preprocess_calls[0] += 1
            return {"t": d["w"]}

        result2 = cache.get_or_create(
            fingerprint, device, preprocess=counting_preprocess, raw_tensors=lambda: {"w": raw_data}
        )
        assert preprocess_calls[0] == 1
        assert isinstance(result2, ttnn.Tensor)
        ttnn.deallocate(result2, force=True)

    def test_different_fingerprints_different_artifacts(self, cache_dir, device):
        """Two different fingerprints produce two separate cache entries."""
        cache = TensorCache(cache_dir)

        fingerprint1 = _make_fingerprint(
            source=SourceTensorSelection(names=("a",)),
            target=TensorTarget(name="t", dtype=ttnn.bfloat16, tile_shape=(32, 32)),
        )
        fingerprint2 = _make_fingerprint(
            source=SourceTensorSelection(names=("b",)),
            target=TensorTarget(name="t", dtype=ttnn.bfloat16, tile_shape=(32, 32)),
        )

        data = torch.randn(16, 16, dtype=torch.bfloat16)
        identity = lambda d: {"t": d["w"]}
        raw = lambda: {"w": data}

        r1 = cache.get_or_create(fingerprint1, device, preprocess=identity, raw_tensors=raw)
        r2 = cache.get_or_create(fingerprint2, device, preprocess=identity, raw_tensors=raw)

        artifact_id_1 = compute_artifact_id(fingerprint1)
        artifact_id_2 = compute_artifact_id(fingerprint2)
        assert artifact_id_1 != artifact_id_2
        assert isinstance(cache._lookup(artifact_id_1), PresentCacheEntry)
        assert isinstance(cache._lookup(artifact_id_2), PresentCacheEntry)

        ttnn.deallocate(r1, force=True)
        ttnn.deallocate(r2, force=True)

    def test_dict_raw_tensors(self, cache_dir, device):
        """raw_tensors can be a plain dict instead of a callable."""
        cache = TensorCache(cache_dir)

        fingerprint = _make_fingerprint(
            source=SourceTensorSelection(names=("w",)),
            target=TensorTarget(name="t", dtype=ttnn.bfloat16, tile_shape=(32, 32)),
        )
        data = torch.randn(16, 16, dtype=torch.bfloat16)

        result = cache.get_or_create(fingerprint, device, preprocess=lambda d: {"t": d["w"]}, raw_tensors={"w": data})
        assert isinstance(result, ttnn.Tensor)
        ttnn.deallocate(result, force=True)

    def test_row_major_round_trip(self, cache_dir, device):
        """ROW_MAJOR_LAYOUT tensors round-trip through cache without tile kwarg."""
        cache = TensorCache(cache_dir)

        target = TensorTarget(
            name="emb",
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fingerprint = _make_fingerprint(
            source=SourceTensorSelection(names=("model.embed_tokens.weight",)),
            target=target,
        )
        data = torch.randn(64, 32, dtype=torch.bfloat16)

        result = cache.get_or_create(
            fingerprint, device, preprocess=lambda d: {"emb": d["w"].contiguous()}, raw_tensors={"w": data}
        )
        assert isinstance(result, ttnn.Tensor)

        result2 = cache.get_or_create(
            fingerprint, device, preprocess=lambda d: {"emb": d["w"].contiguous()}, raw_tensors={"w": data}
        )
        assert isinstance(result2, ttnn.Tensor)

        ttnn.deallocate(result, force=True)
        ttnn.deallocate(result2, force=True)


class TestCacheContext:
    def test_factory_produces_valid_fingerprint(self):
        ctx = CacheContext(
            schema_version=1,
            hf_model_id="deepseek-ai/DeepSeek-V3",
            hf_revision="abc123",
            mesh_shape=(4, 2),
        )
        target = TensorTarget(name="embedding", dtype=ttnn.bfloat16, transform_version=1)
        source = SourceTensorSelection(names=("model.embed_tokens.weight",))
        fingerprint = ctx.fingerprint(source=source, target=target)

        assert isinstance(fingerprint, Fingerprint)
        assert fingerprint.schema_version == 1
        assert fingerprint.hf_model_id == "deepseek-ai/DeepSeek-V3"
        assert fingerprint.hf_revision == "abc123"
        assert fingerprint.mesh_shape == (4, 2)
        assert fingerprint.target is target
        assert fingerprint.source is source

    def test_different_contexts_different_ids(self):
        ctx1 = CacheContext(schema_version=1, hf_model_id="m", hf_revision="a", mesh_shape=(4, 2))
        ctx2 = CacheContext(schema_version=1, hf_model_id="m", hf_revision="b", mesh_shape=(4, 2))
        target = TensorTarget(name="t")
        source = SourceTensorSelection(names=("w",))
        fingerprint1 = ctx1.fingerprint(source=source, target=target)
        fingerprint2 = ctx2.fingerprint(source=source, target=target)
        assert compute_artifact_id(fingerprint1) != compute_artifact_id(fingerprint2)

    def test_same_context_same_ids(self):
        ctx = CacheContext(schema_version=1, hf_model_id="m", hf_revision="a", mesh_shape=(4, 2))
        target = TensorTarget(name="t")
        source = SourceTensorSelection(names=("w",))
        fingerprint1 = ctx.fingerprint(source=source, target=target)
        fingerprint2 = ctx.fingerprint(source=source, target=target)
        assert compute_artifact_id(fingerprint1) == compute_artifact_id(fingerprint2)


def _sample_fusion_group_spec(**overrides) -> FusionGroupSpec:
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    region = RegionSpec(
        core_range_set=crs,
        subtensors=(
            OverlappedTensorSpec(name="w", core_range_set=crs, raw_tensor_shape=(64, 32), dtype=ttnn.bfloat16),
        ),
    )
    defaults = dict(
        name="test_fusion",
        regions=(region,),
        sharding_strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        mesh_mapper_config=ReplicateMeshMapper(),
    )
    defaults.update(overrides)
    return FusionGroupSpec(**defaults)


UNITTEST_TOY_FUSION_SPEC = FusionGroupSpec(
    name="_unittest_toy",
    regions=(),
    mesh_mapper_config=ReplicateMeshMapper(),
)


def _create_unittest_toy(
    preprocessed: dict[str, torch.Tensor],
    device,
    move_to_device: bool,
) -> tuple[ttnn.Tensor, dict[str, OverlappedTensor]]:
    """Two 32x32 bfloat16 tiles stacked vertically on one WIDTH_SHARDED core; TILE layout."""
    for k in ("a", "b"):
        if k not in preprocessed:
            raise KeyError("preprocessed must contain 'a' and 'b' for _unittest_toy")
        t = preprocessed[k]
        if tuple(t.shape) != (32, 32):
            raise ValueError(f"_unittest_toy expects (32,32) tensors, got {k}={tuple(t.shape)}")

    a = preprocessed["a"].to(dtype=torch.bfloat16).contiguous()
    b = preprocessed["b"].to(dtype=torch.bfloat16).contiguous()
    combined = torch.cat([a, b], dim=0)
    assert combined.shape == (64, 32)

    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(crs, (64, 32), ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    mesh_mapper = ttnn.ReplicateTensorToMesh(device)
    device_for_torch = device if move_to_device else None

    fused = ttnn.from_torch(
        combined,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_for_torch,
        memory_config=mem_config,
        mesh_mapper=mesh_mapper,
        tile=ttnn.Tile((32, 32)),
    )

    tile = fused.get_tile()
    ts = tuple(tile.tile_shape)
    tile_bytes = tile.get_tile_size(ttnn.bfloat16)
    total_one = tile_bytes

    v_a = OverlappedTensor(
        fused_tensor=fused,
        tensor_shape=(32, 32),
        shard_shape=(32, 32),
        core_range_set=crs,
        dtype=ttnn.bfloat16,
        tile_shape=ts,
        byte_offset=0,
        total_size=total_one,
    )
    v_b = OverlappedTensor(
        fused_tensor=fused,
        tensor_shape=(32, 32),
        shard_shape=(32, 32),
        core_range_set=crs,
        dtype=ttnn.bfloat16,
        tile_shape=ts,
        byte_offset=total_one,
        total_size=total_one,
    )
    return fused, {"a": v_a, "b": v_b}


def _create_overlapped_tensor_fused_with_toy(
    spec,
    preprocessed,
    device,
    *,
    move_to_device=False,
):
    """Wrapper that handles _unittest_toy locally, delegates production specs to the library."""
    if spec.name == "_unittest_toy":
        return _create_unittest_toy(preprocessed, device, move_to_device)
    from models.demos.deepseek_v3_b1.weights.cache.fuse import create_overlapped_tensor

    return create_overlapped_tensor(spec, preprocessed, device, move_to_device=move_to_device)


class TestFusionGroupFingerprint:
    def test_fusion_canonical_is_dict(self):
        fp = _make_fingerprint(target=_sample_fusion_group_spec())
        c = canonical(fp)
        assert c["target"]["kind"] == "fusion_group"
        assert c["target"]["name"] == "test_fusion"
        assert c["target"]["sharding_strategy"] == "WIDTH_SHARDED"
        assert c["target"]["regions"][0]["subtensors"][0]["overlap_priority"] is None

    def test_fusion_determinism(self):
        fp1 = _make_fingerprint(target=_sample_fusion_group_spec())
        fp2 = _make_fingerprint(target=_sample_fusion_group_spec())
        assert compute_artifact_id(fp1) == compute_artifact_id(fp2)

    def test_fusion_sensitivity_name(self):
        fp1 = _make_fingerprint(target=_sample_fusion_group_spec(name="a"))
        fp2 = _make_fingerprint(target=_sample_fusion_group_spec(name="b"))
        assert compute_artifact_id(fp1) != compute_artifact_id(fp2)

    def test_fusion_sensitivity_sharding_strategy(self):
        fp1 = _make_fingerprint(
            target=_sample_fusion_group_spec(sharding_strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED)
        )
        fp2 = _make_fingerprint(
            target=_sample_fusion_group_spec(sharding_strategy=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
        )
        assert compute_artifact_id(fp1) != compute_artifact_id(fp2)

    def test_fusion_sensitivity_mesh_mapper(self):
        fp1 = _make_fingerprint(target=_sample_fusion_group_spec(mesh_mapper_config=ReplicateMeshMapper()))
        fp2 = _make_fingerprint(target=_sample_fusion_group_spec(mesh_mapper_config=ShardMeshMapper(dim=1)))
        assert compute_artifact_id(fp1) != compute_artifact_id(fp2)

    def test_fusion_sensitivity_overlap_priority(self):
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        base = dict(name="w", core_range_set=crs, raw_tensor_shape=(64, 32), dtype=ttnn.bfloat16)
        st_lo = OverlappedTensorSpec(**base, overlap_priority=0)
        st_hi = OverlappedTensorSpec(**base, overlap_priority=1)
        fp1 = _make_fingerprint(
            target=FusionGroupSpec(name="test_fusion", regions=(RegionSpec(core_range_set=crs, subtensors=(st_lo,)),))
        )
        fp2 = _make_fingerprint(
            target=FusionGroupSpec(name="test_fusion", regions=(RegionSpec(core_range_set=crs, subtensors=(st_hi,)),))
        )
        assert compute_artifact_id(fp1) != compute_artifact_id(fp2)


class TestFusionGroupPerCore:
    """:class:`FusionGroupSpec.per_core` authoring + fingerprint invariants."""

    def _single_region(self, name: str) -> RegionSpec:
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        st = OverlappedTensorSpec(name=name, core_range_set=crs, raw_tensor_shape=(64, 32), dtype=ttnn.bfloat16)
        return RegionSpec(core_range_set=crs, subtensors=(st,))

    def test_default_per_core_is_false(self):
        region = self._single_region("w0")
        spec = FusionGroupSpec(name="legacy", regions=(region,))
        assert spec.per_core is False

    def test_canonical_includes_per_core_when_false(self):
        region = self._single_region("w0")
        spec = FusionGroupSpec(name="legacy", regions=(region,))
        fp = _make_fingerprint(target=spec)
        c = canonical(fp)
        assert c["target"]["per_core"] is False

    def test_canonical_includes_per_core_when_true(self):
        region = self._single_region("w0")
        spec = FusionGroupSpec(name="pc", regions=(region,), per_core=True)
        fp = _make_fingerprint(target=spec)
        c = canonical(fp)
        assert c["target"]["per_core"] is True

    def test_fingerprint_sensitivity_per_core(self):
        region = self._single_region("w0")
        spec_off = FusionGroupSpec(name="g", regions=(region,), per_core=False)
        spec_on = FusionGroupSpec(name="g", regions=(region,), per_core=True)
        assert compute_artifact_id(_make_fingerprint(target=spec_off)) != compute_artifact_id(
            _make_fingerprint(target=spec_on)
        )


class TestCreateOverlappedTensorUnittestToy:
    def test_toy_produces_views(self, device):
        a = torch.randn(32, 32, dtype=torch.bfloat16)
        b = torch.randn(32, 32, dtype=torch.bfloat16)
        fused, views = _create_unittest_toy({"a": a, "b": b}, device, move_to_device=True)
        assert "a" in views and "b" in views
        assert views["a"].byte_offset == 0
        assert views["b"].byte_offset == views["a"].total_size
        assert views["a"].fused_tensor == fused
        ttnn.deallocate(fused, force=True)


class TestGetOrCreateFused:
    @pytest.fixture()
    def cache_dir(self, tmp_path):
        return tmp_path / "tensor_cache_fused"

    @pytest.fixture(autouse=True)
    def _patch_fuse(self, monkeypatch):
        """Route _unittest_toy through the local factory instead of the library fuse module."""
        import models.demos.deepseek_v3_b1.weights.cache.cache as _cache_mod

        monkeypatch.setattr(_cache_mod, "_create_overlapped_tensor_fused", _create_overlapped_tensor_fused_with_toy)

    def test_miss_then_hit(self, cache_dir, device):
        from models.demos.deepseek_v3_b1.weights.cache.cache import TensorCache

        cache = TensorCache(cache_dir)
        preprocess_calls = [0]

        def preprocess(tensors):
            preprocess_calls[0] += 1
            return {"a": tensors["a"].clone(), "b": tensors["b"].clone()}

        ctx = CacheContext(
            schema_version=1,
            hf_model_id="deepseek-ai/DeepSeek-V3",
            hf_revision="d1a891dd58e6bb0a671bfc6f3046e29e3478e924",
            mesh_shape=(4, 2),
        )
        fingerprint = ctx.fingerprint(
            source=SourceTensorSelection(names=("src.a", "src.b")),
            target=UNITTEST_TOY_FUSION_SPEC,
        )

        raw_a = torch.randn(32, 32, dtype=torch.bfloat16)
        raw_b = torch.randn(32, 32, dtype=torch.bfloat16)

        v1 = cache.get_or_create(
            fingerprint,
            device,
            preprocess=preprocess,
            raw_tensors=lambda: {"a": raw_a, "b": raw_b},
        )
        assert preprocess_calls[0] == 1
        assert set(v1.keys()) == {"a", "b"}

        v2 = cache.get_or_create(
            fingerprint,
            device,
            preprocess=preprocess,
            raw_tensors=lambda: {"a": raw_a, "b": raw_b},
        )
        assert preprocess_calls[0] == 1

        ttnn.deallocate(v1["a"].fused_tensor, force=True)
        ttnn.deallocate(v2["a"].fused_tensor, force=True)

    def test_corrupt_fused_recovery(self, cache_dir, device):
        from models.demos.deepseek_v3_b1.weights.cache.cache import TensorCache

        cache = TensorCache(cache_dir)
        ctx = CacheContext(
            schema_version=1,
            hf_model_id="m",
            hf_revision="r",
            mesh_shape=(4, 2),
        )
        fingerprint = ctx.fingerprint(
            source=SourceTensorSelection(names=("x",)),
            target=UNITTEST_TOY_FUSION_SPEC,
        )
        raw_a = torch.randn(32, 32, dtype=torch.bfloat16)
        raw_b = torch.randn(32, 32, dtype=torch.bfloat16)

        v1 = cache.get_or_create(
            fingerprint,
            device,
            preprocess=lambda d: {"a": d["a"], "b": d["b"]},
            raw_tensors=lambda: {"a": raw_a, "b": raw_b},
        )
        ttnn.deallocate(v1["a"].fused_tensor, force=True)

        artifact_id = compute_artifact_id(fingerprint)
        data_path = cache._content_addressed_paths(artifact_id).data_path
        assert data_path.is_file()
        data_path.unlink()

        preprocess_calls = [0]

        def counting_preprocess(d):
            preprocess_calls[0] += 1
            return {"a": d["a"], "b": d["b"]}

        v2 = cache.get_or_create(
            fingerprint,
            device,
            preprocess=counting_preprocess,
            raw_tensors=lambda: {"a": raw_a, "b": raw_b},
        )
        assert preprocess_calls[0] == 1
        ttnn.deallocate(v2["a"].fused_tensor, force=True)


# ---------------------------------------------------------------------------
# SRAM hot-expert CompressedTensor cache (post-pack byte cache)
# ---------------------------------------------------------------------------


def _make_sram_assigner(formats=("bfp8", "bfp4"), threshold: float = 0.993):
    """Construct a CompressedTensorAssigner with the given formats."""
    from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensorAssigner

    return CompressedTensorAssigner(metric="pcc", threshold=threshold, formats=list(formats))


def _make_sram_l1_per_core_mem_config(K: int, N: int, core_grid: ttnn.CoreRangeSet) -> ttnn.MemoryConfig:
    """Build a WIDTH_SHARDED L1 mem_config matching ``build_sram_routed_proj_ct`` shape rules."""
    num_cores = core_grid.num_cores()
    assert N % num_cores == 0, f"N ({N}) must divide num_cores ({num_cores})"
    per_core_N = N // num_cores
    assert per_core_N % 32 == 0, f"per_core_N ({per_core_N}) must be tile-aligned"
    shard_spec = ttnn.ShardSpec(core_grid, [K, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


def _build_l1_core_grid(device, num_cores: int) -> ttnn.CoreRangeSet:
    """Pick *num_cores* compute cores from the device, skipping known DRAM workers."""
    grid = device.compute_with_storage_grid_size()
    dram_workers = {(0, 0), (0, 3), (0, 7), (0, 9), (7, 1), (7, 4), (7, 6), (7, 9)}
    cores: list[ttnn.CoreCoord] = []
    for y in range(grid.y):
        for x in range(grid.x):
            if (x, y) in dram_workers:
                continue
            cores.append(ttnn.CoreCoord(x, y))
            if len(cores) == num_cores:
                break
        if len(cores) == num_cores:
            break
    assert len(cores) == num_cores, f"Need {num_cores} non-DRAM cores, got {len(cores)}"
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _make_sram_target(
    *,
    name: str = "sram_test_layer3_expert0_gate_proj",
    tensor_shape: tuple[int, int] = (128, 128),
    tile_hw: int = 32,
    memory_config: ttnn.MemoryConfig | None = None,
    per_core_allocation: bool = True,
    mesh_mapper_config=ReplicateMeshMapper(),
    assigner_fingerprint: str = "deadbeefcafef00d",
    assignment_hash: str = "",
    transform_version: int = 1,
) -> SramCompressedTensorTarget:
    """Build a SramCompressedTensorTarget with sensible defaults; override fields via kwargs."""
    return SramCompressedTensorTarget(
        name=name,
        tensor_shape=tensor_shape,
        tile_hw=tile_hw,
        memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        per_core_allocation=per_core_allocation,
        mesh_mapper_config=mesh_mapper_config,
        assigner_fingerprint=assigner_fingerprint,
        assignment_hash=assignment_hash,
        transform_version=transform_version,
    )


def _make_sram_fingerprint(target: SramCompressedTensorTarget, *, source_names=("w_a", "w_b")) -> Fingerprint:
    """Build a Fingerprint over a SramCompressedTensorTarget with single-device defaults."""
    return Fingerprint(
        schema_version=1,
        source=SourceTensorSelection(names=tuple(source_names)),
        hf_model_id="deepseek-v3-test",
        hf_revision="rev-sram",
        mesh_shape=(1, 1),
        target=target,
    )


class TestSramCompressedFingerprint:
    """Host-only fingerprint tests for ``SramCompressedTensorTarget`` (no device)."""

    def test_canonical_shape(self):
        """Canonical form is a deterministic dict carrying every cache-key field."""
        target = _make_sram_target()
        fingerprint = _make_sram_fingerprint(target)
        c = canonical(fingerprint)
        assert c["target"]["kind"] == "sram_compressed_tensor"
        assert c["target"]["name"] == target.name
        assert c["target"]["tensor_shape"] == list(target.tensor_shape)
        assert c["target"]["tile_hw"] == target.tile_hw
        assert c["target"]["per_core_allocation"] is True
        assert c["target"]["mesh_mapper_config"] == {"strategy": "replicate", "dim": None, "dims": None}
        assert c["target"]["assigner_fingerprint"] == target.assigner_fingerprint
        assert c["target"]["assignment_hash"] == ""
        assert c["target"]["transform_version"] == 1

    def test_determinism(self):
        """Same inputs → same artifact_id across calls."""
        target_a = _make_sram_target()
        target_b = _make_sram_target()
        assert compute_artifact_id(_make_sram_fingerprint(target_a)) == compute_artifact_id(
            _make_sram_fingerprint(target_b)
        )

    def test_sensitive_to_assigner_fingerprint(self):
        """Different assigner config → different artifact_id (assignment bytes will differ)."""
        a = _make_sram_target(assigner_fingerprint="aaaaaaaaaaaaaaaa")
        b = _make_sram_target(assigner_fingerprint="bbbbbbbbbbbbbbbb")
        assert compute_artifact_id(_make_sram_fingerprint(a)) != compute_artifact_id(_make_sram_fingerprint(b))

    def test_sensitive_to_assignment_hash(self):
        """Different BSPM assignment bytes → different artifact_id."""
        a = _make_sram_target(assigner_fingerprint="", assignment_hash="0011223344556677")
        b = _make_sram_target(assigner_fingerprint="", assignment_hash="ffeeddccbbaa9988")
        assert compute_artifact_id(_make_sram_fingerprint(a)) != compute_artifact_id(_make_sram_fingerprint(b))

    def test_sensitive_to_transform_version(self):
        """Bumping transform_version invalidates existing artifacts."""
        v1 = _make_sram_target(transform_version=1)
        v2 = _make_sram_target(transform_version=2)
        assert compute_artifact_id(_make_sram_fingerprint(v1)) != compute_artifact_id(_make_sram_fingerprint(v2))

    def test_sensitive_to_mesh_mapper(self):
        """Replicate vs Shard vs Shard2d produce distinct fingerprints."""
        rep = _make_sram_target(mesh_mapper_config=ReplicateMeshMapper())
        sh0 = _make_sram_target(mesh_mapper_config=ShardMeshMapper(dim=0))
        sh01 = _make_sram_target(mesh_mapper_config=Shard2dMeshMapper(dims=(0, 1)))
        ids = {
            compute_artifact_id(_make_sram_fingerprint(rep)),
            compute_artifact_id(_make_sram_fingerprint(sh0)),
            compute_artifact_id(_make_sram_fingerprint(sh01)),
        }
        assert len(ids) == 3

    def test_sensitive_to_tensor_shape(self):
        """Different tensor_shape → different artifact_id (e.g. K=128 vs K=256 layouts)."""
        a = _make_sram_target(tensor_shape=(128, 128))
        b = _make_sram_target(tensor_shape=(256, 128))
        assert compute_artifact_id(_make_sram_fingerprint(a)) != compute_artifact_id(_make_sram_fingerprint(b))

    def test_name_is_part_of_key(self):
        """``name`` is part of the cache key — distinct experts must not collide."""
        a = _make_sram_target(name="sram_layer3_expert0_gate_proj")
        b = _make_sram_target(name="sram_layer3_expert1_gate_proj")
        assert compute_artifact_id(_make_sram_fingerprint(a)) != compute_artifact_id(_make_sram_fingerprint(b))


class TestAssignerFingerprint:
    """``assigner_fingerprint`` digest behaviour (host-only)."""

    def test_none_returns_empty(self):
        assert assigner_fingerprint(None) == ""

    def test_same_config_same_digest(self):
        a = _make_sram_assigner(formats=("bfp8", "bfp4"), threshold=0.99)
        b = _make_sram_assigner(formats=("bfp8", "bfp4"), threshold=0.99)
        assert assigner_fingerprint(a) == assigner_fingerprint(b)

    def test_format_order_invariant(self):
        """Format order doesn't affect per-tile decisions, so the digest is order-invariant."""
        a = _make_sram_assigner(formats=("bfp8", "bfp4"))
        b = _make_sram_assigner(formats=("bfp4", "bfp8"))
        assert assigner_fingerprint(a) == assigner_fingerprint(b)

    def test_different_formats_different_digest(self):
        a = _make_sram_assigner(formats=("bfp8", "bfp4"))
        b = _make_sram_assigner(formats=("bfp4",))
        assert assigner_fingerprint(a) != assigner_fingerprint(b)

    def test_different_threshold_different_digest(self):
        a = _make_sram_assigner(threshold=0.993)
        b = _make_sram_assigner(threshold=0.999)
        assert assigner_fingerprint(a) != assigner_fingerprint(b)


class TestCanonicalMeshMapper:
    """``to_canonical_mesh_mapper`` translation from runtime ``ttnn.MeshMapperConfig`` to dataclass form."""

    def test_none_to_replicate(self):
        assert to_canonical_mesh_mapper(None) == ReplicateMeshMapper()

    def test_single_shard(self):
        mc = ttnn.MeshMapperConfig([ttnn.PlacementShard(0)])
        assert to_canonical_mesh_mapper(mc) == ShardMeshMapper(dim=0)

    def test_single_replicate_placement(self):
        """A single PlacementReplicate maps back to the canonical ReplicateMeshMapper."""
        mc = ttnn.MeshMapperConfig([ttnn.PlacementReplicate()])
        assert to_canonical_mesh_mapper(mc) == ReplicateMeshMapper()

    def test_shard_2d(self):
        mc = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)])
        assert to_canonical_mesh_mapper(mc) == Shard2dMeshMapper(dims=(0, 1))

    def test_two_replicate_placements_normalize_to_replicate(self):
        """[Replicate, Replicate] is degenerate — collapse to ReplicateMeshMapper."""
        mc = ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementReplicate()])
        assert to_canonical_mesh_mapper(mc) == ReplicateMeshMapper()


@requires_hybrid_allocator
class TestSramCompressedRoundTrip:
    """End-to-end disk round-trip for SRAM hot-expert CompressedTensors via ``get_or_create_sram_compressed_expert``.

    The whole class is gated on ``TT_METAL_ALLOCATOR_MODE_HYBRID=1`` because every test below builds a
    ``per_core_allocation=True`` WIDTH_SHARDED L1 ``CompressedTensor``, which the C++ allocator only
    supports under hybrid mode.  Without the marker the tests would crash with a TT_FATAL deep inside
    ``ttnn.from_torch`` instead of skipping politely.
    """

    K = 128
    N = 128
    NUM_CORES = 4  # N=128 / 4 = 32 → tile-aligned per_core_N

    @pytest.fixture()
    def cache_dir(self, tmp_path):
        return tmp_path / "sram_compressed_cache"

    def _build_fingerprint(
        self,
        device,
        *,
        assigner,
        memory_config,
        source_names=("expert0_gate_proj",),
        name: str = "sram_layer3_expert0_gate_proj",
    ) -> Fingerprint:
        target = SramCompressedTensorTarget(
            name=name,
            tensor_shape=(self.K, self.N),
            tile_hw=32,
            memory_config=memory_config,
            per_core_allocation=True,
            mesh_mapper_config=ReplicateMeshMapper(),
            assigner_fingerprint=assigner_fingerprint(assigner),
            assignment_hash="",
            transform_version=1,
        )
        return Fingerprint(
            schema_version=1,
            source=SourceTensorSelection(names=tuple(source_names)),
            hf_model_id="deepseek-v3-test",
            hf_revision="rev-sram",
            mesh_shape=(1, 1),
            target=target,
        )

    def test_miss_then_hit(self, cache_dir, device):
        """Cold call writes shards.bin + metadata.json; warm call reads them with no extra writes."""
        torch.manual_seed(0)
        weight = torch.randn(self.K, self.N)

        cache = TensorCache(cache_dir)
        core_grid = _build_l1_core_grid(device, self.NUM_CORES)
        memory_config = _make_sram_l1_per_core_mem_config(self.K, self.N, core_grid)
        assigner = _make_sram_assigner()
        fp = self._build_fingerprint(device, assigner=assigner, memory_config=memory_config)

        weight_calls = [0]

        def weight_provider():
            weight_calls[0] += 1
            return weight

        ct_miss = get_or_create_sram_compressed_expert(
            cache,
            fp,
            device,
            weight_provider=weight_provider,
            assigner=assigner,
            memory_config=memory_config,
            per_core_allocation=True,
        )
        assert weight_calls[0] == 1, "cold path must consume the weight provider"

        artifact_id = compute_artifact_id(fp)
        obj_dir = cache_dir / "objects" / artifact_id[:2] / artifact_id
        shards_path = obj_dir / "shards.bin"
        meta_path = obj_dir / "metadata.json"
        assert shards_path.is_file(), "cold miss must write shards.bin"
        assert meta_path.is_file(), "cold miss must write metadata.json"
        assert shards_path.stat().st_size > 0, "shards.bin must contain packed bytes"

        # Warm hit — same fingerprint, same cache → no new writes, no weight access.
        weight_calls[0] = 0
        snapshot_mtime = shards_path.stat().st_mtime_ns
        ct_hit = get_or_create_sram_compressed_expert(
            cache,
            fp,
            device,
            weight_provider=weight_provider,
            assigner=assigner,
            memory_config=memory_config,
            per_core_allocation=True,
        )
        assert weight_calls[0] == 0, "warm hit must not invoke the weight provider"
        assert shards_path.stat().st_mtime_ns == snapshot_mtime, "warm hit must not rewrite shards.bin"

        # Reconstructed assignment is byte-identical to the cold pack's.
        assert np.array_equal(ct_miss._assignment_flat, ct_hit._assignment_flat)
        assert ct_miss.shape == ct_hit.shape

    def test_corrupt_recovery(self, cache_dir, device):
        """Deleting shards.bin causes a transparent rebuild on next access."""
        torch.manual_seed(1)
        weight = torch.randn(self.K, self.N)

        cache = TensorCache(cache_dir)
        core_grid = _build_l1_core_grid(device, self.NUM_CORES)
        memory_config = _make_sram_l1_per_core_mem_config(self.K, self.N, core_grid)
        assigner = _make_sram_assigner()
        fp = self._build_fingerprint(device, assigner=assigner, memory_config=memory_config)

        get_or_create_sram_compressed_expert(
            cache,
            fp,
            device,
            weight_provider=lambda: weight,
            assigner=assigner,
            memory_config=memory_config,
            per_core_allocation=True,
        )

        artifact_id = compute_artifact_id(fp)
        obj_dir = cache_dir / "objects" / artifact_id[:2] / artifact_id
        shards_path = obj_dir / "shards.bin"
        assert shards_path.is_file()
        shards_path.unlink()

        weight_calls = [0]

        def weight_provider():
            weight_calls[0] += 1
            return weight

        ct = get_or_create_sram_compressed_expert(
            cache,
            fp,
            device,
            weight_provider=weight_provider,
            assigner=assigner,
            memory_config=memory_config,
            per_core_allocation=True,
        )
        assert weight_calls[0] == 1, "corrupted entry must trigger a cold rebuild"
        assert shards_path.is_file(), "rebuild must repopulate shards.bin"
        assert ct._assignment_flat is not None

    def test_different_fingerprints_distinct_artifacts(self, cache_dir, device):
        """Two fingerprints (e.g. distinct expert names) produce two separate cache entries."""
        torch.manual_seed(2)
        weight_a = torch.randn(self.K, self.N)
        weight_b = torch.randn(self.K, self.N)

        cache = TensorCache(cache_dir)
        core_grid = _build_l1_core_grid(device, self.NUM_CORES)
        memory_config = _make_sram_l1_per_core_mem_config(self.K, self.N, core_grid)
        assigner = _make_sram_assigner()

        fp_a = self._build_fingerprint(
            device,
            assigner=assigner,
            memory_config=memory_config,
            source_names=("expert0_gate_proj",),
            name="sram_layer3_expert0_gate_proj",
        )
        fp_b = self._build_fingerprint(
            device,
            assigner=assigner,
            memory_config=memory_config,
            source_names=("expert1_gate_proj",),
            name="sram_layer3_expert1_gate_proj",
        )
        assert compute_artifact_id(fp_a) != compute_artifact_id(fp_b)

        get_or_create_sram_compressed_expert(
            cache,
            fp_a,
            device,
            weight_provider=lambda: weight_a,
            assigner=assigner,
            memory_config=memory_config,
            per_core_allocation=True,
        )
        get_or_create_sram_compressed_expert(
            cache,
            fp_b,
            device,
            weight_provider=lambda: weight_b,
            assigner=assigner,
            memory_config=memory_config,
            per_core_allocation=True,
        )

        shards = sorted((cache_dir / "objects").rglob("shards.bin"))
        assert len(shards) == 2, f"expected 2 distinct cache entries, found {len(shards)}"

    def test_ephemeral_cache_skips_disk(self, cache_dir, device):
        """``EphemeralTensorCache`` short-circuits to a cold build with no on-disk artifact."""
        torch.manual_seed(3)
        weight = torch.randn(self.K, self.N)

        ephemeral = EphemeralTensorCache(move_to_device=True)
        core_grid = _build_l1_core_grid(device, self.NUM_CORES)
        memory_config = _make_sram_l1_per_core_mem_config(self.K, self.N, core_grid)
        assigner = _make_sram_assigner()
        fp = self._build_fingerprint(device, assigner=assigner, memory_config=memory_config)

        ct = get_or_create_sram_compressed_expert(
            ephemeral,
            fp,
            device,
            weight_provider=lambda: weight,
            assigner=assigner,
            memory_config=memory_config,
            per_core_allocation=True,
        )
        assert ct._assignment_flat is not None
        # ``cache_dir`` is reserved by the fixture but never written: ephemeral path bypasses disk entirely.
        assert not (cache_dir / "objects").exists()

    def test_get_or_create_rejects_sram_target(self, cache_dir, device):
        """Generic ``TensorCache.get_or_create`` must redirect SRAM targets to the dedicated wrapper."""
        cache = TensorCache(cache_dir)
        core_grid = _build_l1_core_grid(device, self.NUM_CORES)
        memory_config = _make_sram_l1_per_core_mem_config(self.K, self.N, core_grid)
        fp = self._build_fingerprint(device, assigner=_make_sram_assigner(), memory_config=memory_config)
        with pytest.raises(TypeError, match="SramCompressedTensorTarget"):
            cache.get_or_create(
                fp,
                device,
                preprocess=lambda t: t,
                raw_tensors={},
            )
