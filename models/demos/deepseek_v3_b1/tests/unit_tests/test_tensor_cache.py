# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the content-addressed TensorCache (standalone tensors).

- Fingerprint determinism and sensitivity (no device needed).
- CAS directory layout after store.
- Cache round-trip: miss -> store -> hit -> load.
- Content hash verification.
- Corrupt entry recovery.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_b1.tensor_cache.cache import TensorCache
from models.demos.deepseek_v3_b1.tensor_cache.fingerprint import canonical, compute_artifact_id
from models.demos.deepseek_v3_b1.tensor_cache.types import (
    CacheEntryState,
    Fingerprint,
    FingerprintContext,
    MeshMapperConfig,
    SourceTensorSelection,
    TensorTarget,
)


def _make_fingerprint(**overrides) -> Fingerprint:
    """Build a Fingerprint with sensible defaults; override any field via kwargs."""
    defaults = dict(
        schema_version=1,
        source=SourceTensorSelection(names=("weight_a", "weight_b")),
        hf_model_id="deepseek-ai/DeepSeek-V3",
        hf_revision="d1a891dd58e6bb0a671bfc6f3046e29e3478e924",
        transform_version=4,
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


# ---------------------------------------------------------------------------
# Fingerprint tests (no device needed)
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_determinism(self):
        """Same inputs produce the same artifact_id across calls."""
        fp1 = _make_fingerprint()
        fp2 = _make_fingerprint()
        assert compute_artifact_id(fp1) == compute_artifact_id(fp2)

    def test_canonical_is_dict(self):
        fp = _make_fingerprint()
        c = canonical(fp)
        assert isinstance(c, dict)
        assert c["schema_version"] == 1
        assert c["source"] == ["weight_a", "weight_b"]
        assert c["target"]["kind"] == "tensor"
        assert c["target"]["tile_shape"] == [32, 32]
        assert c["target"]["mesh_mapper_config"] == {"strategy": "replicate", "dim": None, "dims": None}

    def test_canonical_source_sorted(self):
        """Source names are sorted in canonical form regardless of input order."""
        fp = _make_fingerprint(source=SourceTensorSelection(names=("z_weight", "a_weight")))
        c = canonical(fp)
        assert c["source"] == ["a_weight", "z_weight"]

    def test_sensitivity_source(self):
        fp1 = _make_fingerprint(source=SourceTensorSelection(names=("a",)))
        fp2 = _make_fingerprint(source=SourceTensorSelection(names=("b",)))
        assert compute_artifact_id(fp1) != compute_artifact_id(fp2)

    def test_sensitivity_transform_version(self):
        fp1 = _make_fingerprint(transform_version=1)
        fp2 = _make_fingerprint(transform_version=2)
        assert compute_artifact_id(fp1) != compute_artifact_id(fp2)

    def test_sensitivity_mesh_shape(self):
        fp1 = _make_fingerprint(mesh_shape=(4, 2))
        fp2 = _make_fingerprint(mesh_shape=(8, 4))
        assert compute_artifact_id(fp1) != compute_artifact_id(fp2)

    def test_sensitivity_dtype(self):
        t1 = TensorTarget(name="t", dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        t2 = TensorTarget(name="t", dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        fp1 = _make_fingerprint(target=t1)
        fp2 = _make_fingerprint(target=t2)
        assert compute_artifact_id(fp1) != compute_artifact_id(fp2)

    def test_sensitivity_tile_shape(self):
        t1 = TensorTarget(name="t", tile_shape=(32, 32))
        t2 = TensorTarget(name="t", tile_shape=(16, 16))
        fp1 = _make_fingerprint(target=t1)
        fp2 = _make_fingerprint(target=t2)
        assert compute_artifact_id(fp1) != compute_artifact_id(fp2)

    def test_sensitivity_hf_revision(self):
        fp1 = _make_fingerprint(hf_revision="aaa")
        fp2 = _make_fingerprint(hf_revision="bbb")
        assert compute_artifact_id(fp1) != compute_artifact_id(fp2)

    def test_sensitivity_schema_version(self):
        fp1 = _make_fingerprint(schema_version=1)
        fp2 = _make_fingerprint(schema_version=2)
        assert compute_artifact_id(fp1) != compute_artifact_id(fp2)

    def test_sensitivity_name(self):
        t1 = TensorTarget(name="gate_bias")
        t2 = TensorTarget(name="shared_down_proj")
        fp1 = _make_fingerprint(target=t1)
        fp2 = _make_fingerprint(target=t2)
        assert compute_artifact_id(fp1) != compute_artifact_id(fp2)

    def test_sensitivity_mesh_mapper_config(self):
        t1 = TensorTarget(name="t", mesh_mapper_config=MeshMapperConfig("replicate"))
        t2 = TensorTarget(name="t", mesh_mapper_config=MeshMapperConfig("shard", dim=1))
        t3 = TensorTarget(name="t", mesh_mapper_config=MeshMapperConfig("shard_2d", dims=(0, 1)))
        t4 = TensorTarget(name="t", mesh_mapper_config=MeshMapperConfig("shard_2d", dims=(None, 1)))
        fp1 = _make_fingerprint(target=t1)
        fp2 = _make_fingerprint(target=t2)
        fp3 = _make_fingerprint(target=t3)
        fp4 = _make_fingerprint(target=t4)
        ids = {
            compute_artifact_id(fp1),
            compute_artifact_id(fp2),
            compute_artifact_id(fp3),
            compute_artifact_id(fp4),
        }
        assert len(ids) == 4

    def test_canonical_includes_mesh_mapper_config(self):
        t = TensorTarget(name="t", mesh_mapper_config=MeshMapperConfig("shard", dim=1))
        fp = _make_fingerprint(target=t)
        c = canonical(fp)
        assert c["target"]["mesh_mapper_config"] == {"strategy": "shard", "dim": 1, "dims": None}

    def test_canonical_mesh_mapper_config_shard_2d(self):
        t = TensorTarget(name="t", mesh_mapper_config=MeshMapperConfig("shard_2d", dims=(0, 1)))
        fp = _make_fingerprint(target=t)
        c = canonical(fp)
        assert c["target"]["mesh_mapper_config"] == {"strategy": "shard_2d", "dim": None, "dims": [0, 1]}

    def test_canonical_mesh_mapper_config_shard_2d_with_none_dim(self):
        t = TensorTarget(name="t", mesh_mapper_config=MeshMapperConfig("shard_2d", dims=(None, 1)))
        fp = _make_fingerprint(target=t)
        c = canonical(fp)
        assert c["target"]["mesh_mapper_config"] == {"strategy": "shard_2d", "dim": None, "dims": [None, 1]}

    def test_artifact_id_is_hex_sha256(self):
        fp = _make_fingerprint()
        aid = compute_artifact_id(fp)
        assert len(aid) == 64
        int(aid, 16)  # valid hex


# ---------------------------------------------------------------------------
# CAS storage layout tests (no device needed, uses internals)
# ---------------------------------------------------------------------------


class TestCasLayout:
    def test_store_creates_expected_files(self, tmp_path):
        """After _store, the object dir contains data.tensorbin, manifest.json, metadata.json."""
        cache = TensorCache(tmp_path)
        fp = _make_fingerprint()
        aid = compute_artifact_id(fp)

        tensor_host = ttnn.from_torch(
            torch.randn(4, 4, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tile=ttnn.Tile((32, 32)),
        )
        paths = cache._store(aid, fp, tensor_host)

        assert paths.object_dir.is_dir()
        assert paths.data_path.is_file()
        assert (paths.object_dir / "manifest.json").is_file()
        assert (paths.object_dir / "metadata.json").is_file()

    def test_store_directory_structure(self, tmp_path):
        """Object is stored under objects/{id[:2]}/{id}/."""
        cache = TensorCache(tmp_path)
        fp = _make_fingerprint()
        aid = compute_artifact_id(fp)

        tensor_host = ttnn.from_torch(
            torch.randn(4, 4, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tile=ttnn.Tile((32, 32)),
        )
        cache._store(aid, fp, tensor_host)

        expected_dir = tmp_path / "objects" / aid[:2] / aid
        assert expected_dir.is_dir()

    def test_tmp_empty_after_store(self, tmp_path):
        """The tmp/ directory should be empty after a successful store."""
        cache = TensorCache(tmp_path)
        fp = _make_fingerprint()
        aid = compute_artifact_id(fp)

        tensor_host = ttnn.from_torch(
            torch.randn(4, 4, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tile=ttnn.Tile((32, 32)),
        )
        cache._store(aid, fp, tensor_host)

        tmp_dir = tmp_path / "tmp"
        if tmp_dir.exists():
            assert list(tmp_dir.iterdir()) == []

    def test_content_hash_matches(self, tmp_path):
        """metadata.json content_hash matches SHA-256 of data.tensorbin."""
        cache = TensorCache(tmp_path)
        fp = _make_fingerprint()
        aid = compute_artifact_id(fp)

        tensor_host = ttnn.from_torch(
            torch.randn(4, 4, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tile=ttnn.Tile((32, 32)),
        )
        paths = cache._store(aid, fp, tensor_host)

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
        fp = _make_fingerprint()
        aid = compute_artifact_id(fp)

        tensor_host = ttnn.from_torch(
            torch.randn(4, 4, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tile=ttnn.Tile((32, 32)),
        )
        paths = cache._store(aid, fp, tensor_host)

        with open(paths.object_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["fingerprint"] == canonical(fp)
        assert manifest["logical_name"] == "test_tensor"


# ---------------------------------------------------------------------------
# Lookup tests
# ---------------------------------------------------------------------------


class TestLookup:
    def test_absent(self, tmp_path):
        cache = TensorCache(tmp_path)
        entry = cache._lookup("deadbeef" * 8)
        assert entry.state is CacheEntryState.ABSENT
        assert entry.paths is None

    def test_present_after_store(self, tmp_path):
        cache = TensorCache(tmp_path)
        fp = _make_fingerprint()
        aid = compute_artifact_id(fp)

        tensor_host = ttnn.from_torch(
            torch.randn(4, 4, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tile=ttnn.Tile((32, 32)),
        )
        cache._store(aid, fp, tensor_host)
        entry = cache._lookup(aid)
        assert entry.state is CacheEntryState.PRESENT
        assert entry.paths is not None

    def test_corrupt_when_data_missing(self, tmp_path):
        """If the object dir exists but data.tensorbin is missing, entry is corrupt."""
        cache = TensorCache(tmp_path)
        fp = _make_fingerprint()
        aid = compute_artifact_id(fp)

        tensor_host = ttnn.from_torch(
            torch.randn(4, 4, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tile=ttnn.Tile((32, 32)),
        )
        paths = cache._store(aid, fp, tensor_host)
        paths.data_path.unlink()

        entry = cache._lookup(aid)
        assert entry.state is CacheEntryState.CORRUPT


# ---------------------------------------------------------------------------
# Full round-trip tests (require device)
# ---------------------------------------------------------------------------


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

        fp = _make_fingerprint(
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
            fp,
            device,
            preprocess=preprocess,
            raw_tensors=lambda: {"raw": raw_data},
        )
        assert preprocess_call_count[0] == 1
        assert isinstance(result1, ttnn.Tensor)

        result2 = cache.get_or_create(
            fp,
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

        fp = _make_fingerprint(
            source=SourceTensorSelection(names=("w",)),
            target=TensorTarget(name="t", dtype=ttnn.bfloat16, tile_shape=(32, 32)),
        )

        result1 = cache.get_or_create(
            fp, device, preprocess=lambda d: {"t": d["w"]}, raw_tensors=lambda: {"w": raw_data}
        )

        raw_called = [False]

        def raw_tensors_spy():
            raw_called[0] = True
            return {"w": raw_data}

        result2 = cache.get_or_create(fp, device, preprocess=lambda d: {"t": d["w"]}, raw_tensors=raw_tensors_spy)
        assert not raw_called[0]

        ttnn.deallocate(result1, force=True)
        ttnn.deallocate(result2, force=True)

    def test_corrupt_recovery(self, cache_dir, device):
        """Deleting data.tensorbin should cause a transparent rebuild on next access."""
        cache = TensorCache(cache_dir)
        raw_data = torch.randn(16, 16, dtype=torch.bfloat16)

        fp = _make_fingerprint(
            source=SourceTensorSelection(names=("w",)),
            target=TensorTarget(name="t", dtype=ttnn.bfloat16, tile_shape=(32, 32)),
        )

        result1 = cache.get_or_create(
            fp, device, preprocess=lambda d: {"t": d["w"]}, raw_tensors=lambda: {"w": raw_data}
        )
        ttnn.deallocate(result1, force=True)

        aid = compute_artifact_id(fp)
        data_path = cache._cas_paths(aid).data_path
        assert data_path.is_file()
        data_path.unlink()

        preprocess_calls = [0]

        def counting_preprocess(d):
            preprocess_calls[0] += 1
            return {"t": d["w"]}

        result2 = cache.get_or_create(fp, device, preprocess=counting_preprocess, raw_tensors=lambda: {"w": raw_data})
        assert preprocess_calls[0] == 1
        assert isinstance(result2, ttnn.Tensor)
        ttnn.deallocate(result2, force=True)

    def test_different_fingerprints_different_artifacts(self, cache_dir, device):
        """Two different fingerprints produce two separate cache entries."""
        cache = TensorCache(cache_dir)

        fp1 = _make_fingerprint(
            source=SourceTensorSelection(names=("a",)),
            target=TensorTarget(name="t", dtype=ttnn.bfloat16, tile_shape=(32, 32)),
        )
        fp2 = _make_fingerprint(
            source=SourceTensorSelection(names=("b",)),
            target=TensorTarget(name="t", dtype=ttnn.bfloat16, tile_shape=(32, 32)),
        )

        data = torch.randn(16, 16, dtype=torch.bfloat16)
        identity = lambda d: {"t": d["w"]}
        raw = lambda: {"w": data}

        r1 = cache.get_or_create(fp1, device, preprocess=identity, raw_tensors=raw)
        r2 = cache.get_or_create(fp2, device, preprocess=identity, raw_tensors=raw)

        aid1 = compute_artifact_id(fp1)
        aid2 = compute_artifact_id(fp2)
        assert aid1 != aid2
        assert cache._lookup(aid1).state is CacheEntryState.PRESENT
        assert cache._lookup(aid2).state is CacheEntryState.PRESENT

        ttnn.deallocate(r1, force=True)
        ttnn.deallocate(r2, force=True)

    def test_dict_raw_tensors(self, cache_dir, device):
        """raw_tensors can be a plain dict instead of a callable."""
        cache = TensorCache(cache_dir)

        fp = _make_fingerprint(
            source=SourceTensorSelection(names=("w",)),
            target=TensorTarget(name="t", dtype=ttnn.bfloat16, tile_shape=(32, 32)),
        )
        data = torch.randn(16, 16, dtype=torch.bfloat16)

        result = cache.get_or_create(fp, device, preprocess=lambda d: {"t": d["w"]}, raw_tensors={"w": data})
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
        fp = _make_fingerprint(
            source=SourceTensorSelection(names=("model.embed_tokens.weight",)),
            target=target,
        )
        data = torch.randn(64, 32, dtype=torch.bfloat16)

        result = cache.get_or_create(
            fp, device, preprocess=lambda d: {"emb": d["w"].contiguous()}, raw_tensors={"w": data}
        )
        assert isinstance(result, ttnn.Tensor)

        result2 = cache.get_or_create(
            fp, device, preprocess=lambda d: {"emb": d["w"].contiguous()}, raw_tensors={"w": data}
        )
        assert isinstance(result2, ttnn.Tensor)

        ttnn.deallocate(result, force=True)
        ttnn.deallocate(result2, force=True)


# ---------------------------------------------------------------------------
# FingerprintContext tests (no device needed)
# ---------------------------------------------------------------------------


class TestFingerprintContext:
    def test_factory_produces_valid_fingerprint(self):
        ctx = FingerprintContext(
            schema_version=1,
            hf_model_id="deepseek-ai/DeepSeek-V3",
            hf_revision="abc123",
            transform_version=1,
            mesh_shape=(4, 2),
        )
        target = TensorTarget(name="embedding", dtype=ttnn.bfloat16)
        source = SourceTensorSelection(names=("model.embed_tokens.weight",))
        fp = ctx.fingerprint(source=source, target=target)

        assert isinstance(fp, Fingerprint)
        assert fp.schema_version == 1
        assert fp.hf_model_id == "deepseek-ai/DeepSeek-V3"
        assert fp.hf_revision == "abc123"
        assert fp.transform_version == 1
        assert fp.mesh_shape == (4, 2)
        assert fp.target is target
        assert fp.source is source

    def test_different_contexts_different_ids(self):
        ctx1 = FingerprintContext(
            schema_version=1, hf_model_id="m", hf_revision="a", transform_version=1, mesh_shape=(4, 2)
        )
        ctx2 = FingerprintContext(
            schema_version=1, hf_model_id="m", hf_revision="b", transform_version=1, mesh_shape=(4, 2)
        )
        target = TensorTarget(name="t")
        source = SourceTensorSelection(names=("w",))
        fp1 = ctx1.fingerprint(source=source, target=target)
        fp2 = ctx2.fingerprint(source=source, target=target)
        assert compute_artifact_id(fp1) != compute_artifact_id(fp2)

    def test_same_context_same_ids(self):
        ctx = FingerprintContext(
            schema_version=1, hf_model_id="m", hf_revision="a", transform_version=1, mesh_shape=(4, 2)
        )
        target = TensorTarget(name="t")
        source = SourceTensorSelection(names=("w",))
        fp1 = ctx.fingerprint(source=source, target=target)
        fp2 = ctx.fingerprint(source=source, target=target)
        assert compute_artifact_id(fp1) == compute_artifact_id(fp2)
