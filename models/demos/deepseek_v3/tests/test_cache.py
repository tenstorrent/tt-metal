# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import fcntl
import json
import multiprocessing as mp

import pytest
import torch

import ttnn
from models.demos.deepseek_v3.utils.cache import (
    CacheKey,
    CacheManifest,
    InMemoryCacheStorage,
    OnDiskCacheStorage,
    TensorCache,
    _safe_name_from_manifest,
    compute_fingerprint,
    compute_func_fingerprint,
    create_manifest,
)


def _identity(x):
    return x


def _make_cache_key(name: str = "weight1", hf_config: dict | None = None) -> CacheKey:
    hf_config = {"factor": 1} if hf_config is None else hf_config
    manifest = create_manifest(
        name=name,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        hf_config=hf_config,
        preprocessor=_identity,
        postprocessor=_identity,
    )
    return CacheKey(fingerprint=manifest.get_fingerprint(), manifest=manifest)


def _hold_lock(path: str, ready: mp.Event, release: mp.Event) -> None:
    with open(path, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        ready.set()
        release.wait(timeout=5)


@pytest.fixture
def sample_state_dict():
    """Create a sample state dict for testing."""
    return {
        "weight1": torch.zeros((128, 128), dtype=torch.bfloat16),
        "weight2": torch.ones((256, 256), dtype=torch.bfloat16),
        "weight3": torch.randn((64, 64), dtype=torch.bfloat16),
    }


@pytest.fixture
def sample_hf_config():
    """Create a sample HF config for testing."""
    return {"factor": 2, "hidden_size": 128}


@pytest.fixture
def sample_state_dict_same_shape():
    """State dict with multiple tensors of the same shape for multi-source stack tests."""
    return {
        "weight1": torch.zeros((128, 128), dtype=torch.bfloat16),
        "weight2": torch.ones((128, 128), dtype=torch.bfloat16),
        "weight3": torch.randn((128, 128), dtype=torch.bfloat16),
    }


@pytest.fixture
def cache_storage():
    """Create a fresh cache storage for each test."""
    return InMemoryCacheStorage()


@pytest.fixture
def tensor_cache(sample_state_dict, sample_hf_config, cache_storage):
    """Create a TensorCache instance for testing."""
    return TensorCache(sample_state_dict, sample_hf_config, cache_storage)


def setup_cache_tracker(tensor_cache, monkeypatch):
    """Helper function to set up cache call tracking with monkeypatch."""
    call_tracker = {"has_calls": [], "set_calls": [], "get_calls": []}
    original_has = tensor_cache.storage.has
    original_set = tensor_cache.storage.set
    original_get = tensor_cache.storage.get

    def tracked_has(key: CacheKey) -> bool:
        result = original_has(key)
        call_tracker["has_calls"].append(("has", key, result))
        return result

    def tracked_set(key: CacheKey, tensor: ttnn.Tensor):
        call_tracker["set_calls"].append(("set", key))
        return original_set(key, tensor)

    def tracked_get(key: CacheKey) -> ttnn.Tensor:
        call_tracker["get_calls"].append(("get", key))
        return original_get(key)

    monkeypatch.setattr(tensor_cache.storage, "has", tracked_has)
    monkeypatch.setattr(tensor_cache.storage, "set", tracked_set)
    monkeypatch.setattr(tensor_cache.storage, "get", tracked_get)

    return call_tracker


def clear_call_tracker(call_tracker):
    """Clear all call tracker lists."""
    call_tracker["has_calls"].clear()
    call_tracker["set_calls"].clear()
    call_tracker["get_calls"].clear()


def assert_cache_miss(call_tracker, clear_after=True):
    """Assert that cache operations indicate a cache miss.

    Args:
        call_tracker: The call tracker dictionary
        clear_after: If True, clear the tracker after assertion (default: True)
    """
    assert len(call_tracker["has_calls"]) == 1, f"Expected 1 has() call, got {len(call_tracker['has_calls'])}"
    assert call_tracker["has_calls"][0][1] is not None, "CacheKey should be passed to has()"
    assert isinstance(call_tracker["has_calls"][0][1], CacheKey)
    assert call_tracker["has_calls"][0][1].fingerprint
    assert call_tracker["has_calls"][0][2] is False, "has() should return False for cache miss"
    assert len(call_tracker["set_calls"]) == 1, f"Expected 1 set() call, got {len(call_tracker['set_calls'])}"
    assert len(call_tracker["get_calls"]) == 0, f"Expected 0 get() calls, got {len(call_tracker['get_calls'])}"
    if clear_after:
        clear_call_tracker(call_tracker)


def assert_cache_hit(call_tracker, clear_after=True):
    """Assert that cache operations indicate a cache hit.

    Args:
        call_tracker: The call tracker dictionary
        clear_after: If True, clear the tracker after assertion (default: True)
    """
    assert len(call_tracker["has_calls"]) == 1, f"Expected 1 has() call, got {len(call_tracker['has_calls'])}"
    assert call_tracker["has_calls"][0][2] is True, "has() should return True for cache hit"
    assert isinstance(call_tracker["has_calls"][0][1], CacheKey)
    assert len(call_tracker["get_calls"]) == 1, f"Expected 1 get() call, got {len(call_tracker['get_calls'])}"
    assert len(call_tracker["set_calls"]) == 0, f"Expected 0 set() calls, got {len(call_tracker['set_calls'])}"
    if clear_after:
        clear_call_tracker(call_tracker)


def test_compute_func_fingerprint_named_function():
    """Verify that compute_func_fingerprint returns the same fingerprint for the same function
    when called multiple times, and that the result is a 32-character MD5 hex string."""

    def test_func(x):
        return x * 2

    fingerprint1 = compute_func_fingerprint(test_func)
    fingerprint2 = compute_func_fingerprint(test_func)

    assert fingerprint1 == fingerprint2
    assert isinstance(fingerprint1, str)
    assert len(fingerprint1) == 32  # MD5 hexdigest length


def test_compute_func_fingerprint_different_functions():
    """Verify that different function objects produce different fingerprints even when they
    have identical behavior (fingerprinting is based on object identity/source, not semantics)."""

    def func1(x):
        return x * 3

    def func2(x):
        return x * 3

    fingerprint1 = compute_func_fingerprint(func1)
    fingerprint2 = compute_func_fingerprint(func2)

    assert fingerprint1 != fingerprint2


def test_create_manifest_basic(sample_hf_config):
    """Verify create_manifest returns a CacheManifest whose to_dict() contains the expected
    keys and values for a single tensor name (name, dtype, layout, memory_config, hf_config,
    preprocessor and postprocessor fingerprints)."""

    def preprocessor(x):
        return x

    def postprocessor(x):
        return x

    manifest = create_manifest(
        name="test_weight",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        hf_config=sample_hf_config,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )

    assert isinstance(manifest, CacheManifest)
    d = manifest.to_dict()
    assert d["manifest_version"] == 1
    assert d["name"] == "test_weight"
    assert d["dtype"] == "BFLOAT16"  # dtype.__name__ returns uppercase
    assert d["layout"] == "TILE"  # layout.__name__ returns "TILE" for TILE_LAYOUT
    assert "memory_config" in d
    assert d["mesh_shape"] is None
    assert "hf_config" in d
    assert "preprocessor" in d
    assert "postprocessor" in d


def test_create_manifest_multi_name(sample_hf_config):
    """Verify that when name is a list of strings, create_manifest produces a CacheManifest
    whose to_dict() uses the 'names' key with a sorted JSON array of names."""

    def preprocessor(x):
        return x

    def postprocessor(x):
        return x

    manifest = create_manifest(
        name=["weight_a", "weight_b"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        hf_config=sample_hf_config,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
    assert isinstance(manifest, CacheManifest)
    d = manifest.to_dict()
    assert d["manifest_version"] == 1
    assert "names" in d
    assert d["names"] == json.dumps(sorted(["weight_a", "weight_b"]))
    assert "dtype" in d
    assert "layout" in d
    assert "memory_config" in d
    assert d["mesh_shape"] is None


def test_compute_fingerprint_stable():
    """Verify that compute_fingerprint returns the same 32-character hex string when given
    the same manifest dict twice."""
    manifest = {
        "name": "test",
        "dtype": "bfloat16",
        "layout": "TILE_LAYOUT",
    }

    fingerprint1 = compute_fingerprint(manifest)
    fingerprint2 = compute_fingerprint(manifest)

    assert fingerprint1 == fingerprint2
    assert isinstance(fingerprint1, str)
    assert len(fingerprint1) == 32


def test_compute_fingerprint_different_manifests():
    """Verify that compute_fingerprint returns different hex strings for manifest dicts
    that differ in content (e.g. different tensor names)."""
    manifest1 = {"name": "test1", "dtype": "bfloat16"}
    manifest2 = {"name": "test2", "dtype": "bfloat16"}

    fingerprint1 = compute_fingerprint(manifest1)
    fingerprint2 = compute_fingerprint(manifest2)

    assert fingerprint1 != fingerprint2


def test_cache_storage_set_get(cache_storage):
    """Verify that InMemoryCacheStorage.set stores a tensor under a key and get retrieves
    the same object; has() returns True for that key."""
    tensor = ttnn.from_torch(torch.zeros((32, 32), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cache_key = _make_cache_key(name="key1")

    cache_storage.set(cache_key, tensor)
    retrieved = cache_storage.get(cache_key)

    assert retrieved is tensor
    assert cache_storage.has(cache_key)


def test_cache_storage_has(cache_storage):
    """Verify that has() returns False for keys that were never set, and True for a key
    after set(); a different key remains False."""
    missing_key = _make_cache_key(name="nonexistent")
    assert not cache_storage.has(missing_key)

    tensor = ttnn.from_torch(torch.zeros((32, 32), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    key1 = _make_cache_key(name="key1")
    cache_storage.set(key1, tensor)

    assert cache_storage.has(key1)
    assert not cache_storage.has(_make_cache_key(name="key2"))


def test_cache_storage_get_nonexistent(cache_storage):
    """Verify that get() raises KeyError when the key has not been set."""
    missing_key = _make_cache_key(name="nonexistent")
    with pytest.raises(KeyError):
        cache_storage.get(missing_key)


def test_on_disk_cache_storage_set_get(tmp_path, device):
    tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    cache_key = _make_cache_key(name="weight1")
    storage = OnDiskCacheStorage(tmp_path, device)

    storage.set(cache_key, tensor)
    assert storage.has(cache_key)

    safe_name = _safe_name_from_manifest(cache_key.manifest)
    tensor_path = tmp_path / f"{safe_name}__{cache_key.fingerprint}.tensorbin"
    manifest_path = tmp_path / f"{safe_name}__{cache_key.fingerprint}.manifest.json"
    assert tensor_path.is_file()
    assert manifest_path.is_file()

    loaded = storage.get(cache_key)
    assert torch.allclose(ttnn.to_torch(loaded), ttnn.to_torch(tensor))

    storage_2 = OnDiskCacheStorage(tmp_path, device)
    assert storage_2.has(cache_key)
    loaded_2 = storage_2.get(cache_key)
    assert torch.allclose(ttnn.to_torch(loaded_2), ttnn.to_torch(tensor))


def test_on_disk_cache_storage_writes_manifest_payload(tmp_path, device):
    tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    cache_key = _make_cache_key(name="weight1")
    storage = OnDiskCacheStorage(tmp_path, device)

    storage.set(cache_key, tensor)

    safe_name = _safe_name_from_manifest(cache_key.manifest)
    manifest_path = tmp_path / f"{safe_name}__{cache_key.fingerprint}.manifest.json"
    payload = json.loads(manifest_path.read_text())

    assert payload["fingerprint"] == cache_key.fingerprint
    assert payload.get("name") == cache_key.manifest.name
    assert payload.get("created_at") == payload.get("last_modified")


def test_on_disk_cache_storage_lock_timeout_skips_write(tmp_path, device):
    tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    cache_key = _make_cache_key(name="weight1")
    storage = OnDiskCacheStorage(tmp_path, device, lock_timeout_s=0.0, lock_poll_s=0.01)

    lock_path = storage._lock_path(cache_key)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    proc = ctx.Process(target=_hold_lock, args=(str(lock_path), ready, release))
    proc.start()
    try:
        assert ready.wait(timeout=5)
        storage.set(cache_key, tensor)
    finally:
        release.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()

    assert proc.exitcode == 0

    safe_name = _safe_name_from_manifest(cache_key.manifest)
    tensor_path = tmp_path / f"{safe_name}__{cache_key.fingerprint}.tensorbin"
    manifest_path = tmp_path / f"{safe_name}__{cache_key.fingerprint}.manifest.json"
    assert not tensor_path.exists()
    assert not manifest_path.exists()


def test_on_disk_cache_storage_manifest_includes_mesh_mapper(
    tmp_path, mesh_device, sample_state_dict, sample_hf_config
):
    storage = OnDiskCacheStorage(tmp_path, mesh_device)
    cache = TensorCache(sample_state_dict, sample_hf_config, storage)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, -1))

    cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preprocessor=_identity,
        postprocessor=_identity,
        device=mesh_device,
        mesh_mapper=mapper,
    )

    manifest = create_manifest(
        name=["weight1"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        hf_config=sample_hf_config,
        preprocessor=_identity,
        postprocessor=_identity,
        mesh_mapper=mapper,
        mesh_shape=tuple(mesh_device.shape),
    )
    cache_key = CacheKey(fingerprint=manifest.get_fingerprint(), manifest=manifest)
    safe_name = _safe_name_from_manifest(manifest)
    manifest_path = tmp_path / f"{safe_name}__{cache_key.fingerprint}.manifest.json"
    payload = json.loads(manifest_path.read_text())

    expected_mesh_mapper = manifest.to_dict()["mesh_mapper"]
    if expected_mesh_mapper["mesh_shape_override"] is not None:
        expected_mesh_mapper["mesh_shape_override"] = list(expected_mesh_mapper["mesh_shape_override"])
    assert payload["mesh_mapper"] == expected_mesh_mapper
    assert payload["mesh_shape"] == list(mesh_device.shape)


def test_on_disk_cache_storage_missing_key(tmp_path, device):
    storage = OnDiskCacheStorage(tmp_path, device)
    missing_key = _make_cache_key(name="missing")
    assert not storage.has(missing_key)
    with pytest.raises(KeyError):
        storage.get(missing_key)


def test_tensor_cache_on_disk_cache_hit_uses_disk(tmp_path, device, sample_state_dict, sample_hf_config):
    calls = {"converter": 0}

    def counting_converter(tensor, dtype, layout, memory_config=None, device=None, mesh_mapper=None):
        calls["converter"] += 1
        return ttnn.from_torch(
            tensor,
            dtype=dtype,
            layout=layout,
            device=device,
            memory_config=memory_config,
        )

    storage = OnDiskCacheStorage(tmp_path, device)
    cache = TensorCache(sample_state_dict, sample_hf_config, storage, converter=counting_converter)

    tensor1 = cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
    )
    tensor2 = cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
    )

    assert calls["converter"] == 1
    assert torch.allclose(ttnn.to_torch(tensor1), ttnn.to_torch(tensor2))


def test_tensor_cache_on_disk_mesh_mapper_entries(
    tmp_path, mesh_device, sample_state_dict, sample_hf_config, monkeypatch
):
    storage = OnDiskCacheStorage(tmp_path, mesh_device)
    cache = TensorCache(sample_state_dict, sample_hf_config, storage)
    call_tracker = setup_cache_tracker(cache, monkeypatch)

    replicate_1 = ttnn.ReplicateTensorToMesh(mesh_device)
    tensor_1 = cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preprocessor=_identity,
        postprocessor=_identity,
        device=mesh_device,
        mesh_mapper=replicate_1,
    )
    assert tensor_1 is not None
    assert_cache_miss(call_tracker)

    replicate_2 = ttnn.ReplicateTensorToMesh(mesh_device)
    tensor_2 = cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preprocessor=_identity,
        postprocessor=_identity,
        device=mesh_device,
        mesh_mapper=replicate_2,
    )
    assert tensor_2 is not None
    assert_cache_hit(call_tracker)

    shard_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, -1))
    tensor_shard = cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preprocessor=_identity,
        postprocessor=_identity,
        device=mesh_device,
        mesh_mapper=shard_mapper,
    )
    assert tensor_shard is not None
    assert_cache_miss(call_tracker)


def test_tensor_cache_cache_miss(tensor_cache, monkeypatch, device):
    """Verify that the first get_tensor call for a given (name, dtype, layout) triggers a
    cache miss: storage.has returns False, set is called once, get is not called; returned
    tensor has the requested dtype and layout."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    # First call should be a cache miss
    tensor = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Verify cache miss behavior
    assert tensor is not None
    assert tensor.dtype == ttnn.bfloat16
    assert tensor.layout == ttnn.TILE_LAYOUT
    assert_cache_miss(call_tracker)


def test_tensor_cache_cache_hit(tensor_cache, monkeypatch, device):
    """Verify that a second get_tensor call with the same (name, dtype, layout) is a cache
    hit: storage.has returns True, get is called once, set is not called, and the same
    tensor object is returned."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    # First call - cache miss
    tensor1 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    assert_cache_miss(call_tracker)

    # Second call - cache hit (should return same tensor)
    tensor2 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Should be the same tensor object (cached)
    assert tensor1 is tensor2
    assert_cache_hit(call_tracker)


def test_tensor_cache_different_dtypes(tensor_cache, monkeypatch, device):
    """Verify that requesting the same tensor name with different dtypes (e.g. bfloat16
    vs float32) results in two cache misses and two distinct tensor objects."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    tensor1 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    tensor2 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Second call with different dtype should also be cache miss
    assert tensor1.dtype != tensor2.dtype
    assert tensor1 is not tensor2
    assert_cache_miss(call_tracker)


def test_tensor_cache_different_layouts(tensor_cache, monkeypatch, device):
    """Verify that requesting the same tensor name with different layouts (e.g. TILE vs
    ROW_MAJOR) results in two cache misses and two distinct tensor objects."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    tensor1 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    tensor2 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # Second call with different layout should also be cache miss
    assert tensor1.layout != tensor2.layout
    assert tensor1 is not tensor2
    assert_cache_miss(call_tracker)


def test_tensor_cache_different_preprocessors(tensor_cache, monkeypatch, device):
    """Verify that the same tensor name with different preprocessor callables (e.g. scale
    by 2 vs scale by 3) results in two cache misses and two distinct tensor objects."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    def preprocessor1(x):
        return x * 2

    def preprocessor2(x):
        return x * 3

    tensor1 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=preprocessor1,
        device=device,
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    tensor2 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=preprocessor2,
        device=device,
    )

    # Should be different cache entries (different preprocessors)
    assert tensor1 is not tensor2
    assert_cache_miss(call_tracker)


def test_tensor_cache_different_postprocessors(tensor_cache, monkeypatch, device):
    """Verify that different postprocessor function objects (even with identical behavior)
    produce different cache entries because fingerprinting uses function identity."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    def postprocessor1(x):
        return x

    def postprocessor2(x):
        return x  # Same function but different object

    tensor1 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        postprocessor=postprocessor1,
        device=device,
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    tensor2 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        postprocessor=postprocessor2,
        device=device,
    )

    # Different function objects should create different cache entries
    assert tensor1 is not tensor2
    assert_cache_miss(call_tracker)


def test_tensor_cache_missing_tensor(tensor_cache, device):
    """Verify that get_tensor raises KeyError with a message including the missing name
    when the requested tensor key is not in the state dict."""
    with pytest.raises(KeyError, match="Tensor 'nonexistent' not found"):
        tensor_cache.get_tensor(
            name="nonexistent",
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )


def test_tensor_cache_multiple_tensors(tensor_cache, monkeypatch, device):
    """Verify that requesting different tensor names (weight1, weight2) each causes a cache
    miss and returns distinct tensors; repeating the same requests then yields cache hits
    and the same tensor objects."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    tensor1 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    tensor2 = tensor_cache.get_tensor(
        name="weight2",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Second call with different name should also be cache miss
    assert tensor1 is not tensor2
    assert_cache_miss(call_tracker)

    # Both should be cached (cache hits)
    tensor1_cached = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # First cache hit
    assert tensor1 is tensor1_cached
    assert_cache_hit(call_tracker)

    tensor2_cached = tensor_cache.get_tensor(
        name="weight2",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Second cache hit
    assert tensor2 is tensor2_cached
    assert_cache_hit(call_tracker)


def test_tensor_cache_validation_assertions(tensor_cache, monkeypatch, device):
    """Verify that on a cache hit, the returned tensor is checked for matching dtype and
    layout; the test creates an entry then retrieves it and asserts cache hit and
    correct dtype/layout."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    # Create a cache entry
    tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    # Retrieve from cache - should validate dtype and layout match
    cached_tensor = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Second call should be cache hit
    assert cached_tensor.dtype == ttnn.bfloat16
    assert cached_tensor.layout == ttnn.TILE_LAYOUT
    assert_cache_hit(call_tracker)


def test_tensor_cache_preprocessor_application(tensor_cache, device):
    """Verify that a custom preprocessor (e.g. scaling the tensor) is invoked when
    creating the cached tensor; the returned ttnn tensor is non-None and has the
    requested dtype."""

    def preprocessor(x):
        return x * 2.0

    tensor = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=preprocessor,
        device=device,
    )

    # Verify tensor was created (preprocessor should have been applied)
    assert tensor is not None
    assert tensor.dtype == ttnn.bfloat16


def test_tensor_cache_postprocessor_application(tensor_cache, device):
    """Verify that a custom postprocessor is invoked after conversion; the test uses an
    identity postprocessor to ensure the pipeline runs and returns a valid tensor."""

    def postprocessor(x):
        return x  # Identity, but verifies it's called

    tensor = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        postprocessor=postprocessor,
        device=device,
    )

    # Verify tensor was created (postprocessor should have been applied)
    assert tensor is not None
    assert tensor.dtype == ttnn.bfloat16


def test_tensor_cache_default_preprocessor_postprocessor(tensor_cache, monkeypatch, device):
    """Verify that omitting preprocessor/postprocessor uses default lambdas and that
    explicitly passing a different identity function creates a different cache entry
    (different fingerprint), so the two requests return different tensor objects."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    # Should work with defaults (lambda x: x)
    tensor1 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    # Explicit identity should create different cache entry (different lambda objects)
    def identity(x):
        return x

    tensor2 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=identity,
        postprocessor=identity,
        device=device,
    )

    # Different lambda objects = different fingerprints = different cache entries
    assert tensor1 is not tensor2
    assert_cache_miss(call_tracker)


def test_tensor_cache_hf_config_changes_cache(tensor_cache, sample_state_dict, cache_storage, device):
    """Verify that TensorCaches with different hf_config values produce different cache
    entries for the same tensor request, and that the same hf_config shares the same
    storage (same tensor object when config matches)."""
    # Create cache with config1
    cache1 = TensorCache(sample_state_dict, {"factor": 2}, cache_storage)
    tensor1 = cache1.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Create cache with config2 (different config)
    cache2 = TensorCache(sample_state_dict, {"factor": 3}, cache_storage)
    tensor2 = cache2.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Create cache with config3 (different config)
    cache3 = TensorCache(sample_state_dict, {"factor": 2}, cache_storage)
    tensor3 = cache3.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    assert tensor1 is not tensor2
    assert tensor2 is not tensor3
    assert tensor1 is tensor3


def test_tensor_cache_multi_source_cache_miss_hit(
    sample_state_dict_same_shape, sample_hf_config, cache_storage, monkeypatch, device
):
    """Verify that get_tensor with a list of names and a stack preprocessor: first call
    is a cache miss and produces a tensor; second call with same names and preprocessor
    is a cache hit and returns the same tensor object."""
    cache = TensorCache(sample_state_dict_same_shape, sample_hf_config, cache_storage)
    call_tracker = setup_cache_tracker(cache, monkeypatch)

    def stack_preprocessor(*tensors):
        return torch.stack(tensors, dim=0)

    tensor1 = cache.get_tensor(
        name=["weight1", "weight2"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=stack_preprocessor,
        device=device,
    )
    assert tensor1 is not None
    assert tensor1.dtype == ttnn.bfloat16
    assert tensor1.layout == ttnn.TILE_LAYOUT
    assert_cache_miss(call_tracker)

    tensor2 = cache.get_tensor(
        name=["weight1", "weight2"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=stack_preprocessor,
        device=device,
    )
    assert tensor1 is tensor2
    assert_cache_hit(call_tracker)


def test_tensor_cache_multi_source_different_names_different_entries(
    sample_state_dict_same_shape, sample_hf_config, cache_storage, monkeypatch, device
):
    """Verify that get_tensor with different name lists (e.g. [weight1, weight2] vs
    [weight2, weight3]) and the same preprocessor results in two cache misses and two
    distinct tensor objects."""
    cache = TensorCache(sample_state_dict_same_shape, sample_hf_config, cache_storage)
    call_tracker = setup_cache_tracker(cache, monkeypatch)

    def stack_preprocessor(*tensors):
        return torch.stack(tensors, dim=0)

    tensor_1_2 = cache.get_tensor(
        name=["weight1", "weight2"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=stack_preprocessor,
        device=device,
    )
    assert_cache_miss(call_tracker)

    tensor_2_3 = cache.get_tensor(
        name=["weight2", "weight3"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=stack_preprocessor,
        device=device,
    )
    assert tensor_1_2 is not tensor_2_3
    assert_cache_miss(call_tracker)


def test_tensor_cache_multi_source_different_preprocessor_different_entries(
    sample_state_dict_same_shape, sample_hf_config, cache_storage, monkeypatch, device
):
    """Verify that the same list of names with different multi-tensor preprocessors (e.g.
    stack vs sum) produces two cache misses and two distinct tensor objects."""
    cache = TensorCache(sample_state_dict_same_shape, sample_hf_config, cache_storage)
    call_tracker = setup_cache_tracker(cache, monkeypatch)

    def stack_preprocessor(*tensors):
        return torch.stack(tensors, dim=0)

    def sum_preprocessor(*tensors):
        return sum(tensors)

    tensor_stack = cache.get_tensor(
        name=["weight1", "weight2"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=stack_preprocessor,
        device=device,
    )
    assert_cache_miss(call_tracker)

    tensor_sum = cache.get_tensor(
        name=["weight1", "weight2"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=sum_preprocessor,
        device=device,
    )
    assert tensor_stack is not tensor_sum
    assert_cache_miss(call_tracker)


def test_tensor_cache_multi_source_missing_tensor(tensor_cache, device):
    """Verify that get_tensor with a list of names raises KeyError when one of the names
    is not in the state dict (e.g. ['weight1', 'nonexistent'])."""

    def stack_preprocessor(*tensors):
        return torch.stack(tensors, dim=0)

    with pytest.raises(KeyError, match="Tensor 'nonexistent' not found"):
        tensor_cache.get_tensor(
            name=["weight1", "nonexistent"],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            preprocessor=stack_preprocessor,
            device=device,
        )


def test_tensor_cache_memory_config_validation(tensor_cache, monkeypatch, mesh_device):
    """Verify that when device and memory_config are provided, the cache key includes
    memory_config: same request yields a cache hit and matching memory_config(); a
    request for the same name with a different memory_config is a cache miss and returns
    a tensor with the new memory config."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    # Create a cache entry
    cached_tensor = tensor_cache.get_tensor(
        name="weight1",
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=mesh_device,
    )
    # First call should be cache miss
    assert_cache_miss(call_tracker)
    assert cached_tensor.memory_config() == ttnn.L1_MEMORY_CONFIG
    assert cached_tensor.storage_type() == ttnn.StorageType.DEVICE

    # Retrieve from cache - should validate memory config match
    cached_tensor = tensor_cache.get_tensor(
        name="weight1",
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=mesh_device,
    )

    # Second call should be cache hit
    assert_cache_hit(call_tracker)
    assert cached_tensor.memory_config() == ttnn.L1_MEMORY_CONFIG
    assert cached_tensor.storage_type() == ttnn.StorageType.DEVICE

    # Load the same tensor again with a different memory config should be a cache miss
    cached_tensor = tensor_cache.get_tensor(
        name="weight1",
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=mesh_device,
    )
    assert_cache_miss(call_tracker)
    assert cached_tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert cached_tensor.storage_type() == ttnn.StorageType.DEVICE


def test_tensor_cache_mesh_mapper_different_entries(tensor_cache, monkeypatch, mesh_device):
    """Verify that different mesh_mappers create different cache entries: same (device, memory_config)
    with mapper A yields cache miss then hit; with mapper B yields another cache miss and a different
    tensor."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    mapper_a = ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, -1))
    tensor_a1 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=mesh_device,
        mesh_mapper=mapper_a,
    )
    assert_cache_miss(call_tracker)
    assert tensor_a1 is not None

    mapper_a2 = ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, -1))
    tensor_a2 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=mesh_device,
        mesh_mapper=mapper_a2,
    )
    assert tensor_a1 is tensor_a2
    assert_cache_hit(call_tracker)

    mapper_b = ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(-1, 0))
    tensor_b = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=mesh_device,
        mesh_mapper=mapper_b,
    )
    assert tensor_b is not tensor_a1
    assert_cache_miss(call_tracker)


def test_tensor_cache_replicate_mesh_mapper_same_entry(tensor_cache, monkeypatch, mesh_device):
    """Verify that replicate mesh mappers share the same cache entry: two ReplicateTensorToMesh(mesh_device)
    instances produce the same cache key, so the second request is a cache hit."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    mapper_1 = ttnn.ReplicateTensorToMesh(mesh_device)
    tensor_1 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=mesh_device,
        mesh_mapper=mapper_1,
    )
    assert_cache_miss(call_tracker)
    assert tensor_1 is not None

    mapper_2 = ttnn.ReplicateTensorToMesh(mesh_device)
    tensor_2 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=mesh_device,
        mesh_mapper=mapper_2,
    )
    assert tensor_1 is tensor_2
    assert_cache_hit(call_tracker)

    # Different mapper (shard) => different cache entry and different tensor
    shard_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, -1))
    tensor_shard = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=mesh_device,
        mesh_mapper=shard_mapper,
    )
    assert tensor_shard is not tensor_1
    assert_cache_miss(call_tracker)
