# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

import ttnn
from models.demos.deepseek_v3.utils.cache import (
    CacheManifest,
    CacheStorage,
    InMemoryCacheStorage,
    OnDiskCacheStorage,
    TensorCache,
    compute_fingerprint,
    compute_func_fingerprint,
    create_manifest,
    identity_postprocessor,
    identity_preprocessor,
)


@pytest.fixture(autouse=True)
def _default_device(device):
    return device


def get_tensor(cache: TensorCache, *args, **kwargs):
    if "device" not in kwargs:
        device = ttnn.GetDefaultDevice()
        if device is None:
            raise RuntimeError("Default device is not initialized for get_tensor()")
        kwargs["device"] = device
    return cache.get_tensor(*args, **kwargs)


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

    def tracked_has(key: str) -> bool:
        result = original_has(key)
        call_tracker["has_calls"].append(("has", key, result))
        return result

    def tracked_set(key: str, tensor: ttnn.Tensor, *, manifest: CacheManifest):
        call_tracker["set_calls"].append(("set", key))
        return original_set(key, tensor, manifest=manifest)

    def tracked_get(key: str) -> ttnn.Tensor:
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
    assert call_tracker["has_calls"][0][1] is not None, "Fingerprint should be passed to has()"
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
    assert d["name"] == "test_weight"
    assert d["dtype"] == "BFLOAT16"  # dtype.__name__ returns uppercase
    assert d["layout"] == "TILE"  # layout.__name__ returns "TILE" for TILE_LAYOUT
    assert "memory_config" in d
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
    assert "names" in d
    assert d["names"] == json.dumps(sorted(["weight_a", "weight_b"]))
    assert "dtype" in d
    assert "layout" in d
    assert "memory_config" in d


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


def _minimal_manifest(name: str = "key1"):
    """Create a minimal manifest for storage tests."""
    return create_manifest(
        name,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
        {},
        lambda x: x,
        lambda x: x,
        mesh_mapper=None,
    )


def test_cache_storage_set_get(cache_storage):
    """Verify that InMemoryCacheStorage.set stores a tensor under a key and get retrieves
    the same object; has() returns True for that key."""
    tensor = ttnn.from_torch(torch.zeros((32, 32), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    manifest = _minimal_manifest("key1")

    cache_storage.set("key1", tensor, manifest=manifest)
    retrieved = cache_storage.get("key1")

    assert retrieved is tensor
    assert cache_storage.has("key1")


def test_cache_storage_has(cache_storage):
    """Verify that has() returns False for keys that were never set, and True for a key
    after set(); a different key remains False."""
    assert not cache_storage.has("nonexistent")

    tensor = ttnn.from_torch(torch.zeros((32, 32), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cache_storage.set("key1", tensor, manifest=_minimal_manifest("key1"))

    assert cache_storage.has("key1")
    assert not cache_storage.has("key2")


def test_cache_storage_get_nonexistent(cache_storage):
    """Verify that get() raises KeyError when the key has not been set."""
    with pytest.raises(KeyError):
        cache_storage.get("nonexistent")


def test_inmemory_cache_storage_keys_and_get_manifest(cache_storage):
    """InMemoryCacheStorage: keys() returns stored keys; get_manifest() returns stored CacheManifest; KeyError on miss."""
    tensor = ttnn.from_torch(
        torch.ones((8, 8), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    manifest = create_manifest(
        "model.layers.0.weight",
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
        {},
        lambda x: x,
        lambda x: x,
        mesh_mapper=None,
    )
    cache_storage.set("fp1", tensor, manifest=manifest)
    assert cache_storage.keys() == ["fp1"]
    stored = cache_storage.get_manifest("fp1")
    assert isinstance(stored, CacheManifest)
    assert stored.name == "model.layers.0.weight"
    with pytest.raises(KeyError):
        cache_storage.get_manifest("nonexistent")


def test_ondisk_cache_storage_set_has_get(tmp_path, device):
    """OnDiskCacheStorage: set stores tensor and meta, has() and get() work; loads to configured device."""
    storage = OnDiskCacheStorage(tmp_path, device=device)
    tensor = ttnn.from_torch(torch.zeros((32, 32), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    manifest = create_manifest(
        "weight1",
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
        {"hidden_size": 128},
        lambda x: x,
        lambda x: x,
        mesh_mapper=None,
    )
    storage.set("fp1", tensor, manifest=manifest)
    assert storage.has("fp1")
    loaded = storage.get("fp1")
    assert loaded.shape == tensor.shape
    assert loaded.dtype == tensor.dtype
    assert loaded.layout == tensor.layout
    assert not storage.has("nonexistent")
    with pytest.raises(KeyError):
        storage.get("nonexistent")


def test_ondisk_cache_storage_keys_and_get_manifest(tmp_path, device):
    """OnDiskCacheStorage: keys() returns stored keys; get_manifest() returns CacheManifest; KeyError on miss."""
    storage = OnDiskCacheStorage(tmp_path, device=device)
    tensor = ttnn.from_torch(
        torch.ones((8, 8), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    manifest = create_manifest(
        "model.layers.0.weight",
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
        {},
        lambda x: x,
        lambda x: x,
        mesh_mapper=None,
    )
    storage.set("fp_entries", tensor, manifest=manifest)
    keys = storage.keys()
    assert keys == ["fp_entries"]
    stored = storage.get_manifest("fp_entries")
    assert isinstance(stored, CacheManifest)
    assert stored.name == "model.layers.0.weight"
    assert stored.dtype == ttnn.bfloat16
    assert stored.layout == ttnn.ROW_MAJOR_LAYOUT
    with pytest.raises(KeyError):
        storage.get_manifest("nonexistent")


def test_tensor_cache_accepts_protocol_storage(sample_state_dict, sample_hf_config):
    """TensorCache works with any storage that implements the CacheStorage protocol (e.g. InMemoryCacheStorage)."""
    storage: CacheStorage = InMemoryCacheStorage()
    cache = TensorCache(sample_state_dict, sample_hf_config, storage)
    tensor = get_tensor(cache, "weight1", dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    assert tensor is not None
    assert storage.has(
        create_manifest(
            ["weight1"],
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            ttnn.DRAM_MEMORY_CONFIG,
            sample_hf_config,
            identity_preprocessor,
            identity_postprocessor,
            None,
        ).get_fingerprint()
    )
    assert len(storage.keys()) == 1
    manifest = storage.get_manifest(storage.keys()[0])
    name = manifest.name if isinstance(manifest.name, str) else list(manifest.name)
    assert "weight1" in name


def test_tensor_cache_cache_miss(tensor_cache, monkeypatch):
    """Verify that the first get_tensor call for a given (name, dtype, layout) triggers a
    cache miss: storage.has returns False, set is called once, get is not called; returned
    tensor has the requested dtype and layout."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    # First call should be a cache miss
    tensor = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Verify cache miss behavior
    assert tensor is not None
    assert tensor.dtype == ttnn.bfloat16
    assert tensor.layout == ttnn.TILE_LAYOUT
    assert_cache_miss(call_tracker)


def test_tensor_cache_cache_hit(tensor_cache, monkeypatch):
    """Verify that a second get_tensor call with the same (name, dtype, layout) is a cache
    hit: storage.has returns True, get is called once, set is not called, and the same
    tensor object is returned."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    # First call - cache miss
    tensor1 = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    assert_cache_miss(call_tracker)

    # Second call - cache hit (should return same tensor)
    tensor2 = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Should be the same tensor object (cached)
    assert tensor1 is tensor2
    assert_cache_hit(call_tracker)


def test_tensor_cache_different_dtypes(tensor_cache, monkeypatch):
    """Verify that requesting the same tensor name with different dtypes (e.g. bfloat16
    vs float32) results in two cache misses and two distinct tensor objects."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    tensor1 = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    tensor2 = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
    )

    # Second call with different dtype should also be cache miss
    assert tensor1.dtype != tensor2.dtype
    assert tensor1 is not tensor2
    assert_cache_miss(call_tracker)


def test_tensor_cache_different_layouts(tensor_cache, monkeypatch):
    """Verify that requesting the same tensor name with different layouts (e.g. TILE vs
    ROW_MAJOR) results in two cache misses and two distinct tensor objects."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    tensor1 = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    tensor2 = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Second call with different layout should also be cache miss
    assert tensor1.layout != tensor2.layout
    assert tensor1 is not tensor2
    assert_cache_miss(call_tracker)


def test_tensor_cache_different_preprocessors(tensor_cache, monkeypatch):
    """Verify that the same tensor name with different preprocessor callables (e.g. scale
    by 2 vs scale by 3) results in two cache misses and two distinct tensor objects."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    def preprocessor1(x):
        return x * 2

    def preprocessor2(x):
        return x * 3

    tensor1 = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=preprocessor1,
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    tensor2 = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=preprocessor2,
    )

    # Should be different cache entries (different preprocessors)
    assert tensor1 is not tensor2
    assert_cache_miss(call_tracker)


def test_tensor_cache_different_postprocessors(tensor_cache, monkeypatch):
    """Verify that different postprocessor function objects (even with identical behavior)
    produce different cache entries because fingerprinting uses function identity."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    def postprocessor1(x):
        return x

    def postprocessor2(x):
        return x  # Same function but different object

    tensor1 = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        postprocessor=postprocessor1,
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    tensor2 = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        postprocessor=postprocessor2,
    )

    # Different function objects should create different cache entries
    assert tensor1 is not tensor2
    assert_cache_miss(call_tracker)


def test_tensor_cache_missing_tensor(tensor_cache):
    """Verify that get_tensor raises KeyError with a message including the missing name
    when the requested tensor key is not in the state dict."""
    with pytest.raises(KeyError, match="Tensor 'nonexistent' not found"):
        get_tensor(
            tensor_cache,
            name="nonexistent",
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )


def test_tensor_cache_multiple_tensors(tensor_cache, monkeypatch):
    """Verify that requesting different tensor names (weight1, weight2) each causes a cache
    miss and returns distinct tensors; repeating the same requests then yields cache hits
    and the same tensor objects."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    tensor1 = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    tensor2 = get_tensor(
        tensor_cache,
        name="weight2",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Second call with different name should also be cache miss
    assert tensor1 is not tensor2
    assert_cache_miss(call_tracker)

    # Both should be cached (cache hits)
    tensor1_cached = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # First cache hit
    assert tensor1 is tensor1_cached
    assert_cache_hit(call_tracker)

    tensor2_cached = get_tensor(
        tensor_cache,
        name="weight2",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Second cache hit
    assert tensor2 is tensor2_cached
    assert_cache_hit(call_tracker)


def test_tensor_cache_validation_assertions(tensor_cache, monkeypatch):
    """Verify that on a cache hit, the returned tensor is checked for matching dtype and
    layout; the test creates an entry then retrieves it and asserts cache hit and
    correct dtype/layout."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    # Create a cache entry
    get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    # Retrieve from cache - should validate dtype and layout match
    cached_tensor = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Second call should be cache hit
    assert cached_tensor.dtype == ttnn.bfloat16
    assert cached_tensor.layout == ttnn.TILE_LAYOUT
    assert_cache_hit(call_tracker)


def test_tensor_cache_preprocessor_application(tensor_cache):
    """Verify that a custom preprocessor (e.g. scaling the tensor) is invoked when
    creating the cached tensor; the returned ttnn tensor is non-None and has the
    requested dtype."""

    def preprocessor(x):
        return x * 2.0

    tensor = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=preprocessor,
    )

    # Verify tensor was created (preprocessor should have been applied)
    assert tensor is not None
    assert tensor.dtype == ttnn.bfloat16


def test_tensor_cache_postprocessor_application(tensor_cache):
    """Verify that a custom postprocessor is invoked after conversion; the test uses an
    identity postprocessor to ensure the pipeline runs and returns a valid tensor."""

    def postprocessor(x):
        return x  # Identity, but verifies it's called

    tensor = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        postprocessor=postprocessor,
    )

    # Verify tensor was created (postprocessor should have been applied)
    assert tensor is not None
    assert tensor.dtype == ttnn.bfloat16


def test_tensor_cache_default_preprocessor_postprocessor(tensor_cache, monkeypatch):
    """Verify that omitting preprocessor/postprocessor uses default lambdas and that
    explicitly passing a different identity function creates a different cache entry
    (different fingerprint), so the two requests return different tensor objects."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    # Should work with defaults (lambda x: x)
    tensor1 = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    # Explicit identity should create different cache entry (different lambda objects)
    def identity(x):
        return x

    tensor2 = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=identity,
        postprocessor=identity,
    )

    # Different lambda objects = different fingerprints = different cache entries
    assert tensor1 is not tensor2
    assert_cache_miss(call_tracker)


def test_tensor_cache_hf_config_changes_cache(tensor_cache, sample_state_dict, cache_storage):
    """Verify that TensorCaches with different hf_config values produce different cache
    entries for the same tensor request, and that the same hf_config shares the same
    storage (same tensor object when config matches)."""
    # Create cache with config1
    cache1 = TensorCache(sample_state_dict, {"factor": 2}, cache_storage)
    tensor1 = get_tensor(
        cache1,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Create cache with config2 (different config)
    cache2 = TensorCache(sample_state_dict, {"factor": 3}, cache_storage)
    tensor2 = get_tensor(
        cache2,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Create cache with config3 (different config)
    cache3 = TensorCache(sample_state_dict, {"factor": 2}, cache_storage)
    tensor3 = get_tensor(
        cache3,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    assert tensor1 is not tensor2
    assert tensor2 is not tensor3
    assert tensor1 is tensor3


def test_tensor_cache_multi_source_cache_miss_hit(
    sample_state_dict_same_shape, sample_hf_config, cache_storage, monkeypatch
):
    """Verify that get_tensor with a list of names and a stack preprocessor: first call
    is a cache miss and produces a tensor; second call with same names and preprocessor
    is a cache hit and returns the same tensor object."""
    cache = TensorCache(sample_state_dict_same_shape, sample_hf_config, cache_storage)
    call_tracker = setup_cache_tracker(cache, monkeypatch)

    def stack_preprocessor(*tensors):
        return torch.stack(tensors, dim=0)

    tensor1 = get_tensor(
        cache,
        name=["weight1", "weight2"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=stack_preprocessor,
    )
    assert tensor1 is not None
    assert tensor1.dtype == ttnn.bfloat16
    assert tensor1.layout == ttnn.TILE_LAYOUT
    assert_cache_miss(call_tracker)

    tensor2 = get_tensor(
        cache,
        name=["weight1", "weight2"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=stack_preprocessor,
    )
    assert tensor1 is tensor2
    assert_cache_hit(call_tracker)


def test_tensor_cache_multi_source_different_names_different_entries(
    sample_state_dict_same_shape, sample_hf_config, cache_storage, monkeypatch
):
    """Verify that get_tensor with different name lists (e.g. [weight1, weight2] vs
    [weight2, weight3]) and the same preprocessor results in two cache misses and two
    distinct tensor objects."""
    cache = TensorCache(sample_state_dict_same_shape, sample_hf_config, cache_storage)
    call_tracker = setup_cache_tracker(cache, monkeypatch)

    def stack_preprocessor(*tensors):
        return torch.stack(tensors, dim=0)

    tensor_1_2 = get_tensor(
        cache,
        name=["weight1", "weight2"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=stack_preprocessor,
    )
    assert_cache_miss(call_tracker)

    tensor_2_3 = get_tensor(
        cache,
        name=["weight2", "weight3"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=stack_preprocessor,
    )
    assert tensor_1_2 is not tensor_2_3
    assert_cache_miss(call_tracker)


def test_tensor_cache_multi_source_different_preprocessor_different_entries(
    sample_state_dict_same_shape, sample_hf_config, cache_storage, monkeypatch
):
    """Verify that the same list of names with different multi-tensor preprocessors (e.g.
    stack vs sum) produces two cache misses and two distinct tensor objects."""
    cache = TensorCache(sample_state_dict_same_shape, sample_hf_config, cache_storage)
    call_tracker = setup_cache_tracker(cache, monkeypatch)

    def stack_preprocessor(*tensors):
        return torch.stack(tensors, dim=0)

    def sum_preprocessor(*tensors):
        return sum(tensors)

    tensor_stack = get_tensor(
        cache,
        name=["weight1", "weight2"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=stack_preprocessor,
    )
    assert_cache_miss(call_tracker)

    tensor_sum = get_tensor(
        cache,
        name=["weight1", "weight2"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=sum_preprocessor,
    )
    assert tensor_stack is not tensor_sum
    assert_cache_miss(call_tracker)


def test_tensor_cache_multi_source_missing_tensor(tensor_cache):
    """Verify that get_tensor with a list of names raises KeyError when one of the names
    is not in the state dict (e.g. ['weight1', 'nonexistent'])."""

    def stack_preprocessor(*tensors):
        return torch.stack(tensors, dim=0)

    with pytest.raises(KeyError, match="Tensor 'nonexistent' not found"):
        get_tensor(
            tensor_cache,
            name=["weight1", "nonexistent"],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            preprocessor=stack_preprocessor,
        )


def test_tensor_cache_memory_config_validation(tensor_cache, monkeypatch, mesh_device):
    """Verify that when device and memory_config are provided, the cache key includes
    memory_config: same request yields a cache hit and matching memory_config(); a
    request for the same name with a different memory_config is a cache miss and returns
    a tensor with the new memory config."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    # Create a cache entry
    cached_tensor = get_tensor(
        tensor_cache,
        name="weight1",
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=mesh_device,
    )
    # First call should be cache miss
    assert_cache_miss(call_tracker)
    assert cached_tensor.memory_config() == ttnn.L1_MEMORY_CONFIG
    assert cached_tensor.storage_type() == ttnn.StorageType.DEVICE

    # Retrieve from cache - should validate memory config match
    cached_tensor = get_tensor(
        tensor_cache,
        name="weight1",
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=mesh_device,
    )

    # Second call should be cache hit
    assert_cache_hit(call_tracker)
    assert cached_tensor.memory_config() == ttnn.L1_MEMORY_CONFIG
    assert cached_tensor.storage_type() == ttnn.StorageType.DEVICE

    # Load the same tensor again with a different memory config should be a cache miss
    cached_tensor = get_tensor(
        tensor_cache,
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
    tensor_a1 = get_tensor(
        tensor_cache,
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
    tensor_a2 = get_tensor(
        tensor_cache,
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
    tensor_b = get_tensor(
        tensor_cache,
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
    tensor_1 = get_tensor(
        tensor_cache,
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
    tensor_2 = get_tensor(
        tensor_cache,
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
    tensor_shard = get_tensor(
        tensor_cache,
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=mesh_device,
        mesh_mapper=shard_mapper,
    )
    assert tensor_shard is not tensor_1
    assert_cache_miss(call_tracker)
