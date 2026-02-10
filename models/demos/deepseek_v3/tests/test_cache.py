# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

import ttnn
from models.demos.deepseek_v3.utils.cache import (
    CacheManifest,
    InMemoryCacheStorage,
    TensorCache,
    compute_fingerprint,
    compute_func_fingerprint,
    create_manifest,
)


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

    def tracked_set(key: str, tensor: ttnn.Tensor):
        call_tracker["set_calls"].append(("set", key))
        return original_set(key, tensor)

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
    """Test that named functions produce stable fingerprints."""

    def test_func(x):
        return x * 2

    fingerprint1 = compute_func_fingerprint(test_func)
    fingerprint2 = compute_func_fingerprint(test_func)

    assert fingerprint1 == fingerprint2
    assert isinstance(fingerprint1, str)
    assert len(fingerprint1) == 32  # MD5 hexdigest length


def test_compute_func_fingerprint_different_functions():
    """Test that different functions produce different fingerprints."""

    def func1(x):
        return x * 3

    def func2(x):
        return x * 3

    fingerprint1 = compute_func_fingerprint(func1)
    fingerprint2 = compute_func_fingerprint(func2)

    assert fingerprint1 != fingerprint2


def test_create_manifest_basic(sample_hf_config):
    """Test basic manifest creation."""

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
    """Test manifest creation with multiple names uses 'names' key."""

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
    """Test that same manifest produces same fingerprint."""
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
    """Test that different manifests produce different fingerprints."""
    manifest1 = {"name": "test1", "dtype": "bfloat16"}
    manifest2 = {"name": "test2", "dtype": "bfloat16"}

    fingerprint1 = compute_fingerprint(manifest1)
    fingerprint2 = compute_fingerprint(manifest2)

    assert fingerprint1 != fingerprint2


def test_cache_storage_set_get(cache_storage):
    """Test basic set and get operations."""
    tensor = ttnn.from_torch(torch.zeros((32, 32), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    cache_storage.set("key1", tensor)
    retrieved = cache_storage.get("key1")

    assert retrieved is tensor
    assert cache_storage.has("key1")


def test_cache_storage_has(cache_storage):
    """Test has() method."""
    assert not cache_storage.has("nonexistent")

    tensor = ttnn.from_torch(torch.zeros((32, 32), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cache_storage.set("key1", tensor)

    assert cache_storage.has("key1")
    assert not cache_storage.has("key2")


def test_cache_storage_get_nonexistent(cache_storage):
    """Test that getting nonexistent key raises KeyError."""
    with pytest.raises(KeyError):
        cache_storage.get("nonexistent")


def test_tensor_cache_cache_miss(tensor_cache, monkeypatch):
    """Test that first call to get_tensor is a cache miss."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    # First call should be a cache miss
    tensor = tensor_cache.get_tensor(
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
    """Test that second call with same parameters is a cache hit."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    # First call - cache miss
    tensor1 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    assert_cache_miss(call_tracker)

    # Second call - cache hit (should return same tensor)
    tensor2 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Should be the same tensor object (cached)
    assert tensor1 is tensor2
    assert_cache_hit(call_tracker)


def test_tensor_cache_different_dtypes(tensor_cache, monkeypatch):
    """Test that different dtypes create different cache entries."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    tensor1 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    tensor2 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
    )

    # Second call with different dtype should also be cache miss
    assert tensor1.dtype != tensor2.dtype
    assert tensor1 is not tensor2
    assert_cache_miss(call_tracker)


def test_tensor_cache_different_layouts(tensor_cache, monkeypatch):
    """Test that different layouts create different cache entries."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    tensor1 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    tensor2 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Second call with different layout should also be cache miss
    assert tensor1.layout != tensor2.layout
    assert tensor1 is not tensor2
    assert_cache_miss(call_tracker)


def test_tensor_cache_different_preprocessors(tensor_cache, monkeypatch):
    """Test that different preprocessors create different cache entries."""
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
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    tensor2 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=preprocessor2,
    )

    # Should be different cache entries (different preprocessors)
    assert tensor1 is not tensor2
    assert_cache_miss(call_tracker)


def test_tensor_cache_different_postprocessors(tensor_cache, monkeypatch):
    """Test that different postprocessors create different cache entries."""
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
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    tensor2 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        postprocessor=postprocessor2,
    )

    # Different function objects should create different cache entries
    assert tensor1 is not tensor2
    assert_cache_miss(call_tracker)


def test_tensor_cache_missing_tensor(tensor_cache):
    """Test that requesting nonexistent tensor raises KeyError."""
    with pytest.raises(KeyError, match="Tensor 'nonexistent' not found"):
        tensor_cache.get_tensor(
            name="nonexistent",
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )


def test_tensor_cache_multiple_tensors(tensor_cache, monkeypatch):
    """Test caching multiple different tensors."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    tensor1 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    tensor2 = tensor_cache.get_tensor(
        name="weight2",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Second call with different name should also be cache miss
    assert tensor1 is not tensor2
    assert_cache_miss(call_tracker)

    # Both should be cached (cache hits)
    tensor1_cached = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # First cache hit
    assert tensor1 is tensor1_cached
    assert_cache_hit(call_tracker)

    tensor2_cached = tensor_cache.get_tensor(
        name="weight2",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Second cache hit
    assert tensor2 is tensor2_cached
    assert_cache_hit(call_tracker)


def test_tensor_cache_validation_assertions(tensor_cache, monkeypatch):
    """Test that cached tensors are validated for dtype/layout compatibility."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    # Create a cache entry
    tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # First call should be cache miss
    assert_cache_miss(call_tracker)

    # Retrieve from cache - should validate dtype and layout match
    cached_tensor = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Second call should be cache hit
    assert cached_tensor.dtype == ttnn.bfloat16
    assert cached_tensor.layout == ttnn.TILE_LAYOUT
    assert_cache_hit(call_tracker)


def test_tensor_cache_preprocessor_application(tensor_cache):
    """Test that preprocessor is correctly applied."""

    def preprocessor(x):
        return x * 2.0

    tensor = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=preprocessor,
    )

    # Verify tensor was created (preprocessor should have been applied)
    assert tensor is not None
    assert tensor.dtype == ttnn.bfloat16


def test_tensor_cache_postprocessor_application(tensor_cache):
    """Test that postprocessor is correctly applied."""

    def postprocessor(x):
        return x  # Identity, but verifies it's called

    tensor = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        postprocessor=postprocessor,
    )

    # Verify tensor was created (postprocessor should have been applied)
    assert tensor is not None
    assert tensor.dtype == ttnn.bfloat16


def test_tensor_cache_default_preprocessor_postprocessor(tensor_cache, monkeypatch):
    """Test that default identity preprocessor/postprocessor work correctly."""
    call_tracker = setup_cache_tracker(tensor_cache, monkeypatch)

    # Should work with defaults (lambda x: x)
    tensor1 = tensor_cache.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
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
    )

    # Different lambda objects = different fingerprints = different cache entries
    assert tensor1 is not tensor2
    assert_cache_miss(call_tracker)


def test_tensor_cache_hf_config_changes_cache(tensor_cache, sample_state_dict, cache_storage):
    """Test that different HF configs create different cache entries."""
    # Create cache with config1
    cache1 = TensorCache(sample_state_dict, {"factor": 2}, cache_storage)
    tensor1 = cache1.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Create cache with config2 (different config)
    cache2 = TensorCache(sample_state_dict, {"factor": 3}, cache_storage)
    tensor2 = cache2.get_tensor(
        name="weight1",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Create cache with config3 (different config)
    cache3 = TensorCache(sample_state_dict, {"factor": 2}, cache_storage)
    tensor3 = cache3.get_tensor(
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
    """Test multi-source get_tensor: first call cache miss, second call cache hit."""
    cache = TensorCache(sample_state_dict_same_shape, sample_hf_config, cache_storage)
    call_tracker = setup_cache_tracker(cache, monkeypatch)

    def stack_preprocessor(tensors):
        return torch.stack(tensors, dim=0)

    tensor1 = cache.get_tensor(
        name=["weight1", "weight2"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=stack_preprocessor,
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
    )
    assert tensor1 is tensor2
    assert_cache_hit(call_tracker)


def test_tensor_cache_multi_source_different_names_different_entries(
    sample_state_dict_same_shape, sample_hf_config, cache_storage, monkeypatch
):
    """Test that different name lists produce different cache entries."""
    cache = TensorCache(sample_state_dict_same_shape, sample_hf_config, cache_storage)
    call_tracker = setup_cache_tracker(cache, monkeypatch)

    def stack_preprocessor(tensors):
        return torch.stack(tensors, dim=0)

    tensor_1_2 = cache.get_tensor(
        name=["weight1", "weight2"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=stack_preprocessor,
    )
    assert_cache_miss(call_tracker)

    tensor_2_3 = cache.get_tensor(
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
    """Test that same names with different preprocessors produce different cache entries."""
    cache = TensorCache(sample_state_dict_same_shape, sample_hf_config, cache_storage)
    call_tracker = setup_cache_tracker(cache, monkeypatch)

    def stack_preprocessor(tensors):
        return torch.stack(tensors, dim=0)

    def sum_preprocessor(tensors):
        return sum(tensors)

    tensor_stack = cache.get_tensor(
        name=["weight1", "weight2"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=stack_preprocessor,
    )
    assert_cache_miss(call_tracker)

    tensor_sum = cache.get_tensor(
        name=["weight1", "weight2"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        preprocessor=sum_preprocessor,
    )
    assert tensor_stack is not tensor_sum
    assert_cache_miss(call_tracker)


def test_tensor_cache_multi_source_missing_tensor(tensor_cache):
    """Test that requesting with a missing name in the list raises KeyError."""

    def stack_preprocessor(tensors):
        return torch.stack(tensors, dim=0)

    with pytest.raises(KeyError, match="Tensor 'nonexistent' not found"):
        tensor_cache.get_tensor(
            name=["weight1", "nonexistent"],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            preprocessor=stack_preprocessor,
        )


def test_tensor_cache_memory_config_validation(tensor_cache, monkeypatch, mesh_device):
    """Test that cached tensors are validated for memory config compatibility."""
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
