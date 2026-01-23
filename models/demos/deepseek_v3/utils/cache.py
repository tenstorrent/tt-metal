import inspect
import json
from hashlib import md5
from typing import Callable, Sequence

import torch

import ttnn


def compute_func_fingerprint(func: Callable) -> str:
    """
    Compute a fingerprint for a function/callable.

    For named functions, attempts to use source code for stable fingerprinting.
    """
    source = inspect.getsource(func)
    name = getattr(func, "__name__", "")
    fingerprint_input = f"{name}:{source}"
    return md5(fingerprint_input.encode()).hexdigest()


def create_manifest(
    name: str | Sequence[str],
    dtype: ttnn.DataType,
    layout: ttnn.Layout,
    hf_config: dict,
    preprocessor: Callable,
    postprocessor: Callable,
) -> dict[str, str]:
    """
    The manifest does not use the content of the tensor and only uses the 'name' (i.e. the one in the HF state dict).
    If the contents of the tensors in the state dict changes we would need to explicitly bust the cache.
    For multiple names, uses a deterministic sorted list for cache key stability.
    """
    if isinstance(name, str):
        names_key = "name"
        names_value = name
    else:
        names_key = "names"
        names_value = json.dumps(sorted(name), sort_keys=True)
    return {
        names_key: names_value,
        "dtype": dtype.__name__,
        "layout": layout.__name__,
        "hf_config": json.dumps(hf_config, sort_keys=True),
        "preprocessor": compute_func_fingerprint(preprocessor),
        "postprocessor": compute_func_fingerprint(postprocessor),
    }


def compute_fingerprint(manifest: dict[str, str]) -> str:
    return md5(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


class InMemoryCacheStorage:
    """
    A cache backed by host memory. Does not persist across runs.
    We could make this more advanced by implementing LRU caching. Injected into TensorCache to facilitate unit testing.
    """

    def __init__(self):
        self.cache = {}

    def set(self, key: str, tensor: ttnn.Tensor):
        self.cache[key] = tensor

    def has(self, key: str) -> bool:
        return key in self.cache

    def get(self, key: str) -> ttnn.Tensor:
        return self.cache[key]


# TODO: Implement as Union[InMemoryCacheStorage, OnDiskCacheStorage, RedisCacheStorage, etc.]
CacheStorage = InMemoryCacheStorage


def default_converter(tensor: torch.Tensor, dtype: ttnn.DataType, layout: ttnn.Layout) -> ttnn.Tensor:
    tensor = ttnn.from_torch(tensor, dtype=dtype, layout=layout)
    return tensor


class TensorCache:
    def __init__(
        self,
        state_dict: dict[str, torch.Tensor],
        hf_config: dict,
        storage: CacheStorage,
        converter: Callable = default_converter,
    ):
        self.state_dict = state_dict
        self.hf_config = hf_config
        self.storage = storage
        self.converter = converter

    def cache_entry_exists_for_fingerprint(self, fingerprint: str):
        return self.storage.has(fingerprint)

    def get_tensor(
        self,
        name: str | Sequence[str],
        dtype: ttnn.DataType,
        layout: ttnn.Layout,
        preprocessor: Callable = lambda x: x,
        postprocessor: Callable = lambda x: x,
    ) -> ttnn.Tensor:
        names = [name] if isinstance(name, str) else list(name)
        manifest = create_manifest(names, dtype, layout, self.hf_config, preprocessor, postprocessor)
        fingerprint = compute_fingerprint(manifest)

        if not self.cache_entry_exists_for_fingerprint(fingerprint):
            for n in names:
                if n not in self.state_dict:
                    raise KeyError(
                        f"Tensor '{n}' not found in state_dict. Available keys: {list(self.state_dict.keys())}"
                    )
            # Use sorted order for deterministic cache key and consistent preprocessor input
            ordered_names = sorted(names)
            hf_tensors = [self.state_dict[n] for n in ordered_names]
            if len(hf_tensors) == 1:
                preprocessed_hf_tensor = preprocessor(hf_tensors[0])
            else:
                preprocessed_hf_tensor = preprocessor(hf_tensors)
            tensor = self.converter(preprocessed_hf_tensor, dtype, layout)
            tensor = postprocessor(tensor)
            self.storage.set(fingerprint, tensor)
            return tensor
        else:
            # Validate cached tensor matches requested dtype and layout
            cached_tensor = self.storage.get(fingerprint)
            assert cached_tensor.dtype == dtype, (
                f"Cached tensor dtype mismatch: expected {dtype}, got {cached_tensor.dtype}. "
                f"This should not happen if fingerprinting is correct."
            )
            assert cached_tensor.layout == layout, (
                f"Cached tensor layout mismatch: expected {layout}, got {cached_tensor.layout}. "
                f"This should not happen if fingerprinting is correct."
            )
            return cached_tensor


hf_weights = "model.safetensors"
hf_config = {"factor": 2}

"""
Initial state_dict is created as a function of the hf_config so we will need to trigger rebuild any persistent caches when the config changes
"""

"""
tensors = {
    "weight1": torch.zeros((hf_config["factor"] * 128, hf_config["factor"] * 128), dtype=torch.bfloat16),
    "weight2": torch.zeros((hf_config["factor"] * 256, hf_config["factor"] * 256), dtype=torch.bfloat16),
}
save_file(tensors, hf_weights)

state_dict = {}
with safe_open(hf_weights, framework="pt", device="cpu") as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key)  # TODO: Is this lazy? We should probably make it lazy

cache = TensorCache(state_dict, hf_config, InMemoryCacheStorage())
cache.get_tensor("weight1", dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)  # Should be a cache miss
cache.get_tensor("weight1", dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)  # Should be a cache hit
cache.get_tensor(
    "weight1", dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, preprocessor=lambda x: x * 2
)  # Should be a cache miss
cache.get_tensor("weight1", dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)  # SHould be a cache miss
"""
