import json
from hashlib import md5
from typing import Callable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import ttnn


def compute_func_fingerprint(func: Callable) -> str:
    code = func.__code__.co_code
    return md5(code).hexdigest()


def create_manifest(
    name: str, dtype: ttnn.DataType, layout: ttnn.Layout, schema: int, preprocessor: Callable, postprocessor: Callable
) -> dict[str, str]:
    """
    The manifest does not use the content of the tensor and only uses the 'name' (i.e. the one in the HF state dict).
    If the contents of the tensors in the state dict changes we would need to explicitly bust the cache.
    """
    return {
        "name": name,
        "dtype": dtype.__name__,
        "layout": layout.__name__,
        "schema": str(schema),
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
        self.cache.get(key)


# TODO: Implement as Union[InMemoryCacheStorage, OnDiskCacheStorage, RedisCacheStorage, etc.]
CacheStorage = InMemoryCacheStorage


class TensorCache:
    def __init__(self, state_dict: dict[str, torch.Tensor], hf_config: dict, storage: CacheStorage):
        self.state_dict = state_dict
        self.hf_config = hf_config
        self.storage = storage

    def cache_entry_exists_for_fingerprint(self, fingerprint: str):
        return self.storage.has(fingerprint)

    def get_tensor(
        self,
        name: str,
        dtype: ttnn.DataType,
        layout: ttnn.Layout,
        preprocessor: Callable = lambda x: x,
        postprocessor: Callable = lambda x: x,
    ) -> ttnn.Tensor:
        manifest = create_manifest(name, dtype, layout, self.hf_config, preprocessor, postprocessor)
        fingerprint = compute_fingerprint(manifest)

        if not self.cache_entry_exists_for_fingerprint(fingerprint):
            hf_tensor = self.state_dict[name]
            preprocessed_hf_tensor = preprocessor(hf_tensor)
            tensor = ttnn.from_torch(preprocessed_hf_tensor, dtype=dtype, layout=layout)
            tensor = postprocessor(tensor)
            self.storage.set(fingerprint, tensor)

        return self.storage.get(fingerprint)


hf_weights = "model.safetensors"
hf_config = {"factor": 2}

"""
Initial state_dict is created as a function of the hf_config so we will need to trigger rebuild any persistent caches when the config changes
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
