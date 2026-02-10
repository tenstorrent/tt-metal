import inspect
import json
from dataclasses import dataclass
from hashlib import md5
from typing import Any, Callable, Optional, Sequence

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


def memory_config_to_dict(memory_config: ttnn.MemoryConfig) -> dict:
    return {
        "memory_layout": memory_config.memory_layout.__name__,
        "buffer_type": memory_config.buffer_type.__name__,
        "shard_spec": str(memory_config.shard_spec),
        "is_sharded": memory_config.is_sharded(),
        "interleaved": memory_config.interleaved,
        "hash": int(memory_config.__hash__()),
    }


@dataclass
class CacheManifest:
    """
    Holds all inputs that define a cache entry for a tensor.
    Does not use tensor content—only name (HF state dict key), dtype, layout,
    memory_config, hf_config, and preprocessor/postprocessor fingerprints.
    If the contents of the tensors in the state dict change, the cache must be busted explicitly.
    """

    name: str | Sequence[str]
    dtype: ttnn.DataType
    layout: ttnn.Layout
    memory_config: ttnn.MemoryConfig
    hf_config: dict
    preprocessor: Callable
    postprocessor: Callable

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict for fingerprinting."""
        if isinstance(self.name, str):
            names_key = "name"
            names_value = self.name
        else:
            names_key = "names"
            names_value = json.dumps(sorted(self.name), sort_keys=True)
        return {
            names_key: names_value,
            "dtype": self.dtype.__name__,
            "layout": self.layout.__name__,
            "memory_config": memory_config_to_dict(self.memory_config) if self.memory_config is not None else None,
            "hf_config": json.dumps(self.hf_config, sort_keys=True),
            "preprocessor": compute_func_fingerprint(self.preprocessor),
            "postprocessor": compute_func_fingerprint(self.postprocessor),
        }


def create_manifest(
    name: str | Sequence[str],
    dtype: ttnn.DataType,
    layout: ttnn.Layout,
    memory_config: ttnn.MemoryConfig,
    hf_config: dict,
    preprocessor: Callable,
    postprocessor: Callable,
) -> CacheManifest:
    """Build a CacheManifest for the given tensor request parameters."""
    return CacheManifest(
        name=name,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
        hf_config=hf_config,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )


def compute_fingerprint(manifest: dict[str, Any]) -> str:
    """Compute a stable fingerprint from a manifest (or its dict representation)."""
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


def default_converter(
    tensor: torch.Tensor,
    dtype: ttnn.DataType,
    layout: ttnn.Layout,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    device: Optional[ttnn.Device] = None,
) -> ttnn.Tensor:
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, memory_config=memory_config, device=device)


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
        dtype: ttnn.DataType = ttnn.bfloat16,
        layout: ttnn.Layout = ttnn.ROW_MAJOR_LAYOUT,
        preprocessor: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        postprocessor: Callable[[ttnn.Tensor], ttnn.Tensor] = lambda x: x,
        memory_config: Optional[ttnn.MemoryConfig] = ttnn.DRAM_MEMORY_CONFIG,
        device: Optional[ttnn.Device] = None,
    ) -> ttnn.Tensor:
        # Host tensors will have memory_config=ttnn.DRAM_MEMORY_CONFIG so we need to guard against non-DRAM memory configs here
        # In the future we should probably make host tensors have memory_config = None since it doesn't make sense to have a memory config for a host tensor
        if memory_config is not ttnn.DRAM_MEMORY_CONFIG and device is None:
            raise ValueError("Invalid configuration: specified memory config requires a device")

        names = [name] if isinstance(name, str) else list(name)
        manifest = create_manifest(names, dtype, layout, memory_config, self.hf_config, preprocessor, postprocessor)
        fingerprint = compute_fingerprint(manifest.to_dict())

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
            tensor = self.converter(preprocessed_hf_tensor, dtype, layout, memory_config, device)
            tensor = postprocessor(tensor)
            self.storage.set(fingerprint, tensor)
            return tensor
        else:
            # Validate cached tensor matches requested dtype, layout and memory config
            cached_tensor = self.storage.get(fingerprint)
            assert cached_tensor.dtype == dtype, (
                f"Cached tensor dtype mismatch: expected {dtype}, got {cached_tensor.dtype}. "
                f"This should not happen if fingerprinting is correct."
            )
            assert cached_tensor.layout == layout, (
                f"Cached tensor layout mismatch: expected {layout}, got {cached_tensor.layout}. "
                f"This should not happen if fingerprinting is correct."
            )
            assert cached_tensor.memory_config() == memory_config, (
                f"Cached tensor memory config mismatch: expected {memory_config}, got {cached_tensor.memory_config()}. "
                f"This should not happen if fingerprinting is correct."
            )
            return cached_tensor
