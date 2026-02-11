"""
Tensor cache for converted weights with pluggable storage.

TensorCache uses a CacheStorage backend to store and retrieve converted tensors by
fingerprint (derived from manifest). All backends must implement the CacheStorage protocol:

- has(key), get(key), set(key, tensor, *, manifest)
- keys() -> list of cache keys
- get_manifest(key) -> CacheManifest (raises KeyError if key missing)

Manifest is required on set() for every backend; backends that do not persist metadata
still accept and store it (e.g. InMemoryCacheStorage keeps it in memory). Backends that
require a device should accept it in their constructor. get() and get_manifest() raise
KeyError on cache miss. Implementations: InMemoryCacheStorage, OnDiskCacheStorage.
"""
import inspect
import json
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence, Union

import torch

import ttnn

# Type for mesh mapper: CppTensorToMesh (create_mesh_mapper / ShardTensor2dMesh) or ReplicateTensorToMeshWrapper
MeshMapper = ttnn.CppTensorToMesh | ttnn.ReplicateTensorToMeshWrapper


def identity_preprocessor(*tensors: torch.Tensor) -> torch.Tensor:
    if len(tensors) != 1:
        raise ValueError("Default preprocessor expects exactly one tensor input")
    return tensors[0]


def identity_postprocessor(tensor: ttnn.Tensor) -> ttnn.Tensor:
    return tensor


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


def mesh_mapper_to_dict(mesh_mapper: MeshMapper) -> dict:
    config = mesh_mapper.get_config()
    placements = [None if isinstance(p, ttnn.PlacementReplicate) else p.dim for p in config.placements]
    mesh_shape_override_val = config.mesh_shape_override
    mesh_shape_override = tuple(mesh_shape_override_val) if mesh_shape_override_val is not None else None
    return {
        "mesh_shape_override": mesh_shape_override,
        "placements": placements,
    }


@dataclass
class CacheManifest:
    """
    Holds all inputs that define a cache entry for a tensor.
    Does not use tensor content—only name (HF state dict key), dtype, layout,
    memory_config, mesh_mapper, hf_config, and preprocessor/postprocessor fingerprints.
    If the contents of the tensors in the state dict change, the cache must be busted explicitly.
    """

    name: str | Sequence[str]
    dtype: ttnn.DataType
    layout: ttnn.Layout
    memory_config: ttnn.MemoryConfig
    hf_config: dict
    preprocessor: Callable
    postprocessor: Callable
    mesh_mapper: MeshMapper | None = None

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
            "mesh_mapper": mesh_mapper_to_dict(self.mesh_mapper) if self.mesh_mapper is not None else None,
            "hf_config": json.dumps(self.hf_config, sort_keys=True),
            "preprocessor": compute_func_fingerprint(self.preprocessor),
            "postprocessor": compute_func_fingerprint(self.postprocessor),
        }

    def get_fingerprint(self) -> str:
        """Compute a stable fingerprint for this manifest."""
        return compute_fingerprint(self.to_dict())


def create_manifest(
    name: str | Sequence[str],
    dtype: ttnn.DataType,
    layout: ttnn.Layout,
    memory_config: ttnn.MemoryConfig,
    hf_config: dict,
    preprocessor: Callable,
    postprocessor: Callable,
    mesh_mapper: MeshMapper | None = None,
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
        mesh_mapper=mesh_mapper,
    )


def compute_fingerprint(manifest: dict[str, Any]) -> str:
    """Compute a stable fingerprint from a manifest (or its dict representation)."""
    return md5(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def manifest_from_meta_dict(meta: dict[str, Any]) -> CacheManifest:
    """Build a CacheManifest from persisted meta dict (e.g. from disk). Callables are placeholders."""
    name = meta.get("name", "?")
    if isinstance(name, list):
        name = tuple(name)
    dtype_str = meta.get("dtype", "bfloat16")
    layout_str = meta.get("layout", "ROW_MAJOR_LAYOUT")
    dtype = getattr(ttnn, dtype_str, ttnn.bfloat16)
    layout = getattr(ttnn, layout_str, ttnn.ROW_MAJOR_LAYOUT)
    mem_cfg = None
    if meta.get("memory_config"):
        try:
            mem_cfg = ttnn.MemoryConfig.from_json(json.dumps(meta["memory_config"]))
        except Exception:
            pass
    if mem_cfg is None:
        mem_cfg = ttnn.DRAM_MEMORY_CONFIG
    hf_config = meta.get("hf_config", {})
    if isinstance(hf_config, str):
        hf_config = json.loads(hf_config) if hf_config else {}
    return CacheManifest(
        name=name,
        dtype=dtype,
        layout=layout,
        memory_config=mem_cfg,
        hf_config=hf_config,
        preprocessor=lambda x: x,
        postprocessor=lambda x: x,
        mesh_mapper=None,
    )


class CacheStorage(Protocol):
    """
    Contract for cache storage backends used by TensorCache.
    All backends must implement has, get, set (with required manifest), keys(), and get_manifest().
    get() and get_manifest() raise KeyError when the key is missing. Backends that require a
    device should accept it in their constructor.
    """

    def has(self, key: str) -> bool:
        ...

    def get(self, key: str) -> ttnn.Tensor:
        ...

    def set(
        self,
        key: str,
        tensor: ttnn.Tensor,
        *,
        manifest: CacheManifest,
    ) -> None:
        ...

    def keys(self) -> list[str]:
        ...

    def get_manifest(self, key: str) -> CacheManifest:
        ...


class InMemoryCacheStorage:
    """
    Cache backed by host memory. Does not persist across runs.
    Stores both tensor and manifest per key; satisfies CacheStorage protocol.
    """

    def __init__(self) -> None:
        self._tensors: dict[str, ttnn.Tensor] = {}
        self._manifests: dict[str, CacheManifest] = {}

    def has(self, key: str) -> bool:
        return key in self._tensors

    def get(
        self,
        key: str,
    ) -> ttnn.Tensor:
        if key not in self._tensors:
            raise KeyError(f"Cache miss for key {key}")
        return self._tensors[key]

    def set(
        self,
        key: str,
        tensor: ttnn.Tensor,
        *,
        manifest: CacheManifest,
    ) -> None:
        self._tensors[key] = tensor
        self._manifests[key] = manifest

    def keys(self) -> list[str]:
        return list(self._tensors.keys())

    def get_manifest(self, key: str) -> CacheManifest:
        if key not in self._manifests:
            raise KeyError(f"Cache miss for key {key}")
        return self._manifests[key]


def _manifest_and_tensor_to_meta(manifest: CacheManifest, tensor: ttnn.Tensor) -> dict[str, Any]:
    """Build a JSON-serializable meta dict from manifest + tensor for disk persistence."""
    shape = getattr(tensor, "shape", None)
    shape_list = list(shape) if shape is not None else []
    return {
        "name": manifest.name if isinstance(manifest.name, str) else list(manifest.name),
        "dtype": manifest.dtype.__name__,
        "layout": manifest.layout.__name__,
        "memory_config": memory_config_to_dict(manifest.memory_config) if manifest.memory_config else None,
        "mesh_mapper": mesh_mapper_to_dict(manifest.mesh_mapper) if manifest.mesh_mapper else None,
        "hf_config": manifest.hf_config,
        "shape": shape_list,
    }


class OnDiskCacheStorage:
    """
    Cache backed by disk. Persists tensors with ttnn.dump_tensor and metadata in JSON.
    Satisfies CacheStorage: set() requires manifest and writes .meta.json; get_manifest()
    reconstructs CacheManifest from persisted meta (preprocessor/postprocessor are placeholders).
    Safe to use at module scope: on get() we load from disk and place on the configured device.
    """

    TENSOR_SUFFIX = ".tensorbin"
    META_SUFFIX = ".meta.json"

    def __init__(self, cache_dir: Union[str, Path], *, device: ttnn.Device) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if device is None:
            raise ValueError("OnDiskCacheStorage requires a non-None device")
        self.device = device

    def _tensor_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}{self.TENSOR_SUFFIX}"

    def _meta_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}{self.META_SUFFIX}"

    def set(
        self,
        key: str,
        tensor: ttnn.Tensor,
        *,
        manifest: CacheManifest,
    ) -> None:
        path_tensor = self._tensor_path(key)
        path_meta = self._meta_path(key)
        ttnn.dump_tensor(path_tensor, tensor)
        meta = _manifest_and_tensor_to_meta(manifest, tensor)
        with open(path_meta, "w") as f:
            json.dump(meta, f, sort_keys=True)

    def has(self, key: str) -> bool:
        return self._tensor_path(key).is_file()

    def get(
        self,
        key: str,
    ) -> ttnn.Tensor:
        path_tensor = self._tensor_path(key)
        if not path_tensor.is_file():
            raise KeyError(f"Cache miss for key {key}")
        tensor = ttnn.load_tensor(path_tensor, device=self.device)
        return tensor
        """
        path_meta = self._meta_path(key)
        if path_meta.is_file() and self.device is not None:
            with open(path_meta) as f:
                meta = json.load(f)
            mem_cfg_dict = meta.get("memory_config")
            if mem_cfg_dict is not None:
                try:
                    mem_cfg = ttnn.MemoryConfig.from_json(json.dumps(mem_cfg_dict))
                    tensor = tensor.to(device=self.device, mem_config=mem_cfg)
                except Exception:
                    pass
        """
        return tensor

    def keys(self) -> list[str]:
        """Return list of cache keys (fingerprints) without loading tensors."""
        return [p.name[: -len(self.TENSOR_SUFFIX)] for p in self.cache_dir.glob(f"*{self.TENSOR_SUFFIX}")]

    def get_manifest(self, key: str) -> CacheManifest:
        """Return persisted manifest for this key. Raises KeyError if key or meta is missing."""
        path_meta = self._meta_path(key)
        if not path_meta.is_file():
            raise KeyError(f"Cache miss for key {key}")
        with open(path_meta) as f:
            meta = json.load(f)
        return manifest_from_meta_dict(meta)


def default_converter(
    tensor: torch.Tensor,
    dtype: ttnn.DataType,
    layout: ttnn.Layout,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    device: Optional[ttnn.Device] = None,
    mesh_mapper: MeshMapper | None = None,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
        device=device,
        mesh_mapper=mesh_mapper,
    )


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
        *,
        device: ttnn.Device,
        dtype: ttnn.DataType = ttnn.bfloat16,
        layout: ttnn.Layout = ttnn.ROW_MAJOR_LAYOUT,
        preprocessor: Callable[..., torch.Tensor] = identity_preprocessor,
        postprocessor: Callable[[ttnn.Tensor], ttnn.Tensor] = identity_postprocessor,
        memory_config: Optional[ttnn.MemoryConfig] = ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper: MeshMapper | None = None,
    ) -> ttnn.Tensor:
        if device is None:
            raise ValueError("device is required for get_tensor()")
        # Host tensors will have memory_config=ttnn.DRAM_MEMORY_CONFIG so we need to guard against non-DRAM memory configs here
        # In the future we should probably make host tensors have memory_config = None since it doesn't make sense to have a memory config for a host tensor
        if memory_config is not ttnn.DRAM_MEMORY_CONFIG and device is None:
            raise ValueError("Invalid configuration: specified memory config requires a device")

        names = [name] if isinstance(name, str) else list(name)
        manifest = create_manifest(
            names, dtype, layout, memory_config, self.hf_config, preprocessor, postprocessor, mesh_mapper
        )
        fingerprint = manifest.get_fingerprint()

        if not self.cache_entry_exists_for_fingerprint(fingerprint):
            for n in names:
                if n not in self.state_dict:
                    raise KeyError(
                        f"Tensor '{n}' not found in state_dict. Available keys: {list(self.state_dict.keys())}"
                    )
            # Use sorted order for deterministic cache key and consistent preprocessor input
            ordered_names = sorted(names)
            source_tensors = [self.state_dict[n] for n in ordered_names]
            preprocess_source_tensor = preprocessor(*source_tensors)
            tensor = self.converter(preprocess_source_tensor, dtype, layout, memory_config, device, mesh_mapper)
            tensor = postprocessor(tensor)
            self.storage.set(fingerprint, tensor, manifest=manifest)
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
