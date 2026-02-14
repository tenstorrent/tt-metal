import inspect
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

import torch

import ttnn

# Type for mesh mapper: CppTensorToMesh (create_mesh_mapper / ShardTensor2dMesh) or ReplicateTensorToMeshWrapper
MeshMapper = ttnn.CppTensorToMesh | ttnn.ReplicateTensorToMeshWrapper


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
        return compute_fingerprint(self.to_dict())


@dataclass(frozen=True)
class CacheKey:
    fingerprint: str
    manifest: CacheManifest


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CacheStorage(Protocol):
    def set(self, key: CacheKey, tensor: ttnn.Tensor) -> None:
        ...

    def has(self, key: CacheKey) -> bool:
        ...

    def get(self, key: CacheKey) -> ttnn.Tensor:
        ...


class InMemoryCacheStorage:
    """
    A cache backed by host memory. Does not persist across runs.
    We could make this more advanced by implementing LRU caching. Injected into TensorCache to facilitate unit testing.
    """

    def __init__(self):
        self.cache = {}

    def set(self, key: CacheKey, tensor: ttnn.Tensor):
        self.cache[key.fingerprint] = tensor

    def has(self, key: CacheKey) -> bool:
        return key.fingerprint in self.cache

    def get(self, key: CacheKey) -> ttnn.Tensor:
        return self.cache[key.fingerprint]


def _safe_name_from_manifest(manifest: CacheManifest) -> str:
    if isinstance(manifest.name, str):
        raw_names = [manifest.name]
    else:
        raw_names = sorted(manifest.name)
    safe_names = [re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") for name in raw_names]
    safe_name = "__".join(n for n in safe_names if n)
    return safe_name or "tensor"


class OnDiskCacheStorage:
    def __init__(self, cache_root: str | Path, device: ttnn.Device | ttnn.MeshDevice):
        self.cache_root = Path(cache_root)
        self.device = device

    def _tensor_path(self, key: CacheKey) -> Path:
        safe_name = _safe_name_from_manifest(key.manifest)
        return self.cache_root / f"{safe_name}__{key.fingerprint}.tensorbin"

    def _manifest_path(self, key: CacheKey) -> Path:
        safe_name = _safe_name_from_manifest(key.manifest)
        return self.cache_root / f"{safe_name}__{key.fingerprint}.manifest.json"

    def set(self, key: CacheKey, tensor: ttnn.Tensor) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        tensor_path = self._tensor_path(key)
        ttnn.dump_tensor(tensor_path, tensor)

        manifest_path = self._manifest_path(key)
        created_at = None
        if manifest_path.is_file():
            try:
                existing_payload = json.loads(manifest_path.read_text())
                created_at = existing_payload.get("created_at")
            except json.JSONDecodeError:
                raise ValueError(f"Invalid manifest file: {manifest_path}")
        now = _utc_now_iso()
        if created_at is None:
            created_at = now
        manifest_payload = key.manifest.to_dict()
        manifest_payload["fingerprint"] = key.fingerprint
        manifest_payload["created_at"] = created_at
        manifest_payload["last_modified"] = now
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2, sort_keys=True)

    def has(self, key: CacheKey) -> bool:
        return self._tensor_path(key).is_file()

    def get(self, key: CacheKey) -> ttnn.Tensor:
        tensor_path = self._tensor_path(key)
        if not tensor_path.is_file():
            raise KeyError(key.fingerprint)
        return ttnn.load_tensor(tensor_path, device=self.device)


def default_converter(
    tensor: torch.Tensor,
    dtype: ttnn.DataType,
    layout: ttnn.Layout,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    *,
    device: ttnn.Device | ttnn.MeshDevice,
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

    def cache_entry_exists_for_key(self, key: CacheKey):
        return self.storage.has(key)

    def get_tensor(
        self,
        name: str | Sequence[str],
        dtype: ttnn.DataType = ttnn.bfloat16,
        layout: ttnn.Layout = ttnn.ROW_MAJOR_LAYOUT,
        preprocessor: Callable[[Sequence[torch.Tensor]], torch.Tensor] = lambda x: x,
        postprocessor: Callable[[ttnn.Tensor], ttnn.Tensor] = lambda x: x,
        memory_config: Optional[ttnn.MemoryConfig] = ttnn.DRAM_MEMORY_CONFIG,
        *,
        device: ttnn.Device | ttnn.MeshDevice,
        mesh_mapper: MeshMapper | None = None,
    ) -> ttnn.Tensor:
        if device is None:
            raise ValueError("Invalid configuration: device must be provided")

        names = [name] if isinstance(name, str) else list(name)
        manifest = create_manifest(
            names, dtype, layout, memory_config, self.hf_config, preprocessor, postprocessor, mesh_mapper
        )
        fingerprint = manifest.get_fingerprint()
        key = CacheKey(fingerprint=fingerprint, manifest=manifest)

        if not self.cache_entry_exists_for_key(key):
            for n in names:
                if n not in self.state_dict:
                    raise KeyError(
                        f"Tensor '{n}' not found in state_dict. Available keys: {list(self.state_dict.keys())}"
                    )
            # Use sorted order for deterministic cache key and consistent preprocessor input
            ordered_names = sorted(names)
            source_tensors = [self.state_dict[n] for n in ordered_names]
            preprocess_source_tensor = preprocessor(*source_tensors)
            tensor = self.converter(
                preprocess_source_tensor,
                dtype,
                layout,
                memory_config=memory_config,
                device=device,
                mesh_mapper=mesh_mapper,
            )
            tensor = postprocessor(tensor)
            self.storage.set(key, tensor)
            return tensor
        else:
            # Validate cached tensor matches requested dtype, layout and memory config
            cached_tensor = self.storage.get(key)
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
