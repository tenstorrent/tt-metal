# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LazyWeight: Lazy weight loading with explicit TTNN conversion parameters.

This module has NO torch dependency - it accepts any tensor-like object that
ttnn.from_torch() can handle (duck typing with string type hints).

Design principles:
- Explicit parameters over hidden closures (IDE-friendly)
- Duck typing for source tensors (no torch import)
- Deterministic cache fingerprinting from configuration
- Lazy conversion: defer ttnn.from_torch until first access
- Cache-first: skip torch entirely if TTNN cache exists
"""

import hashlib
import pathlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from loguru import logger

import ttnn
from models.common.tensor_utils import get_padded_hidden_dim, pad_to_shape, parse_shard_dims_from_mesh_mapper_config


# todo)) maybe useful to support a mechanism that can be used to disable the cache for every LazyWeight instance
# todo)) needs unit tests to provide coverage
@dataclass
class LazyWeight:
    """
    Lazy weight loading with explicit TTNN conversion parameters.

    The source can be:
    - A tensor directly (duck-typed, typically torch.Tensor)

    All fields except source are optional at construction time and can be
    provided later. get_weight() will check

    Example usage:
        # Fully specified at construction
        weight = LazyWeight(
            source=torch_tensor,
            device=mesh_device,
            dtype=ttnn.bfloat16,
        )
        ttnn_tensor = weight.get_weight()

        # Minimal construction, provide defaults at materialize time
        weight = LazyWeight(source=torch_tensor)
        ttnn_tensor = weight.get_weight(device=mesh_device, dtype=ttnn.bfloat16)

        # Partial specification - override defaults selectively
        weight = LazyWeight(source=torch_tensor, dtype=ttnn.bfloat4_b)
        ttnn_tensor = weight.get_weight(device=mesh_device, dtype=ttnn.bfloat16)  # uses bfloat4_b

        # Using convenience class methods
        weight = LazyWeight.sharded_1d(source=tensor, device=mesh_device, dtype=ttnn.bfloat16, dim=3)
        weight = LazyWeight.replicated(source=tensor, device=mesh_device, dtype=ttnn.bfloat16)

    """

    # Source: duck-typed tensor
    # String annotation documents intent without importing torch
    # todo)) use something like LazyStateDict for the source
    source: "torch.Tensor"

    # All other fields are optional at construction time.
    cache_dir_weight_name: Optional[tuple[Path, str]] = None  # do not cache if None
    pad_value: Optional[float] = 0.0
    dtype: Optional[ttnn.DataType] = ttnn.bfloat16
    # Lazy fields that can be materialize at get_weight time;
    # Still named as public fields to allow users to override at construction time (be proactive if you want).
    device: Optional[ttnn.MeshDevice] = None
    mesh_mapper_config: Optional[ttnn.MeshMapperConfig] = None
    memory_config: Optional[ttnn.MemoryConfig] = None
    layout: Optional[ttnn.Layout] = None

    # Private fields
    _value: Optional[ttnn.Tensor] = field(default=None, repr=False)

    def __post_init__(self):
        assert self.source.shape is not None and len(self.source.shape) > 0, "source must have a shape"

    def get_device_weight(self) -> ttnn.Tensor:
        """
        Get the TTNN tensor, converting from source if needed.

        On first call:
        1. Check cache → load if exists
        2. Otherwise: resolve source → convert → cache

        Subsequent calls return the cached tensor (defaults ignored after first call).
        """
        # todo)) is there a better way to enforce immutability of all fields after construction? and self._value after first call?
        if self._value is not None:
            return self._value

        cache_dir, weight_name = None, None
        if self.cache_dir_weight_name is not None:
            cache_dir, weight_name = self.cache_dir_weight_name

        # Validate required fields
        if self.device is None:
            raise ValueError("device must be provided (either on LazyWeight or in defaults)")
        if self.layout is None:
            raise ValueError("layout must be provided (either on LazyWeight or in defaults)")
        if self.memory_config is None:
            raise ValueError("memory_config must be provided (either on LazyWeight or in defaults)")

        # Try loading from cache first (skip torch entirely if possible)
        cache_file_name = self._get_cache_fill_path(
            cache_dir=cache_dir,
            weight_name=weight_name,
        )
        if cache_file_name and cache_file_name.exists():
            logger.info(f"\033[32m[cache hit]\033[0m Loading tensor from cache: {cache_file_name}")
            self._value = ttnn.load_tensor(str(cache_file_name), device=self.device)
            return self._value

        # Resolve source
        tensor = self.source

        # Get mesh mapper (created from config)
        if self.mesh_mapper_config is not None:
            mesh_mapper = ttnn.create_mesh_mapper(self.device, self.mesh_mapper_config)
            # Auto-pad tensor to satisfy ttnn.from_torch's tile alignment constraint
            tensor = _auto_pad_for_sharding(tensor, self.padded_shape, pad_value=self.pad_value)
        else:
            # None config means replicate
            mesh_mapper = ttnn.replicate_tensor_to_mesh_mapper(self.device)
        is_replicated = self.mesh_mapper_config is None

        # Convert to TTNN and cache
        self._value = _from_torch_and_dump(
            tensor=tensor,
            device=self.device,
            dtype=self.dtype,
            layout=self.layout,
            memory_config=self.memory_config,
            mesh_mapper=mesh_mapper,
            is_replicated=is_replicated,
            cache_file_name=cache_file_name,
        )

        return self._value

    # todo)) refactor into a base class for Lazy and Transparent Pattern (LaTr)
    def is_resolved(self) -> bool:
        """Check if all required fields for weight materialization are set.

        Excludes: _value (private), cache_dir_weight_name (optional caching)
        """
        required_fields = ("source", "pad_value", "dtype", "device", "mesh_mapper_config", "memory_config", "layout")
        return all(getattr(self, field) is not None for field in required_fields)

    def _get_cache_fill_path(
        self,
        cache_dir: Optional[Path],
        weight_name: Optional[str],
    ) -> Optional[Path]:
        """Generate the cache file path based on configuration fingerprint."""
        if cache_dir is None or weight_name is None:
            return None
        fingerprint = self._get_fingerprint()
        return cache_dir / f"{weight_name}_{fingerprint}.tensorbin"

    def _get_fingerprint(
        self,
    ) -> str:
        """
        Generate a unique fingerprint from the conversion configuration.

        This ensures cache invalidation when config changes.
        """
        parts = []

        # source
        # todo)) add better fingerprinting for the source tensor to enable cache invalidation when the source tensor changes; for now, use shape.
        parts.append(f"srcshape_{'_'.join(str(dim) for dim in self.source.shape)}")

        # dtype
        parts.append(f"dtype_{self.dtype.name}")

        # layout
        parts.append(f"layout_{self.layout.name}")

        # memory_config (use hash if non-default)
        if self.memory_config is not None and self.memory_config != ttnn.DRAM_MEMORY_CONFIG:
            parts.append(f"memcfg_{hash(self.memory_config)}")

        # mesh_mapper_config fingerprint
        parts.append(self._get_mesh_mapper_fingerprint())

        # device fingerprint
        device_id = self.device.id() if hasattr(self.device, "id") else "single"
        parts.append(f"device_{device_id}")

        # pad_value (only if non-default)
        if self.pad_value != 0.0:
            parts.append(f"pad_{self.pad_value}")

        return "_".join(parts)

    def _get_mesh_mapper_fingerprint(self) -> str:
        """
        Generate a fingerprint for the mesh_mapper_config.

        Uses the config's __repr__ for deterministic hashing since
        MeshMapperConfig exposes a proper string representation via pybind.
        """
        if self.mesh_mapper_config is None:
            return "replicated"

        # Use repr of config for deterministic fingerprint
        config_repr = repr(self.mesh_mapper_config)
        # Hash to keep fingerprint short but unique
        config_hash = hashlib.md5(config_repr.encode()).hexdigest()[:12]
        return f"mapper_{config_hash}"

    @property
    def padded_shape(self) -> tuple[int, ...]:
        """
        Returns the shape after padding for tile alignment.
        Requires device and mesh_mapper_config to be set.

        Note: source.shape remains the canonical unpadded shape.
        """
        assert self.is_resolved(), "LazyWeight must be resolved to compute padded_shape"

        if self.device is None:
            raise ValueError("device must be set to compute padded_shape")

        shape = list(self.source.shape)
        if self.mesh_mapper_config is None:
            return tuple(shape)  # replicated, no padding

        num_devices = self.device.get_num_devices()
        if num_devices == 1:
            return tuple(shape)

        shard_dims = parse_shard_dims_from_mesh_mapper_config(self.mesh_mapper_config)
        for shard_dim in shard_dims:
            if shard_dim < 0:
                shard_dim = len(shape) + shard_dim
            shape[shard_dim] = get_padded_hidden_dim(shape[shard_dim], num_devices)

        return tuple(shape)


def _auto_pad_for_sharding(
    tensor: "torch.Tensor",
    padded_shape: tuple[int, ...],
    pad_value: float = 0.0,
) -> "torch.Tensor":
    """
    Auto-pad tensor to satisfy ttnn.from_torch's tile alignment constraint for sharding.

    ttnn.from_torch requires physical shard shapes to be tile-aligned. This function
    pads the global tensor to the pre-computed padded_shape.

    Args:
        tensor: Source torch tensor
        padded_shape: Target shape after padding (from LazyWeight.padded_shape)
        pad_value: Value to use for padding (default 0.0)

    Returns:
        Padded tensor if padding was needed, otherwise original tensor
    """
    if tensor.shape != padded_shape:
        logger.debug(f"Auto-padding from {tuple(tensor.shape)} to {padded_shape} for tile alignment")
        return pad_to_shape(tensor, padded_shape, pad_value=pad_value)
    return tensor


def _from_torch_and_dump(
    tensor: "torch.Tensor",
    device: Optional[ttnn.MeshDevice],
    dtype: Optional[ttnn.DataType],
    layout: Optional[ttnn.Layout],
    memory_config: Optional[ttnn.MemoryConfig],
    mesh_mapper: Optional[ttnn.CppTensorToMesh],
    is_replicated: bool,
    cache_file_name: Optional[str],
):
    """
    Convert a torch tensor to TTNN format and optionally cache it.

    Args:
        tensor: Source torch tensor
        device: Target MeshDevice
        dtype: Target data type
        layout: Target layout
        memory_config: Target memory config
        mesh_mapper: The mesh mapper to use for distribution
        is_replicated: If True, cache the unsharded tensor for portability
        cache_file_name: Path to cache file, or None to skip caching
    """
    local_mesh_mapper = None if is_replicated else mesh_mapper
    if cache_file_name is None:
        return ttnn.from_torch(
            tensor,
            dtype=dtype,
            layout=layout,
            mesh_mapper=local_mesh_mapper,
            memory_config=memory_config,
            device=device,
        )

    tensor = ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        # For fully replicated tensors, cache unsharded tensor so it can be loaded on any device.
        mesh_mapper=local_mesh_mapper,
        memory_config=memory_config,
        device=None,
    )
    assert tensor.storage_type() == ttnn.StorageType.HOST, "tensor should be on host"

    assert cache_file_name is not None, "cache_file_name must be provided for caching"
    logger.debug(f"\033[33m[cache miss]\033[0m Generating cache for {cache_file_name}")
    pathlib.Path(cache_file_name).parent.mkdir(parents=True, exist_ok=True)
    ttnn._ttnn.tensor.dump_tensor_flatbuffer(str(cache_file_name), tensor)

    if device is not None:
        tensor = tensor.to(device, memory_config)
    return tensor


def resolve_lazy_weight(weight: LazyWeight, **kwargs) -> LazyWeight:
    """Resolve the None fields of `weight` with the given kwargs; do not override non-None fields"""
    to_set = {k: v for k, v in kwargs.items() if getattr(weight, k, None) is None}
    return replace(weight, **to_set)
