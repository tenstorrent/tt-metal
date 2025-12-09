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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

from loguru import logger  # todo)) logger dependency should be inserted by the user instead!

import ttnn


# todo)) clean up the code to make the interface tighter -- not so much None
# todo)) needs unit tests to provide coverage
@dataclass
class LazyWeight:
    """
    Lazy weight loading with explicit TTNN conversion parameters.

    The source can be:
    - A tensor directly (duck-typed, typically torch.Tensor)
    - A callable that returns a tensor (for truly lazy loading)

    Example usage:
        # With mesh_mapper (multi-device)
        weight = LazyWeight(
            source=torch_tensor,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
        )

        # With single device (mesh_mapper=None)
        weight = LazyWeight(
            source=torch_tensor,
            device=single_device,
            dtype=ttnn.bfloat16,
        )

        # Callable (fully lazy)
        weight = LazyWeight(
            source=lambda: load_checkpoint()["weight"],
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            cache_dir=Path("/tmp/cache"),
            weight_name="my_weight",
        )

        # Access triggers conversion (or cache load)
        ttnn_tensor = weight.get_weight()
    """

    # Source: duck-typed tensor OR callable that returns one
    # String annotation documents intent without importing torch
    source: Union["torch.Tensor", Callable[[], "torch.Tensor"]]

    # TTNN conversion parameters (explicit, not hidden in closure)
    dtype: ttnn.DataType

    # Device placement - device is always required, mesh_mapper is optional for multi-device
    device: ttnn.Device = None  # Required
    mesh_mapper: Optional[ttnn.TensorToMesh] = None  # For multi-device sharding

    layout: Optional[ttnn.Layout] = None  # None → TILE_LAYOUT
    memory_config: Optional[ttnn.MemoryConfig] = None  # None → DRAM_MEMORY_CONFIG
    pad_value: float = 0.0

    # Cache configuration
    cache_dir: Optional[Path] = None
    weight_name: str = "weight"

    # Private - stores the converted tensor
    _value: Optional[ttnn.Tensor] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate that device is provided."""
        if self.device is None:
            raise ValueError("device must be provided")

    def get_weight(self) -> ttnn.Tensor:
        """
        Get the TTNN tensor, converting from source if needed.

        On first call:
        1. Check cache → load if exists
        2. Otherwise: resolve source → convert → cache

        Subsequent calls return the cached tensor.
        """
        if self._value is not None:
            return self._value

        # Try loading from cache first (skip torch entirely if possible)
        cache_path = self._get_cache_path()
        if cache_path and cache_path.exists():
            logger.info(f"Loading tensor from cache: {cache_path}")
            self._value = ttnn.load_tensor(str(cache_path), device=self.device)
            return self._value

        # Resolve source (call if callable, otherwise use directly)
        logger.info(f"Converting tensor '{self.weight_name}' to TTNN")
        tensor = self.source() if callable(self.source) else self.source

        # Get effective layout and memory_config
        effective_layout = self.layout if self.layout is not None else ttnn.TILE_LAYOUT
        effective_memory_config = self.memory_config if self.memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

        # Use ttnn.as_tensor - handles device placement and caching automatically
        self._value = ttnn.as_tensor(
            tensor,
            dtype=self.dtype,
            layout=effective_layout,
            device=self.device,
            mesh_mapper=self.mesh_mapper,
            memory_config=effective_memory_config,
            cache_file_name=cache_path,  # cache_path=None skips caching
        )

        return self._value

    def _get_cache_path(self) -> Optional[Path]:
        """Generate the cache file path based on configuration fingerprint."""
        if self.cache_dir is None:
            return None
        fingerprint = self._get_fingerprint()
        # todo)) this value seems to be just 1?
        return self.cache_dir / f"{self.weight_name}_{fingerprint}.tensorbin"

    def _get_fingerprint(self) -> str:
        """
        Generate a unique fingerprint from the conversion configuration.

        This ensures cache invalidation when config changes.
        """
        parts = []

        # dtype
        parts.append(f"dtype_{self.dtype.name}")

        # layout
        effective_layout = self.layout if self.layout is not None else ttnn.TILE_LAYOUT
        parts.append(f"layout_{effective_layout.name}")

        # memory_config (use hash if non-default)
        if self.memory_config is not None:
            parts.append(f"mem_{hash(self.memory_config)}")

        # mesh_mapper or device fingerprint
        if self.mesh_mapper is not None:
            parts.append(self._get_mapper_fingerprint())
        elif self.device is not None:
            # Single device mode - use device id
            # todo)) using device.id() is not good enough; what if we want to load the same weight on multiple devices each of which is the same except for id? --> data parallelism
            device_id = self.device.id() if hasattr(self.device, "id") else "single"
            parts.append(f"device_{device_id}")

        # pad_value (only if non-default)
        if self.pad_value != 0.0:
            parts.append(f"pad_{self.pad_value}")

        return "_".join(parts)

    def _get_mapper_fingerprint(self) -> str:
        """
        Generate a fingerprint for the mesh_mapper based on its semantic type.

        Since mesh_mapper doesn't have __hash__, we fingerprint by:
        - Type name
        - Key attributes (dim, dims, mesh_shape) if available
        """
        mapper = self.mesh_mapper
        mapper_type = type(mapper).__name__

        # Try to get more specific info based on mapper type
        if hasattr(mapper, "dim"):
            # ShardTensorToMesh has .dim
            return f"{mapper_type}_dim{mapper.dim}"
        if hasattr(mapper, "dims") and hasattr(mapper, "mesh_shape"):
            # ShardTensor2dMesh has .dims and .mesh_shape
            return f"{mapper_type}_dims{mapper.dims}_shape{mapper.mesh_shape}"

        # todo)) use better fallback
        # Fallback: just use type name (e.g., ReplicateTensorToMesh)
        return mapper_type

    def is_cached(self) -> bool:
        """Check if this weight has a valid cache file."""
        cache_path = self._get_cache_path()
        return cache_path is not None and cache_path.exists()

    def clear_cache(self) -> bool:
        """Remove the cache file if it exists. Returns True if file was removed."""
        cache_path = self._get_cache_path()
        if cache_path and cache_path.exists():
            cache_path.unlink()
            logger.info(f"Removed cache file: {cache_path}")
            return True
        return False
