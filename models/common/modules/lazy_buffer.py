# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LazyBuffer: Lazy device buffer allocation for mutable state tensors.

This module has NO torch dependency - it accepts any tensor-like object that
ttnn.from_torch() can handle (duck typing with string type hints).

Design principles:
- Mirrors LazyWeight's allocation contract (source + from_torch() parameters)
- Designed for buffers that are mutated in-place after allocation, NOT immutable model weights
- No disk caching: mutable buffers would corrupt state across instances
- No fingerprinting: without caching, there is no cache to invalidate
- Explicit parameters over hidden closures (IDE-friendly)
- Duck typing for source tensors (no torch import)

See also: LazyWeight in models/common/modules/lazy_weight.py
"""

from dataclasses import dataclass, field, replace
from typing import Optional

import ttnn


@dataclass
class LazyBuffer:
    """
    Lazy-allocated device buffer for mutable state tensors.

    Mirrors LazyWeight's allocation contract (source + from_torch() parameters) but is
    designed for buffers that are mutated in-place after allocation, NOT immutable model weights.

    Key differences from LazyWeight:
    - No disk caching: The device data is overwritten in-place via output_tensor= during
      decode (e.g., penalty masks, token counts). Caching a mutable buffer would cause
      state corruption if loaded by another instance.
    - No fingerprinting: Without caching, there is no cache to invalidate.
    - _value caching is safe: The ttnn.Tensor *handle* returned by get_device_buffer()
      never changes — only the on-device data changes via output_tensor= writes.
      So "allocate once, return same handle" is correct for mutable buffers.

    The only thing that makes these different from weights is post-allocation mutability
    and no disk caching. If a buffer becomes read-only in a future refactor, it can be
    promoted to a LazyWeight with caching enabled.

    See also: LazyWeight in models/common/modules/lazy_weight.py

    Example usage:
        # Fully specified at construction
        buf = LazyBuffer(
            source=torch.zeros(32, 128256, dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_tensor = buf.get_device_buffer()  # allocates on first call

        # Refresh device data without reallocation
        buf.update(torch.ones(32, 128256, dtype=torch.int32))

        # Partial construction — resolve later
        buf = LazyBuffer(source=torch.zeros(32, 1))
        buf = resolve_lazy_buffer(buf, device=mesh_device, dtype=ttnn.int32)
    """

    # Source: initial host tensor values (e.g., torch.zeros, torch.ones).
    # Duck-typed — string annotation avoids torch import at module level.
    source: "torch.Tensor"

    # from_torch() parameters — same fields as LazyWeight (minus cache_dir_weight_name, pad_value).
    # Unlike LazyWeight, mesh_mapper stores a pre-built mapper (e.g., ShardTensor2dMesh)
    # rather than a MeshMapperConfig, because LazyBuffer has no caching/fingerprinting.
    dtype: Optional[ttnn.DataType] = ttnn.int32
    layout: Optional[ttnn.Layout] = ttnn.TILE_LAYOUT
    device: Optional[ttnn.MeshDevice] = None
    mesh_mapper: object = None  # Pre-built mapper (ShardTensor2dMesh, etc.) or None for replicate
    memory_config: Optional[ttnn.MemoryConfig] = None

    # Cached device tensor handle (allocated once, device data mutated in-place)
    _value: Optional[ttnn.Tensor] = field(default=None, repr=False)

    def _get_mesh_mapper(self):
        """Get mesh mapper for from_torch(). Shared by get_device_buffer() and update()."""
        if self.mesh_mapper is not None:
            return self.mesh_mapper
        return ttnn.replicate_tensor_to_mesh_mapper(self.device)

    def _from_torch_args(self, *, device):
        """
        Build the full from_torch() kwargs. Used by both get_device_buffer() and update()
        to ensure the same dtype/layout/mesh_mapper/memory_config are used consistently.
        Only ``device`` differs: real device for allocation, None for host-side update.
        """
        return dict(
            dtype=self.dtype,
            layout=self.layout,
            device=device,
            mesh_mapper=self._get_mesh_mapper(),
            memory_config=self.memory_config,
        )

    def get_device_buffer(self) -> ttnn.Tensor:
        """Allocate on first call, return cached handle thereafter."""
        if self._value is not None:
            return self._value

        if self.device is None:
            raise ValueError("device must be set before materializing buffer")
        if self.layout is None:
            raise ValueError("layout must be set before materializing buffer")

        self._value = ttnn.from_torch(
            self.source,
            **self._from_torch_args(device=self.device),
        )
        return self._value

    def update(self, new_source: "torch.Tensor") -> None:
        """
        Overwrite the device buffer contents with a new source tensor, without reallocating.

        If the buffer has not yet been materialized (get_device_buffer not called), this
        simply replaces self.source for future materialization.

        If the buffer IS already materialized, this performs an in-place device update
        using the SAME from_torch() args as the original allocation (dtype, layout,
        mesh_mapper, memory_config) but with device=None to create a host tensor::

            host_tt = ttnn.from_torch(new_source, **same_args, device=None)
            ttnn.copy_host_to_device_tensor(host_tt, self._value)

        The ttnn.Tensor handle (self._value) is preserved — no DRAM reallocation.

        This encapsulates the pattern seen in:
        - TTPenalties._copy_host_to_device (tt_penalties.py:157-159)
        - SeedManager.get_new_values (generator.py:382-383)
        """
        self.source = new_source
        if self._value is not None:
            host_tt = ttnn.from_torch(
                new_source,
                **self._from_torch_args(device=None),
            )
            ttnn.copy_host_to_device_tensor(host_tt, self._value)

    def is_resolved(self) -> bool:
        """Check if all required fields for materialization are set."""
        return self.device is not None and self.dtype is not None and self.layout is not None


def resolve_lazy_buffer(buf: LazyBuffer, **kwargs) -> LazyBuffer:
    """Resolve None fields of ``buf`` with the given kwargs; do not override non-None fields."""
    to_set = {k: v for k, v in kwargs.items() if getattr(buf, k, None) is None}
    return replace(buf, **to_set)
