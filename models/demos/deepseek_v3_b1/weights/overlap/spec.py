# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""OverlappedTensorSpec — describes one sub-tensor within a fused raw-byte buffer."""

from dataclasses import dataclass

import ttnn

_DTYPE_ELEMENT_BYTES = {
    ttnn.bfloat16: 2,
    ttnn.float32: 4,
    ttnn.uint16: 2,
    ttnn.uint32: 4,
    ttnn.int32: 4,
}


@dataclass(frozen=True)
class OverlappedTensorSpec:
    """Describes one sub-tensor within a fused raw-byte buffer.

    Multiple ``OverlappedTensorSpec`` instances share a single L1
    buffer whose per-core shard is zero-padded to a common maximum
    byte size.  Each spec carries the core range, logical tensor shape,
    dtype, and tile dimensions needed to pack its portion of the shard.

    Tile byte sizes for BFP formats are computed from ``tile_h`` and
    ``tile_w`` rather than stored as constants, so non-standard tile
    shapes (e.g. 1x32, 16x32) are handled automatically.

    Shape tuples follow (height, width) convention.

    ``sharding`` controls how cores partition the per-device tensor:

    - ``WIDTH_SHARDED``: each core gets a column slice (full height,
      partial width).  ``num_cores`` divides the width.
    - ``HEIGHT_SHARDED``: each core gets a row slice (partial height,
      full width).  ``num_cores`` divides the height.

    When data is preprocessed (shuffled / block-sharded) before
    overlapping, ``raw_tensor_shape`` reflects the physical layout
    while ``logical_tensor_shape`` preserves the original per-device
    shape for downstream consumers.

    ``name`` is an optional logical identifier used when the spec is
    embedded in a :class:`RegionSpec` for cache fingerprinting.
    """

    core_range_set: ttnn.CoreRangeSet
    raw_tensor_shape: tuple[int, int]
    dtype: ttnn.DataType
    sharding: ttnn.TensorMemoryLayout = ttnn.TensorMemoryLayout.WIDTH_SHARDED

    tile_h: int = 32
    tile_w: int = 32

    tp_dim: tuple[int | None, int | None] = (None, None)

    logical_tensor_shape: tuple[int, int] | None = None

    overlap_priority: int | None = None

    name: str = ""

    def _tile_bytes(self) -> int:
        num_elements = self.tile_h * self.tile_w
        if self.dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
            _L1_ALIGNMENT = 16
            num_exponents = num_elements // 16
            exponent_bytes = (num_exponents + _L1_ALIGNMENT - 1) // _L1_ALIGNMENT * _L1_ALIGNMENT
            mantissa_bytes = num_elements if self.dtype == ttnn.bfloat8_b else num_elements // 2
            return exponent_bytes + mantissa_bytes
        return num_elements * _DTYPE_ELEMENT_BYTES[self.dtype]

    def _dim_tp(self, tensor_dim: int, mesh_shape: tuple[int, int]) -> int:
        """TP factor for a single tensor dimension (0=height, 1=width)."""
        result = 1
        for mesh_dim, d in enumerate(self.tp_dim):
            if d == tensor_dim:
                result *= mesh_shape[mesh_dim]
        return result

    def _dim_slice_idx(self, tensor_dim: int, row: int, col: int, mesh_shape: tuple[int, int]) -> int:
        """Slice index for a single tensor dimension at mesh coordinate (row, col)."""
        idx = 0
        stride = 1
        mesh_coord = (row, col)
        for mesh_dim in reversed(range(2)):
            if self.tp_dim[mesh_dim] == tensor_dim:
                idx += mesh_coord[mesh_dim] * stride
                stride *= mesh_shape[mesh_dim]
        return idx

    def tp(self, mesh_shape: tuple[int, int]) -> int:
        """Total TP factor (product of all non-None mesh dimensions)."""
        return self._dim_tp(0, mesh_shape) * self._dim_tp(1, mesh_shape)

    def per_device_height(self, mesh_shape: tuple[int, int]) -> int:
        return self.raw_tensor_shape[0] // self._dim_tp(0, mesh_shape)

    def per_device_width(self, mesh_shape: tuple[int, int]) -> int:
        return self.raw_tensor_shape[1] // self._dim_tp(1, mesh_shape)

    def shard_shape(self, mesh_shape: tuple[int, int]) -> tuple[int, int]:
        pdh = self.per_device_height(mesh_shape)
        pdw = self.per_device_width(mesh_shape)
        num_cores = self.core_range_set.num_cores()
        if self.sharding == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            return (pdh, pdw // num_cores)
        else:
            return (pdh // num_cores, pdw)

    def tiles_per_shard(self, mesh_shape: tuple[int, int]) -> int:
        sh, sw = self.shard_shape(mesh_shape)
        return (sh // self.tile_h) * (sw // self.tile_w)

    def shard_bytes(self, mesh_shape: tuple[int, int]) -> int:
        return self.tiles_per_shard(mesh_shape) * self._tile_bytes()


def _core_list(crs: ttnn.CoreRangeSet) -> list[tuple[int, int]]:
    """Ordered list of (x, y) core coordinates from a CoreRangeSet."""
    cores = []
    for cr in crs.ranges():
        for y in range(cr.start.y, cr.end.y + 1):
            for x in range(cr.start.x, cr.end.x + 1):
                cores.append((x, y))
    return cores


def _greedy_place(specs: list[OverlappedTensorSpec], mesh_shape: tuple[int, int]) -> list[int]:
    """Assign byte offsets to specs via First-Fit-Decreasing placement.

    Sorts by ``overlap_priority`` descending (``None`` treated as 0),
    then by ``shard_bytes`` descending.  For each spec, finds the
    earliest byte offset that avoids overlapping with already-placed
    specs on any shared core.

    Returns a list of offsets in the *original* spec order.
    """
    n = len(specs)
    core_sets = [set(_core_list(s.core_range_set)) for s in specs]
    sizes = [s.shard_bytes(mesh_shape) for s in specs]

    order = sorted(
        range(n),
        key=lambda i: (-(specs[i].overlap_priority or 0), -sizes[i]),
    )

    offsets = [0] * n
    for idx in order:
        blocked: list[tuple[int, int]] = []
        for prev in order:
            if prev == idx:
                break
            if core_sets[idx] & core_sets[prev]:
                blocked.append((offsets[prev], offsets[prev] + sizes[prev]))
        blocked.sort()
        offset = 0
        for start, end in blocked:
            if offset + sizes[idx] <= start:
                break
            offset = max(offset, end)
        offsets[idx] = offset
    return offsets


def max_shard_bytes(shard_specs: list[OverlappedTensorSpec], mesh_shape: tuple[int, int]) -> int:
    placements = _greedy_place(shard_specs, mesh_shape)
    return max(off + spec.shard_bytes(mesh_shape) for off, spec in zip(placements, shard_specs))
