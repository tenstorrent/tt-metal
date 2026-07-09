# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A tensor's mesh layout: the ttnn composer/mapper for moving a (possibly sharded) tensor on/off the mesh."""

from __future__ import annotations

from math import prod

import ttnn
import ttml


def _mesh_device():
    """The MeshDevice backing the current context (for shard composers/mappers)."""
    return ttml.autograd.AutoContext.get_instance().get_device()


class Sharding:
    """A tensor's mesh layout (placements + distribution shape), read from its live topology."""

    def __init__(self, placements: list | None, dist_shape: list[int] | None) -> None:
        self._placements = placements
        self._dist_shape = dist_shape

    @classmethod
    def from_tensor(cls, tensor: ttml.autograd.Tensor) -> Sharding:
        try:
            # NATIVE: read topology without coercing precision (avoids a float32/bf16 typecast + cache).
            topology = tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE).tensor_topology()
            placements = list(topology.placements())
            dist_shape = list(topology.distribution_shape())
        except Exception:
            placements, dist_shape = None, None  # no topology (unit mesh / older ttnn build)
        return cls(placements, dist_shape)

    @property
    def placements(self) -> list | None:
        """Per-mesh-axis ttnn placements (``PlacementShard`` / ``PlacementReplicate``), or None on a unit mesh."""
        return self._placements

    @property
    def dist_shape(self) -> list[int] | None:
        """Distribution shape: the mesh extent the tensor is laid out over per axis, or None on a unit mesh."""
        return self._dist_shape

    @property
    def is_fully_replicated(self) -> bool:
        """True if no mesh axis shards this tensor (single device, or replicated on every axis)."""
        return self._placements is None or not any(isinstance(p, ttnn.PlacementShard) for p in self._placements)

    def _is_single_device(self) -> bool:
        """True when the tensor isn't really distributed (no topology, or a 1-device distribution) → one
        host buffer, readable/placeable without a composer/mapper."""
        return self._dist_shape is None or prod(self._dist_shape) <= 1

    def derive_mapper(self):
        """``TensorToMesh`` redistributing a host array onto the mesh exactly as the tensor was distributed,
        or None on a single device. Placements + ``mesh_shape_override`` mirror the live topology
        (cf. ``distribute_as._map_nd``); replicate axes keep their full size so the host copy fans out."""
        if self._is_single_device():
            return None
        config = ttnn.MeshMapperConfig(
            placements=self._placements, mesh_shape_override=ttnn.MeshShape(self._dist_shape)
        )
        return ttnn.create_mesh_mapper(_mesh_device(), config)

    def gather(self, tensor: ttml.autograd.Tensor):
        """Full host array for ``tensor`` in its native dtype, gathered into a single copy.

        On a single device the tensor is one host buffer, read directly. Otherwise a composer keyed on the
        tensor's own topology rebuilds it (cf. ``auto_compose._compose_nd_sharded``): a Shard axis
        concatenates its shards along the sharded dim; a Replicate axis is given ``mesh_shape_override``
        size 1 so the composer takes a single copy instead of duplicating it."""
        dtype = tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE).dtype
        if self._is_single_device():
            return tensor.to_numpy(dtype, precision=ttml.autograd.PreferredPrecision.NATIVE)
        dims: list[int] = []
        shape_override: list[int] = []
        for axis, p in enumerate(self._placements):
            if isinstance(p, ttnn.PlacementShard):
                dims.append(p.dim)
                shape_override.append(self._dist_shape[axis])
            else:  # Replicate: size-1 override → one copy, no duplication to slice off
                dims.append(0)
                shape_override.append(1)
        config = ttnn.MeshComposerConfig(dims=dims, mesh_shape_override=ttnn.MeshShape(shape_override))
        composer = ttnn.create_mesh_composer(_mesh_device(), config)
        return tensor.to_numpy(dtype, composer=composer, precision=ttml.autograd.PreferredPrecision.NATIVE)
