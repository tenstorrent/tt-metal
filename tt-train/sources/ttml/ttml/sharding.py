# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A tensor's mesh layout: its placement on each mesh axis, and the ttnn mapper/composer that move it on/off the mesh."""

from __future__ import annotations

from math import prod

import ttnn
import ttml


def _mesh_device():
    """The MeshDevice backing the current context (for shard composers/mappers)."""
    return ttml.autograd.AutoContext.get_instance().get_device()


class Sharding:
    """A tensor's mesh layout, read from its live topology.

    ``placements`` has one entry per mesh axis, whatever the tensor's label looks like: ttnn's whole-mesh mappers
    label a tensor with a single placement for the entire mesh, which is expanded here (see
    ``ttml.core.distributed.placements_per_axis``). The label itself, which can describe layouts no per-axis list can
    (a dim split over several axes at once), is kept to move the tensor on and off the mesh exactly as it was laid out.
    """

    def __init__(self, value: ttnn.Tensor) -> None:
        topology = value.tensor_topology()
        self._label_placements = list(topology.placements())
        self._label_shape = list(topology.distribution_shape())
        self._placements = ttml.core.distributed.placements_per_axis(value)
        self._mesh_shape = list(_mesh_device().shape)
        self._local_shape = list(value.shape)

    @classmethod
    def from_tensor(cls, tensor: ttml.autograd.Tensor) -> Sharding:
        """NATIVE returns the stored slot as is; the default precision would typecast and cache a copy."""
        return cls(tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE))

    @property
    def placements(self) -> list:
        """The ttnn placement (``PlacementShard`` / ``PlacementReplicate``) on each mesh axis."""
        return self._placements

    def shard_dim(self, axis: int) -> int | None:
        """The tensor dim sharded across mesh axis ``axis``, or None where the tensor is replicated."""
        placement = self._placements[axis]
        return placement.dim if isinstance(placement, ttnn.PlacementShard) else None

    @property
    def is_fully_replicated(self) -> bool:
        """True if no mesh axis shards this tensor (single device, or replicated on every axis)."""
        return all(self.shard_dim(axis) is None for axis in range(len(self._placements)))

    @property
    def shape(self) -> list[int]:
        """The full tensor's shape: each device's shape, times the number of shards along every sharded dim."""
        shape = list(self._local_shape)
        for axis in range(len(self._placements)):
            dim = self.shard_dim(axis)
            if dim is not None:
                shape[dim] *= self._mesh_shape[axis]
        return shape

    def _is_single_device(self) -> bool:
        """True for a 1-device distribution."""
        return prod(self._label_shape) <= 1

    def derive_mapper(self):
        """``TensorToMesh`` redistributing a host array onto the mesh exactly as the tensor was distributed,
        or None on a single device. The label's placements + ``mesh_shape_override`` mirror the live topology
        (cf. ``distribute_as._map_nd``); replicate axes keep their full size so the host copy fans out."""
        if self._is_single_device():
            return None
        config = ttnn.MeshMapperConfig(
            placements=self._label_placements, mesh_shape_override=ttnn.MeshShape(self._label_shape)
        )
        return ttnn.create_mesh_mapper(_mesh_device(), config)

    def gather(self, tensor: ttml.autograd.Tensor):
        """Full host array for ``tensor`` in its native dtype, gathered into a single copy.

        On a single device the tensor is one host buffer, read directly. Otherwise a composer keyed on the
        tensor's label rebuilds it (cf. ``auto_compose._compose_nd_sharded``): a Shard axis concatenates its shards
        along the sharded dim; a Replicate axis is given ``mesh_shape_override`` size 1 so the composer takes a single
        copy instead of duplicating it."""
        dtype = tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE).dtype
        if self._is_single_device():
            return tensor.to_numpy(dtype, precision=ttml.autograd.PreferredPrecision.NATIVE)
        dims: list[int] = []
        shape_override: list[int] = []
        for axis, p in enumerate(self._label_placements):
            if isinstance(p, ttnn.PlacementShard):
                dims.append(p.dim)
                shape_override.append(self._label_shape[axis])
            else:  # Replicate: size-1 override → one copy, no duplication to slice off
                dims.append(0)
                shape_override.append(1)
        config = ttnn.MeshComposerConfig(dims=dims, mesh_shape_override=ttnn.MeshShape(shape_override))
        composer = ttnn.create_mesh_composer(_mesh_device(), config)
        return tensor.to_numpy(dtype, composer=composer, precision=ttml.autograd.PreferredPrecision.NATIVE)
