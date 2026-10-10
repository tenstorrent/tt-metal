# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A tensor's mesh layout: the ttnn composer/mapper for moving a (possibly sharded) tensor on/off the mesh."""

from __future__ import annotations

import warnings
from math import prod

import numpy as np
import ttnn
import ttml


def _mesh_device():
    """The MeshDevice backing the current context (for shard composers/mappers)."""
    return ttml.autograd.AutoContext.get_instance().get_device()


class ReplicaMismatchError(ValueError):
    """A tensor's label says two devices hold copies of the same data, and they do not.

    ``Sharding.gather`` keeps one copy per ``Replicate`` mesh axis, so a ``Replicate`` label over data that differs
    across that axis would silently drop every copy but the first (the C++ flatbuffer writer refuses the same case)."""


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

    def describe(self) -> str:
        """The layout as a short string for messages, e.g. ``[Replicate, Shard(3)] over mesh [1, 2]``."""
        if self._placements is None:
            return "no topology (single device)"
        names = [f"Shard({p.dim})" if isinstance(p, ttnn.PlacementShard) else "Replicate" for p in self._placements]
        return f"[{', '.join(names)}] over mesh {list(self._dist_shape)}"

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

    def gather(self, tensor: ttml.autograd.Tensor, verify_replicas: bool = True):
        """Full host array for ``tensor`` in its native dtype, gathered into a single copy.

        On a single device the tensor is one host buffer, read directly. Otherwise a composer keyed on the
        tensor's own topology rebuilds it (cf. ``auto_compose._compose_nd_sharded``): a Shard axis
        concatenates its shards along the sharded dim, and a Replicate axis contributes one copy.

        ``verify_replicas`` (default): every copy along each Replicate axis is read and byte-compared before one is
        kept, and ``ReplicaMismatchError`` is raised if they differ, i.e. the label does not describe the data. This
        holds the whole replicated extent on the host for a moment (one copy per replica). With ``False``, only the
        first copy is read, trusting the label."""
        dtype = tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE).dtype
        if self._is_single_device():
            return tensor.to_numpy(dtype, precision=ttml.autograd.PreferredPrecision.NATIVE)
        rank = len(tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE).shape)
        shard_dims = {p.dim % rank for p in self._placements if isinstance(p, ttnn.PlacementShard)}
        free_dims = [d for d in range(rank) if d not in shard_dims]
        dims: list[int] = []
        shape_override: list[int] = []
        replica_axes: list[tuple[int, int, int]] = []  # (mesh axis, tensor dim its copies are stacked on, copies)
        for axis, p in enumerate(self._placements):
            if isinstance(p, ttnn.PlacementShard):
                dims.append(p.dim)
                shape_override.append(self._dist_shape[axis])
                continue
            # Replicate: the composer needs a distinct dim per mesh axis, so take one no Shard axis uses.
            dims.append(free_dims.pop(0) if free_dims else 0)
            copies = self._dist_shape[axis]
            if verify_replicas and copies > 1 and dims[-1] not in shard_dims and dims[-1] not in dims[:-1]:
                # Stack this axis's copies along that dim so they can be split apart and compared.
                shape_override.append(copies)
                replica_axes.append((axis, dims[-1], copies))
            else:  # size-1 override -> one copy, no duplication to slice off
                if verify_replicas and copies > 1:
                    warnings.warn(
                        f"Sharding.gather: no free tensor dim to compare the copies on mesh axis {axis} of a "
                        f"rank-{rank} tensor laid out {self.describe()}; keeping the first copy unchecked",
                        stacklevel=2,
                    )
                shape_override.append(1)
        config = ttnn.MeshComposerConfig(dims=dims, mesh_shape_override=ttnn.MeshShape(shape_override))
        composer = ttnn.create_mesh_composer(_mesh_device(), config)
        data = tensor.to_numpy(dtype, composer=composer, precision=ttml.autograd.PreferredPrecision.NATIVE)
        for axis, dim, copies in replica_axes:
            parts = np.split(data, copies, axis=dim)
            first = np.ascontiguousarray(parts[0])
            for index, part in enumerate(parts[1:], start=1):
                if np.ascontiguousarray(part).tobytes() != first.tobytes():
                    raise ReplicaMismatchError(
                        f"labelled {self.describe()}, but copy {index} along mesh axis {axis} differs from copy 0, "
                        "so the data is not replicated there; saving one copy would drop the others"
                    )
            data = first
        return data
