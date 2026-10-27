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


class ReplicaMismatchError(ValueError):
    """A tensor's label does not match its shards: devices the label calls replicas of each other hold different data,
    or the label lists a device that holds no shard, or leaves out one that does.

    ``Sharding.gather`` keeps one copy per ``Replicate`` mesh axis and composes the shards the label lists, so such a
    label would silently drop or misplace data. The check is ``ttnn.verify_shards_match_topology``, the one the
    flatbuffer writer runs before ``dump_tensor`` writes one copy per replica group; its message says which mismatch it
    found."""


def _fatal_message(error: BaseException) -> str:
    """The message of a ttnn ``TT_FATAL`` (the text under its ``info:`` line), or the error's first line."""
    text = str(error)
    if "info:\n" in text:
        text = text.split("info:\n", 1)[1]
    text = text.split("\nbacktrace:", 1)[0].strip()
    return text.splitlines()[0] if text else str(error).splitlines()[0]


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

    def gather(self, tensor: ttml.autograd.Tensor, verify_replicas: bool = False):
        """Full host array for ``tensor`` in its native dtype, gathered into a single copy.

        On a single device the tensor is one host buffer, read directly. Otherwise every shard is read to the host
        once and a composer keyed on the tensor's own topology rebuilds the array from that copy (cf.
        ``auto_compose._compose_nd_sharded``): a Shard axis concatenates its shards along the sharded dim, and a
        Replicate axis contributes one copy.

        With ``verify_replicas`` (off by default; ``save_checkpoint`` turns it on),
        ``ttnn.verify_shards_match_topology`` checks the host copy first: the label must cover the shards both ways, and
        the shards it calls replicas of each other must hold the same values (their tile padding may differ), else
        ``ReplicaMismatchError`` is raised, i.e. the label does not describe the data and composing by it would drop or
        misplace some. The check compares shards already on the host, so it moves no data; replicas whose bytes match
        cost one memcmp, and it allocates only to decode replicas whose padding differs in a layout it cannot walk in
        place, such as block floats."""
        value = tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE)
        dtype = value.dtype
        if self._is_single_device():
            return tensor.to_numpy(dtype, precision=ttml.autograd.PreferredPrecision.NATIVE)
        host = value.cpu()  # a host tensor comes back as is
        if verify_replicas:
            try:
                ttnn.verify_shards_match_topology(host)
            except RuntimeError as e:
                raise ReplicaMismatchError(
                    f"labelled {self.describe()}, which does not match its shards ({_fatal_message(e)})"
                ) from e
        rank = len(value.shape)
        shard_dims = {p.dim % rank for p in self._placements if isinstance(p, ttnn.PlacementShard)}
        free_dims = [d for d in range(rank) if d not in shard_dims]
        dims: list[int] = []
        shape_override: list[int] = []
        for axis, p in enumerate(self._placements):
            if isinstance(p, ttnn.PlacementShard):
                dims.append(p.dim)
                shape_override.append(self._dist_shape[axis])
            else:
                # Replicate: the composer needs a distinct dim per mesh axis, so take one no Shard axis uses; the
                # size-1 override keeps one copy of that axis.
                dims.append(free_dims.pop(0) if free_dims else 0)
                shape_override.append(1)
        config = ttnn.MeshComposerConfig(dims=dims, mesh_shape_override=ttnn.MeshShape(shape_override))
        composer = ttnn.create_mesh_composer(_mesh_device(), config)
        # The composer decodes each shard it keeps straight from the host copy into the array.
        return ttml.autograd.create_tensor(host, False).to_numpy(
            dtype, composer=composer, precision=ttml.autograd.PreferredPrecision.NATIVE
        )
