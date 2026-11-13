# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A tensor's mesh layout: its placement per mesh axis, and the ttnn mapper/composer that move it on/off the mesh."""

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


def shard_dim(tensor: ttml.autograd.Tensor, axis: int) -> int | None:
    """The tensor dim ``tensor`` is sharded along across mesh axis ``axis``, or None where it is replicated.

    ``Sharding.from_tensor(tensor).shard_dim(axis)`` without building the whole layout, for checks run on every forward.
    """
    value = tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE)
    placement = ttml.core.distributed.placements_per_axis(value)[axis]
    return placement.dim % len(value.shape) if isinstance(placement, ttnn.PlacementShard) else None


class Sharding:
    """A tensor's mesh layout, read off its label when made."""

    def __init__(self, value: ttnn.Tensor) -> None:
        topology = value.tensor_topology()
        self._label_placements = list(topology.placements())
        self._label_shape = list(topology.distribution_shape())
        self._local_shape = list(value.shape)
        self._mesh_shape = []
        self._placements = ()
        self._per_axis_error = None
        if value.storage_type() != ttnn.StorageType.DEVICE or not value.is_allocated():
            # Its label is read against the mesh it is on, so there is nothing to read.
            self._per_axis_error = (
                "cannot read the placement per mesh axis of a tensor that is not allocated on a device mesh (a host "
                "tensor, or a deallocated one)"
            )
            return
        self._mesh_shape = list(_mesh_device().shape)
        try:
            self._placements = tuple(ttml.core.distributed.placements_per_axis(value))
        except RuntimeError as e:
            self._per_axis_error = _fatal_message(e)

    @classmethod
    def from_tensor(cls, tensor: ttml.autograd.Tensor) -> Sharding:
        """The layout of ``tensor``."""
        return cls(tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE))

    @property
    def placements(self) -> tuple:
        """The ttnn placement (``PlacementShard`` / ``PlacementReplicate``) on each mesh axis."""
        if self._per_axis_error is not None:
            raise RuntimeError(self._per_axis_error)
        return self._placements

    def shard_dim(self, axis: int) -> int | None:
        """The tensor dim sharded across mesh axis ``axis``, or None where the tensor is replicated."""
        placement = self.placements[axis]
        return placement.dim % len(self._local_shape) if isinstance(placement, ttnn.PlacementShard) else None

    @property
    def is_fully_replicated(self) -> bool:
        """True if no mesh axis shards this tensor (single device, or replicated on every axis)."""
        return all(self.shard_dim(axis) is None for axis in range(len(self.placements)))

    @property
    def global_shape(self) -> list[int]:
        """The tensor's shape across the mesh."""
        shape = list(self._local_shape)
        for axis in range(len(self.placements)):
            dim = self.shard_dim(axis)
            if dim is not None:
                shape[dim] *= self._mesh_shape[axis]
        return shape

    def describe(self) -> str:
        """The label as a short string for messages, e.g. ``[Replicate, Shard(3)] over mesh [1, 2]``."""
        names = [
            f"Shard({p.dim})" if isinstance(p, ttnn.PlacementShard) else "Replicate" for p in self._label_placements
        ]
        return f"[{', '.join(names)}] over mesh {list(self._label_shape)}"

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

    def gather(self, tensor: ttml.autograd.Tensor, verify_replicas: bool = False):
        """Full host array for ``tensor`` in its native dtype, gathered into a single copy.

        On a single device the tensor is one host buffer, read directly. Otherwise every shard is read to the host
        once and a composer keyed on the tensor's label rebuilds the array from that copy (cf.
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
        shard_dims = {p.dim % rank for p in self._label_placements if isinstance(p, ttnn.PlacementShard)}
        free_dims = [d for d in range(rank) if d not in shard_dims]
        dims: list[int] = []
        shape_override: list[int] = []
        for axis, p in enumerate(self._label_placements):
            if isinstance(p, ttnn.PlacementShard):
                dims.append(p.dim)
                shape_override.append(self._label_shape[axis])
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
