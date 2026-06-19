# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A tensor's mesh layout: the ttnn objects needed to move a (possibly sharded) tensor on/off the mesh.

Shared by checkpoint save/load and by callers that just need to ask "is this tensor sharded?"
(e.g. the trainer's clip-grad-norm guard). The layout is read from a live tensor's placements; nothing
is stored on the side.
"""

from __future__ import annotations

import ttnn
import ttml


def _mesh_device():
    """The MeshDevice backing the current context (for shard composers/mappers)."""
    return ttml.autograd.AutoContext.get_instance().get_device()


class Sharding:
    """A tensor's mesh layout, read from its live placements."""

    def __init__(self, placements: list | None, shape: list[int]) -> None:
        self._placements = placements
        self._shape = shape

    @classmethod
    def from_tensor(cls, tensor: ttml.autograd.Tensor) -> Sharding:
        try:
            placements = list(tensor.get_value().tensor_topology().placements())
        except Exception:
            placements = None  # unit mesh / no topology
        return cls(placements, list(tensor.shape()))

    @property
    def is_fully_replicated(self) -> bool:
        """True if no mesh axis shards this tensor (single device, or replicated on every axis)."""
        return self._placements is None or not any(isinstance(p, ttnn.PlacementShard) for p in self._placements)

    def derive_mapper(self):
        """`TensorToMesh` distributing a full host array onto the mesh, or None on a unit mesh."""
        if self._placements is None:
            return None
        return ttnn.create_mesh_mapper(_mesh_device(), ttnn.MeshMapperConfig(self._placements))

    def gather(self, tensor: ttml.autograd.Tensor):
        """Full host array for `tensor` in its native dtype: gather the shards, then drop the replicate dups.

        The mesh composer concatenates every axis (no replicate concept; see `_compose_dims`), so a
        replicated axis comes back duplicated and must be sliced down to one copy here. That coupling
        is why the composer isn't exposed on its own — its raw output is wrong without this slice.
        """
        dtype = tensor.get_value().dtype
        if self._placements is None:
            return tensor.to_numpy(dtype)
        concat_dims, replicate_dims = self._compose_dims()
        composer = ttnn.create_mesh_composer(_mesh_device(), ttnn.MeshComposerConfig(concat_dims))
        arr = tensor.to_numpy(dtype, composer=composer)
        if replicate_dims:  # keep one canonical copy of each stacked replicate axis
            slicer = [slice(None)] * arr.ndim
            for dim, size in replicate_dims:
                slicer[dim] = slice(0, size)
            arr = arr[tuple(slicer)]
        return arr

    def _compose_dims(self) -> tuple[list[int], list[tuple[int, int]]]:
        """Per-mesh-axis concat dims for the gather composer, plus replicate copies to slice off.

        Returns ``(concat_dims, replicate)``. ``concat_dims[i]`` is the tensor dim the composer
        concatenates mesh axis ``i`` along: a Shard axis uses the dim it sharded (so concatenating
        the shards rebuilds it); a Replicate axis has no such dim, so it borrows a spare (unsharded)
        one — concatenating the identical copies there inflates that dim, so its
        ``(tensor_dim, original_size)`` is recorded in ``replicate`` for the caller to slice back to a
        single copy. Mirrors `ttml.fsdp._shard_replicated_param` in reverse.
        """
        rank = len(self._shape)
        sharded = {p.dim for p in self._placements if isinstance(p, ttnn.PlacementShard)}
        spare = [d for d in range(rank) if d not in sharded]
        dims: list = []
        replicate: list = []  # (tensor_dim, original_size)
        next_spare = 0
        for p in self._placements:
            if isinstance(p, ttnn.PlacementShard):
                dims.append(p.dim)
            else:
                if next_spare >= len(spare):
                    raise RuntimeError("checkpointing: too many replicate mesh axes to assign compose dims")
                dims.append(spare[next_spare])
                replicate.append((spare[next_spare], self._shape[spare[next_spare]]))
                next_spare += 1
        return dims, replicate
