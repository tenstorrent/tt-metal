# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""From a tensor's topology label to the boxes its devices hold.

A ``Label`` is the ttnn ``TensorTopology`` in plain Python: one placement per distribution axis, the distribution
shape, and the physical mesh coordinate of every distribution coordinate in row-major order. Both forms a label can
take are handled directly: an N-D label with one placement per mesh axis, and the collapsed ``{N}, [placement]``
label of the whole-mesh mappers, whose single axis walks the mesh in row-major order.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Sequence

from ._boxes import Box
from ._errors import TopologyError


@dataclass(frozen=True)
class Placement:
    kind: str  # "replicate" | "shard"
    dim: int | None = None

    def __post_init__(self) -> None:
        # TODO(awliu-TT): #62347 - accept a "partial" kind here once tensor placements carry a partial-sum label.
        if self.kind not in ("replicate", "shard"):
            raise TopologyError(f"unknown placement kind {self.kind!r}")
        if (self.kind == "shard") != (self.dim is not None):
            raise TopologyError(f"placement {self.kind} must {'have' if self.kind == 'shard' else 'not have'} a dim")

    @staticmethod
    def replicate() -> "Placement":
        return Placement("replicate")

    @staticmethod
    def shard(dim: int) -> "Placement":
        return Placement("shard", int(dim))

    def to_json(self) -> list:
        return ["replicate"] if self.kind == "replicate" else ["shard", self.dim]

    @staticmethod
    def from_json(value: Sequence) -> "Placement":
        # Through the constructor, so a kind this reader does not know is refused instead of read as a shard.
        return Placement(str(value[0]), int(value[1]) if len(value) > 1 else None)


def dist_coords(dist_shape: Sequence[int]) -> list[tuple[int, ...]]:
    """Every distribution coordinate of ``dist_shape`` in row-major order."""
    return [tuple(c) for c in product(*(range(n) for n in dist_shape))]


@dataclass(frozen=True)
class Label:
    placements: tuple[Placement, ...]
    dist_shape: tuple[int, ...]
    mesh_coords: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "placements", tuple(self.placements))
        object.__setattr__(self, "dist_shape", tuple(int(n) for n in self.dist_shape))
        object.__setattr__(self, "mesh_coords", tuple(tuple(int(x) for x in c) for c in self.mesh_coords))
        if len(self.placements) != len(self.dist_shape):
            raise TopologyError(f"label has {len(self.placements)} placements for a {self.dist_shape} distribution")
        if len(self.mesh_coords) != prod(self.dist_shape):
            raise TopologyError(
                f"label lists {len(self.mesh_coords)} mesh coordinates for a {self.dist_shape} distribution"
            )
        if len(set(self.mesh_coords)) != len(self.mesh_coords):
            raise TopologyError(f"label lists a mesh coordinate twice: {self.mesh_coords}")

    @staticmethod
    def normalized(
        placements: Sequence[Placement], dist_shape: Sequence[int], mesh_coords: Sequence[Sequence[int]], rank: int
    ) -> "Label":
        """A label with negative shard dims resolved against the tensor ``rank``."""
        fixed = []
        for p in placements:
            if p.kind == "shard":
                dim = p.dim + rank if p.dim < 0 else p.dim
                if not 0 <= dim < rank:
                    raise TopologyError(f"shard dim {p.dim} is out of range for a rank-{rank} tensor")
                fixed.append(Placement.shard(dim))
            else:
                fixed.append(p)
        return Label(tuple(fixed), tuple(dist_shape), tuple(tuple(c) for c in mesh_coords))

    def shard_axes(self, dim: int) -> list[int]:
        """The distribution axes that shard ``dim``, outermost first."""
        return [a for a, p in enumerate(self.placements) if p.kind == "shard" and p.dim == dim]

    def shard_count(self, dim: int) -> int:
        return prod(self.dist_shape[a] for a in self.shard_axes(dim))

    def to_json(self) -> dict:
        return {
            "placements": [p.to_json() for p in self.placements],
            "distribution_shape": list(self.dist_shape),
            "mesh_coords": [list(c) for c in self.mesh_coords],
        }

    @staticmethod
    def from_json(value: dict) -> "Label":
        return Label(
            tuple(Placement.from_json(p) for p in value["placements"]),
            tuple(value["distribution_shape"]),
            tuple(tuple(c) for c in value["mesh_coords"]),
        )


def global_shape(label: Label, local_shape: Sequence[int]) -> tuple[int, ...]:
    """The global shape a label describes, from the shape every device holds."""
    return tuple(int(n) * label.shard_count(d) for d, n in enumerate(local_shape))


def tp_split(label: Label, fsdp_axis: int | None, rank: int) -> tuple[int, ...]:
    """Per dim of a rank-``rank`` tensor, how many shards the mesh axes other than ``fsdp_axis`` split it into: the TP
    split, which decides a fused weight's row order. FSDP splits rows that are already in canonical order, so it is
    left out."""
    return tuple(prod(label.dist_shape[a] for a in label.shard_axes(d) if a != fsdp_axis) for d in range(rank))


@dataclass(frozen=True)
class Chunk:
    """One device's box of a tensor's global array."""

    mesh_coord: tuple[int, ...]
    box: Box
    group: tuple[int, ...]  # identical for every replica of the same box


def chunk_boxes(label: Label, local_shape_: Sequence[int]) -> list[Chunk]:
    """The box every distribution coordinate holds, in row-major order of the distribution shape.

    Along each sharded tensor dim, the axes sharding it are read outermost first (row-major hierarchical sharding), so
    a collapsed ``{N}, [Shard(d)]`` label and an N-D ``[Shard(d), Shard(d)]`` label give the same boxes.
    """
    local = tuple(int(n) for n in local_shape_)
    chunks: list[Chunk] = []
    shard_axes = {d: label.shard_axes(d) for d in range(len(local))}
    for index, coord in enumerate(dist_coords(label.dist_shape)):
        offsets = []
        for d, n in enumerate(local):
            linear = 0
            for a in shard_axes[d]:
                linear = linear * label.dist_shape[a] + coord[a]
            offsets.append(linear * n)
        group = tuple(coord[a] for a, p in enumerate(label.placements) if p.kind == "shard")
        chunks.append(Chunk(label.mesh_coords[index], Box(tuple(offsets), local), group))
    return chunks


def replica_groups(chunks: Sequence[Chunk]) -> dict[tuple[int, ...], list[Chunk]]:
    """Chunks that hold the same box, keyed by their shard coordinates, in first-seen order. The key fixes the box,
    since ``chunk_boxes`` derives a chunk's offsets from the same coordinates."""
    groups: dict[tuple[int, ...], list[Chunk]] = {}
    for chunk in chunks:
        groups.setdefault(chunk.group, []).append(chunk)
    return groups


def crc31(text: str) -> int:
    """A stable 31-bit hash of ``text``: zlib's CRC32 without its top bit, the same in every process, and small
    enough for the C int that carries a hash between the ranks."""
    return zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF
