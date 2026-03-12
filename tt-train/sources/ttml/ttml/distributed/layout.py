# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Layout primitives: Shard, Replicate, and Layout.

A Layout describes how a tensor is distributed across the mesh dimensions.
It maps 1-to-1 with ttnn TensorTopology placements but is a lightweight
Python object that supports hashing and equality for use as cache keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import ttnn


# ---------------------------------------------------------------------------
# Placement types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shard:
    """Tensor is sharded along *dim* across the corresponding mesh axis."""

    dim: int

    def __repr__(self) -> str:
        return f"Shard({self.dim})"


@dataclass(frozen=True)
class Replicate:
    """Tensor is fully replicated on the corresponding mesh axis."""

    def __repr__(self) -> str:
        return "Replicate()"


Placement = Union[Shard, Replicate]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """Describes how a tensor is placed across an N-dimensional mesh.

    ``placements`` is a tuple with one entry per mesh dimension.  Each entry
    is either ``Shard(dim)`` or ``Replicate()``.
    """

    placements: Tuple[Placement, ...]

    def __post_init__(self):
        if not isinstance(self.placements, tuple):
            object.__setattr__(self, "placements", tuple(self.placements))

    @property
    def ndim(self) -> int:
        return len(self.placements)

    def is_replicated(self) -> bool:
        return all(isinstance(p, Replicate) for p in self.placements)

    def is_sharded_on(self, mesh_axis: int) -> bool:
        if mesh_axis >= len(self.placements):
            return False
        return isinstance(self.placements[mesh_axis], Shard)

    def shard_dim(self, mesh_axis: int) -> Optional[int]:
        p = self.placements[mesh_axis]
        if isinstance(p, Shard):
            return p.dim
        return None

    def with_placement(self, mesh_axis: int, placement: Placement) -> "Layout":
        lst = list(self.placements)
        lst[mesh_axis] = placement
        return Layout(placements=tuple(lst))


def replicated_layout(ndim: int = 1) -> Layout:
    return Layout(placements=tuple(Replicate() for _ in range(ndim)))


# ---------------------------------------------------------------------------
# Conversion to/from TensorTopology
# ---------------------------------------------------------------------------


def layout_from_topology(topology) -> Layout:
    """Extract a Layout from a ttnn TensorTopology object."""
    placements: List[Placement] = []
    for p in topology.placements():
        if isinstance(p, ttnn.PlacementShard):
            placements.append(Shard(dim=p.dim))
        else:
            placements.append(Replicate())
    return Layout(placements=tuple(placements))


def layout_to_mapper_config(layout: Layout) -> ttnn.MeshMapperConfig:
    """Convert a Layout to a ttnn MeshMapperConfig."""
    ttnn_placements = []
    for p in layout.placements:
        if isinstance(p, Shard):
            ttnn_placements.append(ttnn.PlacementShard(p.dim))
        else:
            ttnn_placements.append(ttnn.PlacementReplicate())
    return ttnn.MeshMapperConfig(placements=ttnn_placements)


# ---------------------------------------------------------------------------
# Attaching / reading layout metadata on ttml autograd tensors
# ---------------------------------------------------------------------------

_LAYOUT_ATTR = "_distributed_layout"


def get_layout(tensor) -> Optional[Layout]:
    """Read the Layout attached to a ttml autograd tensor.

    Falls back to reading the underlying ttnn TensorTopology if no
    Python-side layout has been stamped.
    """

    topology = tensor.get_value().tensor_topology()
    return layout_from_topology(topology)


def set_layout(tensor, layout: Layout) -> None:
    """Stamp a Layout on a ttml autograd tensor.

    Updates both the Python-side attribute and the underlying ttnn tensor's
    TensorTopology placements.
    """
    import ttml

    ttnn_placements = []
    for p in layout.placements:
        if isinstance(p, Shard):
            ttnn_placements.append(ttnn.PlacementShard(p.dim))
        else:
            ttnn_placements.append(ttnn.PlacementReplicate())

    ttml.core.distributed.set_tensor_placements(tensor, ttnn_placements)
