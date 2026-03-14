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


class Layout:
    """Describes how a tensor is placed across an N-dimensional mesh.

    ``placements`` is a tuple with one entry per mesh dimension.  Each entry
    is either ``Shard(dim)`` or ``Replicate()``.

    Can be created in several ways:
    - Layout(placements=(Shard(0), Replicate()))  # explicit tuple
    - Layout(ndim=2)  # all Replicate()
    - Layout(ndim=2, axis_placements={1: Shard(-1)})  # Replicate() on axis 0, Shard(-1) on axis 1
    """

    __slots__ = ("placements",)

    def __init__(
        self,
        placements: Tuple[Placement, ...] = None,
        *,
        ndim: int = None,
        axis_placements: dict = None,
    ):
        """Create a Layout.

        Args:
            placements: Explicit tuple of placements (legacy API)
            ndim: Number of mesh dimensions (creates all Replicate() by default)
            axis_placements: Dict mapping mesh_axis -> Placement for non-replicated axes
        """
        if placements is not None:
            # Legacy API: explicit placements tuple
            if not isinstance(placements, tuple):
                placements = tuple(placements)
            object.__setattr__(self, "placements", placements)
        elif ndim is not None:
            # New API: ndim with optional axis_placements
            p_list = [Replicate() for _ in range(ndim)]
            if axis_placements:
                for axis, placement in axis_placements.items():
                    p_list[axis] = placement
            object.__setattr__(self, "placements", tuple(p_list))
        else:
            # Empty layout
            object.__setattr__(self, "placements", ())

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

    def __eq__(self, other):
        if not isinstance(other, Layout):
            return False
        return self.placements == other.placements

    def __hash__(self):
        return hash(self.placements)

    def __repr__(self):
        return f"Layout({self.placements})"


def replicated_layout(ndim: int = 2) -> Layout:
    return Layout(ndim=ndim)


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

    The layout dimensions always match the mesh device dimensions.
    Falls back to reading the underlying ttnn TensorTopology.
    """
    ttnn_tensor = tensor.get_value()
    topology = ttnn_tensor.tensor_topology()

    # Get mesh dimensions from the tensor's device
    # device.shape is a property (MeshShape), not a method
    device = ttnn_tensor.device()
    mesh_ndim = None
    if hasattr(device, "shape"):
        mesh_shape = device.shape  # property, not callable
        mesh_ndim = (
            mesh_shape.dims() if hasattr(mesh_shape, "dims") else len(mesh_shape)
        )

    layout = layout_from_topology(topology)

    # Extend to mesh dimensions if needed
    if mesh_ndim is not None and layout.ndim < mesh_ndim:
        axis_placements = {i: p for i, p in enumerate(layout.placements)}
        layout = Layout(ndim=mesh_ndim, axis_placements=axis_placements)

    return layout


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
