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

from .mesh_runtime import get_runtime
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
    - Layout(ndim=2)  # all Replicate()
    - Layout(ndim=2, axis_placements={1: Shard(-1)})  # Replicate() on axis 0, Shard(-1) on axis 1
    """

    __slots__ = ("placements",)

    def __init__(
        self,
        ndim: int = None,
        axis_placements: dict = None,
    ):
        """Create a Layout.

        Args:
            ndim: Number of mesh dimensions (creates all Replicate() by default)
            axis_placements: Dict mapping mesh_axis -> Placement for sharded axes
        """
        # New API: ndim with optional axis_placements
        if ndim is None:
            runtime = get_runtime()
            if runtime is not None:
                ndim = len(runtime.mesh_shape)
            else:
                raise ValueError("ndim is required if no runtime is set")

        p_list = [Replicate() for _ in range(ndim)]
        if axis_placements:
            for axis, placement in axis_placements.items():
                p_list[axis] = placement
        object.__setattr__(self, "placements", tuple(p_list))

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
        ndim = len(self.placements)
        axis_placements = {}
        for i in range(ndim):
            p = placement if i == mesh_axis else self.placements[i]
            if isinstance(p, Shard):
                axis_placements[i] = p
        return Layout(ndim=ndim, axis_placements=axis_placements)

    def build_mapper(self, mesh_device, tensor_rank: int = 4):
        """Build a TensorToMesh mapper for this layout.

        Returns a shard mapper for the first ``Shard`` placement found,
        or a replicate mapper if fully replicated.
        """
        import ttml

        for mesh_axis, placement in enumerate(self.placements):
            if isinstance(placement, Shard):
                dim = (
                    placement.dim if placement.dim >= 0 else tensor_rank + placement.dim
                )
                return ttml.core.distributed.shard_tensor_to_mesh_mapper(
                    mesh_device, dim, mesh_axis
                )
        return ttml.core.distributed.replicate_tensor_to_mesh_mapper(mesh_device)

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


def layout_from_topology(topology, mesh_ndim: int) -> Layout:
    """Extract a Layout from a ttnn TensorTopology object."""
    placements: List[Placement] = []
    for p in topology.placements():
        if isinstance(p, ttnn.PlacementShard):
            placements.append(Shard(dim=p.dim))
        else:
            placements.append(Replicate())
    axis_placements = {
        i: placements[i]
        for i in range(len(placements))
        if isinstance(placements[i], Shard)
    }
    return Layout(ndim=mesh_ndim, axis_placements=axis_placements)


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
    mesh_ndim = device.shape.dims()

    layout = layout_from_topology(topology, mesh_ndim)

    # Extend to mesh dimensions if needed
    if layout.ndim < mesh_ndim:
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
