# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Per-device tensor state: what each device holds, and where it came from."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .ir import Dist, Graph, Mesh, TensorSymbol
from .region import RegionSet


@dataclass
class SymbolState:
    """Materialisation of one tensor symbol across the whole mesh."""

    symbol: str
    dist: Dist
    regions: Dict[int, RegionSet]  # device -> logical region owned
    value_id: str
    producer: Optional[str] = None  # node id, or None for graph inputs
    tainted: bool = False  # produced (transitively) by an op with unknown semantics
    #: ttnn calls with no shim/analyzer semantics that this value flowed through.
    #: Non-empty means the metadata here is *assumed*, not derived, so findings
    #: that depend on it are withheld rather than reported (see :mod:`.rules`).
    unregistered: Tuple[str, ...] = ()

    def region(self, device: int) -> RegionSet:
        return self.regions[device]

    def copy(self) -> "SymbolState":
        return SymbolState(
            symbol=self.symbol,
            dist=self.dist,
            regions=dict(self.regions),
            value_id=self.value_id,
            producer=self.producer,
            tainted=self.tainted,
            unregistered=self.unregistered,
        )


@dataclass
class Diagnostic:
    """Something the analyzer could not prove, or that looks wrong in the graph."""

    node: str
    code: str
    message: str

    def __str__(self) -> str:
        return "[%s] %s: %s" % (self.code, self.node, self.message)


@dataclass
class Snapshot:
    """Forward state captured immediately before / after a node."""

    node: str
    before: Dict[str, SymbolState]
    after: Dict[str, SymbolState]


@dataclass
class ForwardResult:
    graph: Graph
    final: Dict[str, SymbolState]
    snapshots: Dict[str, Snapshot] = field(default_factory=dict)
    diagnostics: List[Diagnostic] = field(default_factory=list)

    def before(self, node_id: str) -> Dict[str, SymbolState]:
        return self.snapshots[node_id].before

    def after(self, node_id: str) -> Dict[str, SymbolState]:
        return self.snapshots[node_id].after


DemandMap = Dict[str, Dict[int, RegionSet]]


@dataclass
class DemandResult:
    graph: Graph
    demand: DemandMap  # symbol -> device -> region needed downstream
    diagnostics: List[Diagnostic] = field(default_factory=list)

    def of(self, symbol: str, device: int, ndim: int) -> RegionSet:
        return self.demand.get(symbol, {}).get(device, RegionSet.empty(ndim))


def initial_state(graph: Graph) -> Dict[str, SymbolState]:
    """Materialise graph inputs from their declared placements."""
    state: Dict[str, SymbolState] = {}
    for sid, placement in graph.placements.items():
        sym = graph.symbol(sid)
        dist = placement.dist.normalized(sym.ndim)
        regions = {}
        for dev in graph.mesh.devices():
            regions[dev] = (
                placement.region if placement.region is not None else device_region(graph.mesh, sym, dist, dev)
            )
        state[sid] = SymbolState(
            symbol=sid,
            dist=dist,
            regions=regions,
            value_id=sym.value_id or sid,
            producer=None,
        )
    return state


def device_region(mesh: Mesh, sym: TensorSymbol, dist: Dist, device: int) -> RegionSet:
    """The region implied by ``dist`` for ``device`` (intersection of shards)."""
    region = sym.full_region()
    for mesh_axis, tensor_axis in enumerate(dist.shard):
        if tensor_axis is None:
            continue
        idx = mesh.index_in_group(device, mesh_axis)
        count = mesh.size(mesh_axis)
        region = region.intersect(RegionSet.shard(sym.shape, tensor_axis, idx, count))
    return region


def shard_index_owning(mesh: Mesh, sym: TensorSymbol, dist: Dist, device: int) -> Tuple[int, ...]:
    return tuple(mesh.index_in_group(device, m) for m in range(len(dist.shard)))
