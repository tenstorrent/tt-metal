# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Analyzer IR: device mesh, tensor symbols, distributions, nodes, graphs.

Design notes
------------
* **SSA.** Every node output is a fresh symbol, so "was this value invalidated
  between two collectives?" reduces to comparing ``value_id``s instead of
  tracking writes.
* **value_id is the *mathematical* value.** Collectives (all-gather,
  reduce-scatter, all-reduce) change only *where* and *how* a value is
  materialised, never what it is, so they propagate their input's ``value_id``.
  Compute ops mint a new one. This single rule is what lets the redundancy
  checker prove "device already holds this exact data".
* **Dist is materialisation metadata.** ``shard[m]`` says which tensor axis is
  fractured across mesh axis ``m``; ``partial[m]`` says the value is an
  unreduced partial sum over mesh axis ``m``. Regions remain the source of
  truth for coverage (they can be finer than ``shard`` after slices).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .region import RegionSet

# Symbol kinds -----------------------------------------------------------------
ACT = "activation"
PARAM = "param"  # weights / biases: constant across denoise steps

# Bytes per logical element, by analyzer dtype tag (roadmap blocker 13).
#
# Block-float formats are not whole-byte: tt-metal stores an n-bit mantissa per
# element plus one shared 8-bit exponent per 16-element block, so the amortised
# cost carries a +1/16 byte of exponent overhead per element that whole-byte
# accounting drops. bfp8_b = 1 + 1/16 = 1.0625, bfp4_b = 0.5 + 1/16 = 0.5625.
# This is what a collective actually pushes across a link, so it belongs in the
# fabric-cost estimate rather than the logical-region math.
_BFP_EXPONENT_OVERHEAD = 1.0 / 16  # one 8-bit exponent shared per 16 elements
_ELEM_BYTES = {
    "bf16": 2.0,
    "fp16": 2.0,
    "fp32": 4.0,
    "bfp8_b": 1.0 + _BFP_EXPONENT_OVERHEAD,
    "bfp4_b": 0.5 + _BFP_EXPONENT_OVERHEAD,
    # legacy tags kept as aliases so an older graph JSON still costs correctly
    "bf8_b": 1.0 + _BFP_EXPONENT_OVERHEAD,
    "bf4_b": 0.5 + _BFP_EXPONENT_OVERHEAD,
}


def elem_bytes_for(dtype: str) -> float:
    return _ELEM_BYTES.get(dtype, 2.0)


# Node kinds -------------------------------------------------------------------
COMPUTE = "compute"
COMM = "comm"
META = "meta"

#: Op name used for a call the dry run saw but has no semantics for. Its node is
#: complete (inputs, outputs, shapes, source) but its metadata is *assumed*, so
#: any finding whose proof passes through one is withheld rather than downgraded.
UNREGISTERED_OP = "unregistered"

#: Frames under this prefix are model code; the rest of ``models/tt_dit`` is
#: shared library code. A finding is actionable at the model call site, not at the
#: ``layers/linear.py`` line underneath it (roadmap blocker 44).
MODEL_CODE_PREFIX = "models/tt_dit/models/"

#: Shared, model-agnostic building blocks. A frame in one of these is *not* the
#: model call site an engineer navigates to — it is the plumbing the model called.
#: Everything else under ``models/tt_dit`` (models/, encoders/, pipelines/) is model code.
LIBRARY_PREFIXES = ("models/tt_dit/layers/", "models/tt_dit/parallel/", "models/tt_dit/utils/")


def is_model_frame(frame: str) -> bool:
    """True when ``frame`` is model code (the actionable call site), not shared library.

    Covers DiT blocks (``models/``), the text encoder (``encoders/``), and pipelines —
    anything under ``models/tt_dit`` that is not one of :data:`LIBRARY_PREFIXES`. The
    engineer's ask: the ``feedforward.py`` line under a flagged reduce-scatter is plumbing;
    the transformer block that *called* that feedforward is the line to open."""
    return "models/tt_dit/" in frame and not any(p in frame for p in LIBRARY_PREFIXES)


@dataclass(frozen=True)
class Mesh:
    """A 2-D device mesh, matching ``ttnn.MeshDevice.shape``."""

    shape: Tuple[int, int]
    axis_names: Tuple[str, str] = ("axis0", "axis1")
    arch: str = "wormhole_b0"
    topology: str = "Linear"

    @property
    def num_devices(self) -> int:
        return self.shape[0] * self.shape[1]

    def devices(self) -> List[int]:
        return list(range(self.num_devices))

    def coord(self, device: int) -> Tuple[int, int]:
        return (device // self.shape[1], device % self.shape[1])

    def device_id(self, row: int, col: int) -> int:
        return row * self.shape[1] + col

    def size(self, mesh_axis: int) -> int:
        return self.shape[mesh_axis]

    def groups(self, mesh_axis: int) -> List[Tuple[int, ...]]:
        """Communication groups along ``mesh_axis`` (one per orthogonal index)."""
        rows, cols = self.shape
        if mesh_axis == 0:
            return [tuple(self.device_id(r, c) for r in range(rows)) for c in range(cols)]
        return [tuple(self.device_id(r, c) for c in range(cols)) for r in range(rows)]

    def group_of(self, device: int, mesh_axis: int) -> Tuple[int, ...]:
        for g in self.groups(mesh_axis):
            if device in g:
                return g
        raise KeyError(device)

    def index_in_group(self, device: int, mesh_axis: int) -> int:
        return self.coord(device)[mesh_axis]


def source_chain(stack: Sequence[str], loc: Optional[str] = None) -> List[str]:
    """Source lines to show for a node, outermost model frame first.

    The innermost frame is usually shared library code: a duplicate gather
    reported at ``layers/linear.py:250`` is true but not actionable, while the
    ``to_qkv`` call above it is. Return the model frame and everything under it,
    so a report can lead with the former and still name the latter (blocker 44).
    """
    if not stack:
        return [loc] if loc else []
    model = [i for i, frame in enumerate(stack) if is_model_frame(frame)]
    cut = model[0] if model else len(stack) - 1
    return list(reversed(list(stack)[: cut + 1]))


@dataclass(frozen=True)
class Dist:
    """Per-mesh-axis materialisation of a tensor."""

    shard: Tuple[Optional[int], ...]  # tensor axis fractured across this mesh axis, or None
    partial: Tuple[bool, ...]  # value is an unreduced partial sum over this mesh axis

    @staticmethod
    def replicated(mesh: Mesh) -> "Dist":
        n = len(mesh.shape)
        return Dist(tuple([None] * n), tuple([False] * n))

    @staticmethod
    def make(mesh: Mesh, shard: Optional[Dict[int, int]] = None, partial: Sequence[int] = ()) -> "Dist":
        n = len(mesh.shape)
        sh: List[Optional[int]] = [None] * n
        for mesh_axis, tensor_axis in (shard or {}).items():
            sh[mesh_axis] = tensor_axis
        pa = [m in set(partial) for m in range(n)]
        return Dist(tuple(sh), tuple(pa))

    def with_shard(self, mesh_axis: int, tensor_axis: Optional[int]) -> "Dist":
        sh = list(self.shard)
        sh[mesh_axis] = tensor_axis
        return Dist(tuple(sh), self.partial)

    def with_partial(self, mesh_axis: int, value: bool) -> "Dist":
        pa = list(self.partial)
        pa[mesh_axis] = value
        return Dist(self.shard, tuple(pa))

    @property
    def any_partial(self) -> bool:
        return any(self.partial)

    def normalized(self, ndim: int) -> "Dist":
        sh = tuple(None if a is None else a % ndim for a in self.shard)
        return Dist(sh, self.partial)

    def describe(self, mesh: Mesh) -> str:
        bits = []
        for m, name in enumerate(mesh.axis_names[: len(self.shard)]):
            if self.partial[m]:
                bits.append("partial(%s)" % name)
            elif self.shard[m] is None:
                bits.append("replicated(%s)" % name)
            else:
                bits.append("shard(dim%d,%s)" % (self.shard[m], name))
        return ", ".join(bits)


@dataclass
class TensorSymbol:
    id: str
    shape: Tuple[int, ...]
    dtype: str = "bf16"
    kind: str = ACT
    value_id: str = ""
    note: str = ""
    mesh: Optional[str] = None  # which mesh this symbol lives on (None = the graph's primary)
    # Does this value change between denoise steps? Only meaningful on *entries* (a derived
    # symbol's answer is computed from its producers). None means undeclared, which is read as
    # "varying": assuming an undeclared input is constant would invent step-hoisting findings,
    # so the unsafe direction is the one that has to be opted into.
    step_varying: Optional[bool] = None

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def elem_bytes(self) -> float:
        return elem_bytes_for(self.dtype)

    def full_region(self) -> RegionSet:
        return RegionSet.full(self.shape)

    def bytes_of(self, region: RegionSet) -> int:
        return region.volume * self.elem_bytes


@dataclass
class Node:
    id: str
    op: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    mesh_axis: Optional[int] = None
    loc: Optional[str] = None  # "file.py:123" -- the ttnn call site itself
    stack: List[str] = field(default_factory=list)  # a few tt_dit frames, innermost first
    label: Optional[str] = None  # human name, e.g. "attn.to_qkv"
    calls: int = 1  # how many times this node runs per forward (e.g. 38 blocks)
    fused_in: Optional[str] = None  # set when this node models one stage of a fused ttnn op
    mesh: Optional[str] = None  # which mesh this node runs on (None = the graph's primary)

    @property
    def display(self) -> str:
        return self.label or self.id

    @property
    def attribution(self) -> List[str]:
        return source_chain(self.stack, self.loc)

    @property
    def call_site(self) -> Optional[str]:
        """The one line to name this node by (the model frame if there is one)."""
        chain = self.attribution
        return chain[0] if chain else None

    def group(self, mesh: Mesh) -> Optional[Tuple[int, ...]]:
        if self.mesh_axis is None:
            return None
        explicit = self.attrs.get("group")
        if explicit:
            return tuple(explicit)
        return mesh.groups(self.mesh_axis)[0] if mesh.size(self.mesh_axis) else None


@dataclass
class Placement:
    """Initial materialisation of a graph input."""

    dist: Dist
    region: Optional[RegionSet] = None  # defaults to the shard implied by dist


@dataclass
class Graph:
    name: str
    mesh: Mesh
    symbols: Dict[str, TensorSymbol] = field(default_factory=dict)
    nodes: List[Node] = field(default_factory=list)
    placements: Dict[str, Placement] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    # Number of times the whole graph runs back-to-back with unchanged params
    # (denoise steps). Used only to scale param-gather savings.
    steps: int = 1
    meta: Dict[str, Any] = field(default_factory=dict)
    # How the graph's shapes were produced, which decides how far a finding can be
    # trusted (see report.render_trust). One of:
    #   "dry-run"      -- computed by the metadata-only ttnn shim; "the shim believes",
    #                     not verified against real ttnn until on-device conformance.
    #   "hand-written" -- transcribed from the model source by a human (examples/, builder).
    #   "captured"     -- lifted from a real device trace (ground-truth shapes).
    #   "unknown"      -- provenance not recorded; treat as unverified.
    provenance: str = "unknown"
    # Additional meshes beyond the primary ``mesh``, by id (roadmap blocker 22). A
    # whole-pipeline graph spans submeshes -- CFG / encoder / DiT / VAE on separate
    # ``MeshDevice``s -- so a node or symbol can name a non-primary mesh; a single
    # block leaves this empty and everything is on ``mesh``.
    meshes: Dict[str, "Mesh"] = field(default_factory=dict)

    def mesh_for(self, mesh_id: Optional[str]) -> Mesh:
        """Resolve a mesh id to a Mesh; ``None`` (or unknown) is the primary mesh."""
        return self.meshes.get(mesh_id, self.mesh) if mesh_id else self.mesh

    def mesh_of(self, node: Node) -> Mesh:
        return self.mesh_for(node.mesh)

    def mesh_of_symbol(self, sid: str) -> Mesh:
        return self.mesh_for(self.symbols[sid].mesh)

    def symbol(self, sid: str) -> TensorSymbol:
        return self.symbols[sid]

    def node(self, nid: str) -> Node:
        for n in self.nodes:
            if n.id == nid:
                return n
        raise KeyError(nid)

    def producer_of(self, sid: str) -> Optional[Node]:
        for n in self.nodes:
            if sid in n.outputs:
                return n
        return None

    def consumers_of(self, sid: str) -> List[Node]:
        return [n for n in self.nodes if sid in n.inputs]

    def segments(self) -> List[List["Node"]]:
        """Node lists split at readback boundaries (roadmap blocker 43).

        A ``host_read`` marked ``boundary`` ends a device segment: everything
        after it runs on host data (the scheduler/guidance code between forwards),
        or is the next stage of a pipeline. A single block has no readback, so it
        is one segment; a whole-pipeline graph is a sequence of them. The
        foundation for linking encoder → DiT → VAE stages (phase 10).
        """
        segs: List[List[Node]] = []
        cur: List[Node] = []
        for n in self.nodes:
            cur.append(n)
            if n.op in ("host_read", "stage_boundary") and n.attrs.get("boundary"):
                segs.append(cur)
                cur = []
        if cur:
            segs.append(cur)
        return segs

    # -- serialization --------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "name": self.name,
            "steps": self.steps,
            "meta": self.meta,
            "provenance": self.provenance,
            "mesh": _mesh_to_dict(self.mesh),
            "meshes": {mid: _mesh_to_dict(m) for mid, m in self.meshes.items()},
            "symbols": [
                {
                    "id": s.id,
                    "shape": list(s.shape),
                    "dtype": s.dtype,
                    "kind": s.kind,
                    "value_id": s.value_id,
                    "note": s.note,
                    **({"mesh": s.mesh} if s.mesh else {}),
                }
                for s in self.symbols.values()
            ],
            "placements": {
                sid: {"shard": list(p.dist.shard), "partial": list(p.dist.partial)}
                for sid, p in self.placements.items()
            },
            "nodes": [
                {
                    "id": n.id,
                    "op": n.op,
                    "inputs": n.inputs,
                    "outputs": n.outputs,
                    "attrs": n.attrs,
                    "mesh_axis": n.mesh_axis,
                    "loc": n.loc,
                    "stack": n.stack,
                    "label": n.label,
                    "calls": n.calls,
                    "fused_in": n.fused_in,
                    **({"mesh": n.mesh} if n.mesh else {}),
                }
                for n in self.nodes
            ],
            "outputs": self.outputs,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Graph":
        mesh = _mesh_from_dict(d["mesh"])
        g = Graph(
            name=d.get("name", "graph"),
            mesh=mesh,
            steps=d.get("steps", 1),
            meta=d.get("meta", {}),
            provenance=d.get("provenance", "unknown"),
            meshes={mid: _mesh_from_dict(md) for mid, md in d.get("meshes", {}).items()},
        )
        for s in d["symbols"]:
            g.symbols[s["id"]] = TensorSymbol(
                id=s["id"],
                shape=tuple(s["shape"]),
                dtype=s.get("dtype", "bf16"),
                kind=s.get("kind", ACT),
                value_id=s.get("value_id") or s["id"],
                note=s.get("note", ""),
                mesh=s.get("mesh"),
            )
        for sid, p in d.get("placements", {}).items():
            dist = Dist(
                tuple(None if a is None else int(a) for a in p["shard"]),
                tuple(bool(b) for b in p["partial"]),
            )
            g.placements[sid] = Placement(dist=dist)
        for n in d["nodes"]:
            g.nodes.append(
                Node(
                    id=n["id"],
                    op=n["op"],
                    inputs=list(n.get("inputs", [])),
                    outputs=list(n.get("outputs", [])),
                    attrs=dict(n.get("attrs", {})),
                    mesh_axis=n.get("mesh_axis"),
                    loc=n.get("loc"),
                    stack=list(n.get("stack", [])),
                    label=n.get("label"),
                    calls=int(n.get("calls", 1)),
                    fused_in=n.get("fused_in"),
                    mesh=n.get("mesh"),
                )
            )
        g.outputs = list(d.get("outputs", []))
        return g

    @staticmethod
    def from_json(text: str) -> "Graph":
        return Graph.from_dict(json.loads(text))


def _mesh_to_dict(m: Mesh) -> Dict[str, Any]:
    return {"shape": list(m.shape), "axis_names": list(m.axis_names), "arch": m.arch, "topology": m.topology}


def _mesh_from_dict(m: Dict[str, Any]) -> Mesh:
    return Mesh(
        shape=tuple(m["shape"]),
        axis_names=tuple(m.get("axis_names", ("axis0", "axis1"))),
        arch=m.get("arch", "wormhole_b0"),
        topology=m.get("topology", "Linear"),
    )


def derive_value_id(op: str, input_value_ids: Sequence[str], attrs: Optional[Dict[str, Any]] = None) -> str:
    """Structural identity of a computed value (used for equivalence proofs)."""
    payload = json.dumps(
        {"op": op, "ins": list(input_value_ids), "attrs": _stable(attrs or {})},
        sort_keys=True,
        default=str,
    )
    return "v" + hashlib.sha1(payload.encode()).hexdigest()[:10]


def _stable(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop attrs that affect performance but not the mathematical value."""
    ignore = {"group", "num_links", "topology", "loc", "buffer", "persistent", "chunks", "block_size"}
    return {k: v for k, v in sorted(attrs.items()) if k not in ignore}
