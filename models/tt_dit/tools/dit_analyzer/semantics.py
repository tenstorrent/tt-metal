# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Declarative op-semantics registry (Tier 1 of the plan).

Every op contributes two transfer functions:

``apply``   forward availability -- given the per-device state before the node,
            define the per-device state of its outputs.
``demand``  backward necessity -- given what downstream needs from the node's
            outputs, state what it needs from the node's inputs.

Anything not registered here is handled by :data:`UNKNOWN`, which is
deliberately pessimistic (full regions, fresh value, tainted) so the redundancy
rules downgrade or suppress findings that depend on it. Conservative
correctness over aggressive cleverness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .ir import COMM, COMPUTE, META, UNREGISTERED_OP, Dist, Graph, Mesh, Node, TensorSymbol, derive_value_id
from .region import Box, RegionSet
from .state import Diagnostic, SymbolState


# -----------------------------------------------------------------------------
# contexts
# -----------------------------------------------------------------------------
@dataclass
class ApplyCtx:
    graph: Graph
    node: Node
    state: Dict[str, SymbolState]
    diagnostics: List[Diagnostic] = field(default_factory=list)

    @property
    def mesh(self) -> Mesh:
        return self.graph.mesh_of(self.node)  # the mesh this node runs on (blocker 22)

    def sym(self, sid: str) -> TensorSymbol:
        return self.graph.symbol(sid)

    def in_sym(self, i: int = 0) -> TensorSymbol:
        return self.sym(self.node.inputs[i])

    def out_sym(self, i: int = 0) -> TensorSymbol:
        return self.sym(self.node.outputs[i])

    def in_state(self, i: int = 0) -> SymbolState:
        sid = self.node.inputs[i]
        st = self.state.get(sid)
        if st is None:  # undefined input: be pessimistic but keep going
            sym = self.sym(sid)
            self.warn("UNDEFINED_INPUT", "input %s has no producer or placement" % sid)
            st = SymbolState(
                symbol=sid,
                dist=Dist.replicated(self.mesh),
                regions={d: sym.full_region() for d in self.mesh.devices()},
                value_id=sym.value_id or sid,
                tainted=True,
            )
            self.state[sid] = st
        return st

    def define(
        self,
        i: int,
        dist: Dist,
        regions: Dict[int, RegionSet],
        value_id: str,
        tainted: bool = False,
    ) -> None:
        sid = self.node.outputs[i]
        sym = self.sym(sid)
        self.state[sid] = SymbolState(
            symbol=sid,
            dist=dist.normalized(sym.ndim),
            regions=regions,
            value_id=value_id,
            producer=self.node.id,
            tainted=tainted,
            unregistered=self.unregistered_inputs(),
        )

    def unregistered_inputs(self) -> Tuple[str, ...]:
        """Unregistered calls this node's inputs flowed through.

        Propagated for every op automatically, so a spec author cannot forget it:
        a value derived from assumed metadata carries that fact for as long as it
        is live, and the rules refuse to build a proof on it.
        """
        seen: List[str] = []
        if self.node.op == UNREGISTERED_OP:
            seen.append(str(self.node.attrs.get("call", self.node.op)))
        for sid in self.node.inputs:
            st = self.state.get(sid)
            for call in st.unregistered if st else ():
                if call not in seen:
                    seen.append(call)
        return tuple(seen)

    def warn(self, code: str, message: str) -> None:
        self.diagnostics.append(Diagnostic(node=self.node.id, code=code, message=message))

    def group(self) -> Tuple[int, ...]:
        assert self.node.mesh_axis is not None, self.node
        return self.mesh.groups(self.node.mesh_axis)[0]

    def group_of(self, device: int) -> Tuple[int, ...]:
        assert self.node.mesh_axis is not None, self.node
        return self.mesh.group_of(device, self.node.mesh_axis)

    def tainted_inputs(self) -> bool:
        return any(self.state[s].tainted for s in self.node.inputs if s in self.state)

    def value_of_inputs(self) -> List[str]:
        return [self.state[s].value_id if s in self.state else s for s in self.node.inputs]


@dataclass
class DemandCtx:
    graph: Graph
    node: Node
    before: Dict[str, SymbolState]  # forward state before this node
    out_demand: Dict[str, Dict[int, RegionSet]]  # this node's outputs -> dev -> region
    diagnostics: List[Diagnostic] = field(default_factory=list)
    _acc: Dict[str, Dict[int, RegionSet]] = field(default_factory=dict)

    @property
    def mesh(self) -> Mesh:
        return self.graph.mesh_of(self.node)  # the mesh this node runs on (blocker 22)

    def sym(self, sid: str) -> TensorSymbol:
        return self.graph.symbol(sid)

    def demand_out(self, i: int = 0) -> Dict[int, RegionSet]:
        sid = self.node.outputs[i]
        nd = self.sym(sid).ndim
        d = self.out_demand.get(sid, {})
        return {dev: d.get(dev, RegionSet.empty(nd)) for dev in self.mesh.devices()}

    def need(self, sid: str, device: int, region: RegionSet) -> None:
        if region.is_empty:
            return
        nd = self.sym(sid).ndim
        cur = self._acc.setdefault(sid, {}).get(device, RegionSet.empty(nd))
        self._acc[sid][device] = cur.union(region)

    def need_all_local(self, i: int) -> None:
        """Demand whatever the device already holds of input ``i`` (safe default)."""
        sid = self.node.inputs[i]
        st = self.before.get(sid)
        for dev in self.mesh.devices():
            region = st.regions[dev] if st else self.sym(sid).full_region()
            self.need(sid, dev, region)

    def result(self) -> Dict[str, Dict[int, RegionSet]]:
        return self._acc

    def warn(self, code: str, message: str) -> None:
        self.diagnostics.append(Diagnostic(node=self.node.id, code=code, message=message))


@dataclass
class OpSpec:
    name: str
    kind: str
    apply: Callable[[ApplyCtx], None]
    demand: Callable[[DemandCtx], None]
    preserves_value: bool = False
    is_collective: bool = False
    doc: str = ""


REGISTRY: Dict[str, OpSpec] = {}
# ttnn / tt_dit call names -> canonical spec name.
ALIASES: Dict[str, str] = {}


def register(spec: OpSpec, aliases: Sequence[str] = ()) -> OpSpec:
    REGISTRY[spec.name] = spec
    for a in aliases:
        ALIASES[a] = spec.name
    return spec


def lookup(op: str) -> OpSpec:
    name = ALIASES.get(op, op)
    return REGISTRY.get(name, UNKNOWN)


def is_collective(op: str) -> bool:
    return lookup(op).is_collective


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _axis(a: int, ndim: int) -> int:
    return a % ndim


def _rows_axis(sym: TensorSymbol) -> int:
    return sym.ndim - 2


def _cols_axis(sym: TensorSymbol) -> int:
    return sym.ndim - 1


def _slice_region(region: RegionSet, axis: int, lo: int, hi: int) -> RegionSet:
    """Restrict ``region`` to ``[lo, hi)`` on ``axis``."""
    if region.is_empty:
        return region
    box = Box(tuple((0, 1 << 62) for _ in range(region.ndim))).replace_axis(axis, lo, hi)
    # clamp the untouched axes to each box's own extent via intersection
    return region.intersect(RegionSet.of(box))


def _cross(out_sym: TensorSymbol, rows: Optional[Tuple[int, int]], cols: Optional[Tuple[int, int]]) -> RegionSet:
    if rows is None or cols is None:
        return RegionSet.empty(out_sym.ndim)
    box = Box.full(out_sym.shape)
    box = box.replace_axis(_rows_axis(out_sym), *rows)
    box = box.replace_axis(_cols_axis(out_sym), *cols)
    return RegionSet.of(box)


# -----------------------------------------------------------------------------
# communication ops
# -----------------------------------------------------------------------------
def _all_gather_apply(c: ApplyCtx) -> None:
    x = c.in_state(0)
    xs, ys = c.in_sym(0), c.out_sym(0)
    m = c.node.mesh_axis
    dim = _axis(int(c.node.attrs["dim"]), ys.ndim)

    if xs.shape != ys.shape:
        c.warn("SHAPE_MISMATCH", "all-gather input %s and output %s have different logical shapes" % (xs.id, ys.id))
    if x.dist.partial[m]:
        c.warn(
            "GATHER_OF_PARTIAL",
            "all-gather over %s of a partial-sum value: this concatenates unreduced sums" % c.mesh.axis_names[m],
        )
    if x.dist.shard[m] is None:
        c.warn("GATHER_OF_REPLICATED", "input is already replicated over %s" % c.mesh.axis_names[m])
    elif x.dist.shard[m] != dim:
        c.warn(
            "GATHER_AXIS_MISMATCH",
            "gathering dim %d but input is fractured on dim %s over %s" % (dim, x.dist.shard[m], c.mesh.axis_names[m]),
        )

    regions = {}
    for dev in c.mesh.devices():
        acc = RegionSet.empty(ys.ndim)
        for peer in c.group_of(dev):
            acc = acc.union(x.regions[peer])
        regions[dev] = acc
    c.define(0, x.dist.with_shard(m, None), regions, x.value_id, x.tainted)


def _all_gather_demand(c: DemandCtx) -> None:
    xid = c.node.inputs[0]
    x = c.before.get(xid)
    dem = c.demand_out(0)
    for dev in c.mesh.devices():
        want = dem[dev]
        if want.is_empty:
            continue
        for peer in c.mesh.group_of(dev, c.node.mesh_axis):
            owned = x.regions[peer] if x else c.sym(xid).full_region()
            c.need(xid, peer, want.intersect(owned))


def _reduce_scatter_apply(c: ApplyCtx) -> None:
    x = c.in_state(0)
    ys = c.out_sym(0)
    m = c.node.mesh_axis
    dim = _axis(int(c.node.attrs["dim"]), ys.ndim)
    if not x.dist.partial[m]:
        c.warn(
            "RS_OF_NONPARTIAL",
            "reduce-scatter over %s of a value that is not a partial sum on that axis" % c.mesh.axis_names[m],
        )
    regions = {}
    for dev in c.mesh.devices():
        avail = RegionSet.empty(ys.ndim)
        for peer in c.group_of(dev):
            avail = avail.union(x.regions[peer])
        idx = c.mesh.index_in_group(dev, m)
        regions[dev] = avail.intersect(RegionSet.shard(ys.shape, dim, idx, c.mesh.size(m)))
    c.define(0, x.dist.with_partial(m, False).with_shard(m, dim), regions, x.value_id, x.tainted)


def _reduce_scatter_demand(c: DemandCtx) -> None:
    xid = c.node.inputs[0]
    dem = c.demand_out(0)
    for dev in c.mesh.devices():
        want = dem[dev]
        if want.is_empty:
            continue
        for peer in c.mesh.group_of(dev, c.node.mesh_axis):
            c.need(xid, peer, want)  # every contributor must produce that region


def _all_reduce_apply(c: ApplyCtx) -> None:
    x = c.in_state(0)
    ys = c.out_sym(0)
    m = c.node.mesh_axis
    if not x.dist.partial[m]:
        c.warn("AR_OF_NONPARTIAL", "all-reduce over %s of a non-partial value" % c.mesh.axis_names[m])
    regions = {}
    for dev in c.mesh.devices():
        acc = RegionSet.empty(ys.ndim)
        for peer in c.group_of(dev):
            acc = acc.union(x.regions[peer])
        regions[dev] = acc
    c.define(0, x.dist.with_partial(m, False).with_shard(m, None), regions, x.value_id, x.tainted)


_all_reduce_demand = _reduce_scatter_demand


register(
    OpSpec(
        "all_gather",
        COMM,
        _all_gather_apply,
        _all_gather_demand,
        preserves_value=True,
        is_collective=True,
        doc="Concatenate shards of one tensor axis across a mesh axis; every participant ends with the union.",
    ),
    aliases=(
        "ttnn.experimental.all_gather_async",
        "all_gather_async",
        "all_gather_persistent_buffer",
        "vae_all_gather",
        "ccl.all_gather",
    ),
)
register(
    OpSpec(
        "reduce_scatter",
        COMM,
        _reduce_scatter_apply,
        _reduce_scatter_demand,
        preserves_value=True,
        is_collective=True,
        doc="Sum partial results across a mesh axis, leaving each device one shard of the reduced value.",
    ),
    aliases=(
        "ttnn.experimental.reduce_scatter_minimal_async",
        "reduce_scatter_minimal_async",
        "ccl.reduce_scatter",
    ),
)
register(
    OpSpec(
        "all_reduce",
        COMM,
        _all_reduce_apply,
        _all_reduce_demand,
        preserves_value=True,
        is_collective=True,
        doc="Sum partial results across a mesh axis, leaving the full reduced value on every device.",
    ),
    aliases=("ttnn.experimental.all_reduce_async", "all_reduce_async"),
)


def _mesh_partition_apply(c: ApplyCtx) -> None:
    """Scatter a replicated tensor across a mesh axis: each device keeps one shard.

    The dual of all_gather -- it *drops* the peers' shares rather than summing them
    (no partial-sum handling, unlike reduce_scatter). MiniMax-H3 fractures the
    assembled packed sequence onto SP this way before the block stack.
    """
    x = c.in_state(0)
    ys = c.out_sym(0)
    m = c.node.mesh_axis
    dim = _axis(int(c.node.attrs["dim"]), ys.ndim)
    regions = {}
    for dev in c.mesh.devices():
        idx = c.mesh.index_in_group(dev, m)
        regions[dev] = x.regions[dev].intersect(RegionSet.shard(ys.shape, dim, idx, c.mesh.size(m)))
    c.define(0, x.dist.with_shard(m, dim), regions, x.value_id, x.tainted)


def _mesh_partition_demand(c: DemandCtx) -> None:
    # Pure selection of a local shard: a device needs exactly what it keeps, from
    # its own copy -- no peer contributes (nothing is summed).
    xid = c.node.inputs[0]
    dem = c.demand_out(0)
    for dev in c.mesh.devices():
        if not dem[dev].is_empty:
            c.need(xid, dev, dem[dev])


register(
    OpSpec(
        "mesh_partition",
        COMM,
        _mesh_partition_apply,
        _mesh_partition_demand,
        preserves_value=True,
        doc="Scatter a replicated tensor across a mesh axis; each device keeps one shard of the given dim.",
    ),
    aliases=("ttnn.mesh_partition", "ccl.mesh_partition"),
)


# -----------------------------------------------------------------------------
# matmul family
# -----------------------------------------------------------------------------
def _matmul_apply(c: ApplyCtx) -> None:
    x, w = c.in_state(0), c.in_state(1)
    xs, ws, ys = c.in_sym(0), c.in_sym(1), c.out_sym(0)
    k_axis_x, k_axis_w = _cols_axis(xs), _rows_axis(ws)
    n_axis_w = _cols_axis(ws)

    dist = Dist.replicated(c.mesh)
    for m in range(len(c.mesh.shape)):
        xsh = None if x.dist.shard[m] is None else _axis(x.dist.shard[m], xs.ndim)
        wsh = None if w.dist.shard[m] is None else _axis(w.dist.shard[m], ws.ndim)
        if x.dist.partial[m] or w.dist.partial[m]:
            c.warn(
                "MATMUL_OF_PARTIAL",
                "operand is an unreduced partial sum over %s; matmul result will be wrong unless reduced first"
                % c.mesh.axis_names[m],
            )
            dist = dist.with_partial(m, True)
        elif xsh == k_axis_x and wsh == k_axis_w:
            dist = dist.with_partial(m, True)  # contraction is fractured -> partial sums
        elif wsh == n_axis_w:
            dist = dist.with_shard(m, _cols_axis(ys))
        elif xsh is not None and xsh <= _rows_axis(xs):
            dist = dist.with_shard(m, xsh)
        elif xsh == k_axis_x and wsh is None:
            c.warn(
                "MATMUL_K_MISMATCH",
                "activation is fractured on K over %s but the weight is replicated there; "
                "this matmul needs a gather (or a K-sharded weight)" % c.mesh.axis_names[m],
            )
            dist = dist.with_partial(m, True)

    regions = {}
    k_mismatch = False
    for dev in c.mesh.devices():
        rows = x.regions[dev].bounds(_rows_axis(xs))
        cols = w.regions[dev].bounds(n_axis_w)
        regions[dev] = _cross(ys, rows, cols)
        # requirement check: the local K range of x must be matched by w
        kx = x.regions[dev].bounds(k_axis_x)
        kw = w.regions[dev].bounds(k_axis_w)
        if kx is not None and kw is not None and kx != kw:
            c.warn(
                "K_COVERAGE",
                "device %d holds K%s of the activation but K%s of the weight" % (dev, list(kx), list(kw)),
            )
            k_mismatch = True
    # A K-coverage mismatch means the spec cannot line up the contraction -- e.g. a
    # batched activation×activation attention matmul (q@kᵀ) whose K assumption
    # (K = rows of a 2D weight) does not hold. Taint the result so no finding
    # downstream of it is emitted as provable; better honestly-suspicious than
    # confidently wrong (the phase-7 failure mode).
    c.define(
        0,
        dist,
        regions,
        derive_value_id("matmul", c.value_of_inputs(), c.node.attrs),
        c.tainted_inputs() or k_mismatch,
    )


def _matmul_demand(c: DemandCtx) -> None:
    xid, wid = c.node.inputs[0], c.node.inputs[1]
    xs, ws = c.sym(xid), c.sym(wid)
    x, w = c.before.get(xid), c.before.get(wid)
    dem = c.demand_out(0)
    for dev in c.mesh.devices():
        want = dem[dev]
        if want.is_empty:
            continue
        rows = want.bounds(-2)
        cols = want.bounds(-1)
        # activation: the demanded rows over the full local K extent
        kx = (x.regions[dev].bounds(_cols_axis(xs)) if x else None) or (0, xs.shape[_cols_axis(xs)])
        c.need(xid, dev, _cross(xs, rows, kx))
        # weight: the local K extent over the demanded output columns
        kw = (w.regions[dev].bounds(_rows_axis(ws)) if w else None) or (0, ws.shape[_rows_axis(ws)])
        c.need(wid, dev, _cross(ws, kw, cols))
    for i in range(2, len(c.node.inputs)):  # bias and friends
        c.need_all_local(i)


def _conv2d_apply(c: ApplyCtx) -> None:
    """[B,H,W,Cin] -> [B,Hout,Wout,Cout], channel-parallel like a col-parallel matmul.

    The conv contracts the full input channels (and a spatial neighbourhood), so
    the output's channel axis is fractured wherever the weight's out-channel axis
    (axis 0) is; nothing is a partial sum here (the reduced K axis is not sharded).
    """
    from .state import device_region

    ys = c.out_sym(0)
    dist = Dist.replicated(c.mesh)
    if len(c.node.inputs) > 1:
        w, ws = c.in_state(1), c.in_sym(1)
        for m in range(len(c.mesh.shape)):
            a = w.dist.shard[m]
            if a is not None and _axis(a, ws.ndim) == 0:  # weight out-channel axis
                dist = dist.with_shard(m, _cols_axis(ys))  # -> output channel axis (last)
    regions = {d: device_region(c.mesh, ys, dist, d) for d in c.mesh.devices()}
    c.define(0, dist, regions, derive_value_id("conv2d", c.value_of_inputs(), c.node.attrs), c.tainted_inputs())


def _conv2d_demand(c: DemandCtx) -> None:
    # Contracts the full input channels and a spatial neighbourhood, and reads the
    # weight over its output-channel shard: demand whatever each device holds.
    for i in range(len(c.node.inputs)):
        c.need_all_local(i)


register(
    OpSpec(
        "conv2d",
        COMPUTE,
        _conv2d_apply,
        _conv2d_demand,
        doc="2-D convolution (NHWC), channel-parallel: output channels fractured where the weight's out axis is.",
    ),
    aliases=("ttnn.conv2d",),
)


def _embedding_apply(c: ApplyCtx) -> None:
    """[.., S] x [V, H] -> [.., S, H], hidden-parallel like a col-parallel matmul.

    A data-dependent row gather, so the result is a fresh value on every device;
    its hidden axis is fractured wherever the weight's hidden axis (its last) is.
    """
    from .state import device_region

    ys = c.out_sym(0)
    xs = c.in_sym(0)  # indices
    dist = Dist.replicated(c.mesh)
    # the indices' row-sharding carries onto the output's leading axes (same index)
    x = c.in_state(0)
    for m in range(len(c.mesh.shape)):
        a = x.dist.shard[m]
        if a is not None:
            dist = dist.with_shard(m, _axis(a, xs.ndim))
    if len(c.node.inputs) > 1:
        w, ws = c.in_state(1), c.in_sym(1)
        for m in range(len(c.mesh.shape)):
            a = w.dist.shard[m]
            if a is not None and _axis(a, ws.ndim) == _cols_axis(ws):  # weight hidden axis (last)
                dist = dist.with_shard(m, _cols_axis(ys))  # -> output hidden axis (last)
    regions = {d: device_region(c.mesh, ys, dist, d) for d in c.mesh.devices()}
    c.define(0, dist, regions, derive_value_id("embedding", c.value_of_inputs(), c.node.attrs), c.tainted_inputs())


def _embedding_demand(c: DemandCtx) -> None:
    # A data-dependent gather: any id may select any row, so demand the whole id
    # tensor and whatever weight rows each device holds.
    for i in range(len(c.node.inputs)):
        c.need_all_local(i)


register(
    OpSpec(
        "embedding",
        COMPUTE,
        _embedding_apply,
        _embedding_demand,
        doc="Token-id lookup [.., S] x [V, H] -> [.., S, H]; hidden fractured where the weight's hidden axis is.",
    ),
    aliases=("ttnn.embedding",),
)
register(
    OpSpec(
        "matmul",
        COMPUTE,
        _matmul_apply,
        _matmul_demand,
        doc="[..,M,K] x [K,N]. Fractured K yields partial sums; fractured N yields a column-fractured result.",
    ),
    aliases=(
        "ttnn.experimental.minimal_matmul",
        "minimal_matmul",
        "ttnn.linear",
        "ttnn.matmul",
        "linear",
    ),
)


# -----------------------------------------------------------------------------
# pointwise / normalisation
# -----------------------------------------------------------------------------
def _pointwise_apply(c: ApplyCtx) -> None:
    ys = c.out_sym(0)
    primary = None
    dists = []
    for i, sid in enumerate(c.node.inputs):
        st = c.in_state(i)
        if c.sym(sid).shape == ys.shape:
            dists.append((sid, st))
            if primary is None:
                primary = st
    if primary is None:  # everything broadcasts; fall back to input 0
        primary = c.in_state(0)
        dists = [(c.node.inputs[0], primary)]

    for sid, st in dists[1:]:
        if st.dist != primary.dist:
            c.warn(
                "LAYOUT_MISMATCH",
                "operands %s (%s) and %s (%s) are laid out differently"
                % (dists[0][0], primary.dist.describe(c.mesh), sid, st.dist.describe(c.mesh)),
            )

    regions = {}
    for dev in c.mesh.devices():
        acc = primary.regions[dev]
        for _, st in dists[1:]:
            acc = acc.intersect(st.regions[dev])
        regions[dev] = acc

    full_axes = _full_axes(c.node, ys.ndim)
    if full_axes:
        for dev in c.mesh.devices():
            for a in full_axes:
                b = regions[dev].bounds(a)
                if b is not None and b != (0, ys.shape[a]):
                    c.warn(
                        "PARTIAL_REDUCTION_AXIS",
                        "device %d only holds %s of axis %d, but %s reduces over that axis"
                        % (dev, list(b), a, c.node.op),
                    )
    dist = primary.dist
    if dist.any_partial:
        c.warn("POINTWISE_OF_PARTIAL", "pointwise op consumes an unreduced partial sum")
    c.define(
        0,
        dist,
        regions,
        derive_value_id(c.node.attrs.get("fn", c.node.op), c.value_of_inputs(), c.node.attrs),
        c.tainted_inputs(),
    )


_FULL_LAST_AXIS_OPS = {"layernorm", "softmax"}


def _full_axes(node: Node, ndim: int) -> List[int]:
    """Axes an op must see in full on each device (reduction axes)."""
    if "needs_full_axes" in node.attrs:
        return [_axis(a, ndim) for a in node.attrs["needs_full_axes"]]
    if lookup(node.op).name in _FULL_LAST_AXIS_OPS:
        return [ndim - 1]
    return []


def _pointwise_demand(c: DemandCtx) -> None:
    ys = c.sym(c.node.outputs[0])
    full_axes = _full_axes(c.node, ys.ndim)
    dem = c.demand_out(0)
    for i, sid in enumerate(c.node.inputs):
        s = c.sym(sid)
        for dev in c.mesh.devices():
            want = dem[dev]
            if want.is_empty:
                continue
            if s.shape != ys.shape:  # broadcast operand: demand what the device holds
                st = c.before.get(sid)
                c.need(sid, dev, st.regions[dev] if st else s.full_region())
                continue
            need = want
            for a in full_axes:  # a reduction over `a` needs the whole axis
                need = need.map_axis(a, lambda lo, hi, _a=a: (0, s.shape[_a]))
            c.need(sid, dev, need)


for _name, _aliases, _doc in [
    (
        "pointwise",
        (
            # unary/binary ttnn names are declared once in GENERIC_OPS below (which
            # also drives the shim); only the non-generic pointwise aliases remain.
            "gelu_tanh",
            "swiglu",
            "residual_add",
            "rope",
            "scale_shift",
            "addcmul",
            "ttnn.addcmul",
            "ttnn.experimental.rotary_embedding_llama",
            "rotary_embedding_llama",
            "ttnn.experimental.alt_complex_rotate90",
            "alt_complex_rotate90",
        ),
        "Elementwise / axis-local op; keeps the operand layout.",
    ),
    (
        "layernorm",
        ("norm", "rmsnorm", "layer_norm", "rms_norm"),
        "Reduces over the last axis, so it needs full coverage there (set needs_full_axes).",
    ),
    ("softmax", (), "Reduces over the last axis."),
    (
        # VAE group norm: channels are TP-fractured but each device holds whole
        # groups and the spatial axis is not sharded, so the reduction is
        # device-local -- ordinary layout-preserving pointwise, no full-axis need.
        "group_norm",
        ("ttnn.group_norm",),
        "Group norm over NHWC channels; device-local (whole groups per device), keeps the layout.",
    ),
    (
        # dit_fused_distributed_{layernorm,rmsnorm}: the reduction statistics are
        # exchanged inside the kernel over a mesh axis, so the *activation* stays
        # fractured and no activation-sized collective is implied. The stats
        # exchange is orders of magnitude smaller than the tensor, so it is
        # modelled as free rather than as a collective (avoids report noise).
        "distributed_norm",
        (
            "ttnn.experimental.dit_fused_distributed_layernorm",
            "ttnn.experimental.dit_fused_distributed_rmsnorm",
            "dit_fused_distributed_layernorm",
            "dit_fused_distributed_rmsnorm",
            "distributed_layernorm",
            "distributed_rmsnorm",
        ),
        "Distributed norm: exchanges only reduction statistics, so the activation layout is preserved.",
    ),
]:
    register(OpSpec(_name, COMPUTE, _pointwise_apply, _pointwise_demand, doc=_doc), aliases=_aliases)


# -----------------------------------------------------------------------------
# metadata-only ops
# -----------------------------------------------------------------------------
def _suffix_preserved_map(src: Sequence[int], dst: Sequence[int]) -> Dict[int, int]:
    """Trailing axes whose size is unchanged, ``src axis -> dst axis``.

    A reshape that only merges or splits *leading* axes (the VAE's
    ``[B,H,W,C] <-> [B,1,H*W,C]``) leaves a common suffix intact, so a shard on
    any preserved trailing axis survives it (roadmap blocker 17).
    """
    m: Dict[int, int] = {}
    i, j = len(src) - 1, len(dst) - 1
    while i >= 0 and j >= 0 and src[i] == dst[j]:
        m[i] = j
        i -= 1
        j -= 1
    return m


def _identity_apply(c: ApplyCtx) -> None:
    x = c.in_state(0)
    ys = c.out_sym(0)
    if c.in_sym(0).shape == ys.shape:
        regions = dict(x.regions)
        dist = x.dist
    else:  # trivial rank change (squeeze / unsqueeze of size-1 axes)
        mapping = _trivial_axis_map(c.in_sym(0).shape, ys.shape)
        if mapping is None:
            xs = c.in_sym(0)
            suffix = _suffix_preserved_map(xs.shape, ys.shape)
            if suffix and all(a is None or _axis(a, xs.ndim) in suffix for a in x.dist.shard):
                # Only leading (unsharded) axes are reshaped; every shard sits on a
                # preserved trailing axis, so keep it and set the reshaped prefix to
                # full. The value and the shard's coverage survive, so this is not
                # opaque and not tainted -- the VAE's spatial merge (blocker 17).
                dist = Dist(
                    tuple(None if a is None else suffix[_axis(a, xs.ndim)] for a in x.dist.shard), x.dist.partial
                )
                regions = {}
                for d in c.mesh.devices():
                    if x.regions[d].is_empty:
                        regions[d] = RegionSet.empty(ys.ndim)
                        continue
                    box = Box.full(ys.shape)
                    for sa, da in suffix.items():
                        b = x.regions[d].bounds(sa)
                        if b is not None:
                            box = box.replace_axis(da, *b)
                    regions[d] = RegionSet.of(box)
                c.define(0, dist, regions, x.value_id, x.tainted)
                return
            c.warn("OPAQUE_RESHAPE", "cannot track regions through reshape %s -> %s" % (c.in_sym(0).shape, ys.shape))
            regions = {
                d: (RegionSet.empty(ys.ndim) if x.regions[d].is_empty else ys.full_region()) for d in c.mesh.devices()
            }
            c.define(0, Dist.replicated(c.mesh), regions, x.value_id, True)
            return
        regions = {d: _remap(x.regions[d], mapping, ys.shape) for d in c.mesh.devices()}
        dist = Dist(
            tuple(None if a is None else mapping.get(_axis(a, c.in_sym(0).ndim)) for a in x.dist.shard),
            x.dist.partial,
        )
    c.define(0, dist, regions, x.value_id, x.tainted)


def _identity_demand(c: DemandCtx) -> None:
    xid = c.node.inputs[0]
    xs, ys = c.sym(xid), c.sym(c.node.outputs[0])
    dem = c.demand_out(0)
    if xs.shape == ys.shape:
        for dev in c.mesh.devices():
            c.need(xid, dev, dem[dev])
        return
    mapping = _trivial_axis_map(ys.shape, xs.shape)
    if mapping is None:
        suffix = _suffix_preserved_map(xs.shape, ys.shape)  # src=input, dst=output
        if suffix:  # map the preserved trailing bounds back; leading axes -> full
            inverse = {da: sa for sa, da in suffix.items()}
            for dev in c.mesh.devices():
                if dem[dev].is_empty:
                    continue
                box = Box.full(xs.shape)
                for da, sa in inverse.items():
                    b = dem[dev].bounds(da)
                    if b is not None:
                        box = box.replace_axis(sa, *b)
                c.need(xid, dev, RegionSet.of(box))
            return
    for dev in c.mesh.devices():
        if dem[dev].is_empty:
            continue
        c.need(xid, dev, _remap(dem[dev], mapping, xs.shape) if mapping else xs.full_region())


def _trivial_axis_map(src: Sequence[int], dst: Sequence[int]) -> Optional[Dict[int, int]]:
    """Map src axes to dst axes when they differ only by size-1 axes."""
    s = [(i, d) for i, d in enumerate(src) if d != 1]
    t = [(i, d) for i, d in enumerate(dst) if d != 1]
    if [d for _, d in s] != [d for _, d in t]:
        return None
    mapping = {i: j for (i, _), (j, _) in zip(s, t)}
    for i, d in enumerate(src):  # size-1 axes map to the first spare size-1 dst axis
        if d == 1:
            spare = [j for j, dd in enumerate(dst) if dd == 1 and j not in mapping.values()]
            if spare:
                mapping[i] = spare[0]
    return mapping


def _remap(region: RegionSet, mapping: Dict[int, int], dst_shape: Sequence[int]) -> RegionSet:
    if region.is_empty:
        return RegionSet.empty(len(dst_shape))
    boxes = []
    for b in region.boxes:
        out = Box.full(dst_shape)
        for src_axis, dst_axis in mapping.items():
            if src_axis < b.ndim and dst_axis < len(dst_shape):
                lo, hi = b.ranges[src_axis]
                if dst_shape[dst_axis] != 1:
                    out = out.replace_axis(dst_axis, lo, hi)
        boxes.append(out)
    return RegionSet(len(dst_shape), boxes)


register(
    OpSpec(
        "identity",
        META,
        _identity_apply,
        _identity_demand,
        preserves_value=True,
        doc="View / squeeze / unsqueeze / to_layout / typecast: same logical value, tracked regions.",
    ),
    aliases=(
        # shape-changing views keep bespoke shim rules; the pure passthroughs
        # (to_layout / typecast / clone / to_memory_config) are in GENERIC_OPS.
        "view",
        "reshape",
        "squeeze",
        "unsqueeze",
        "unsqueeze_to_4D",
        "ttnn.reshape",
        "ttnn.squeeze",
        "ttnn.unsqueeze",
    ),
)


# One registration per generic op (phase 8, "one registration per op"). Each entry
# declares -- in one place, in the analyzer layer that is the source of truth -- a
# ttnn call's canonical op AND the shim shape-rule that builds its node, so a
# pointwise / passthrough op is a single line here and the shim generates its
# dispatch from this table (see dryrun/ops.py) instead of restating the same names
# in a second file. Bespoke comm/compute ops (matmul, collectives, conv, fused,
# from_torch, reshape, ...) keep an explicit shim rule -- by design, only genuinely
# new semantics should need real thought.
#   ttnn leaf : (canonical analyzer op, shim rule, pointwise fn label)
GENERIC_OPS: Dict[str, Tuple[str, str, Optional[str]]] = {
    "sigmoid": ("pointwise", "unary", "sigmoid"),
    "gelu": ("pointwise", "unary", "gelu"),
    "silu": ("pointwise", "unary", "silu"),
    "tanh": ("pointwise", "unary", "tanh"),
    "sqrt": ("pointwise", "unary", "sqrt"),
    "reciprocal": ("pointwise", "unary", "reciprocal"),
    "neg": ("pointwise", "unary", "neg"),
    "clamp": ("pointwise", "unary", "clamp"),
    "cos": ("pointwise", "unary", "cos"),
    "sin": ("pointwise", "unary", "sin"),
    "add": ("pointwise", "binary", "add"),
    "sub": ("pointwise", "binary", "sub"),
    "subtract": ("pointwise", "binary", "sub"),  # fn normalised to sub
    "mul": ("pointwise", "binary", "mul"),
    "multiply": ("pointwise", "binary", "mul"),  # fn normalised to mul
    "div": ("pointwise", "binary", "div"),
    "to_layout": ("identity", "passthrough", None),
    "typecast": ("identity", "passthrough", None),
    "clone": ("identity", "passthrough", None),
    "to_memory_config": ("identity", "passthrough", None),
}

for _leaf, (_canon, _rule, _fn) in GENERIC_OPS.items():
    ALIASES.setdefault(_leaf, _canon)
    ALIASES.setdefault("ttnn." + _leaf, _canon)


def _slice_apply(c: ApplyCtx) -> None:
    x = c.in_state(0)
    xs, ys = c.in_sym(0), c.out_sym(0)
    axis = _axis(int(c.node.attrs["axis"]), xs.ndim)
    start = int(c.node.attrs["start"])
    stop = int(c.node.attrs.get("stop", start + ys.shape[axis]))
    regions = {}
    for dev in c.mesh.devices():
        got = _slice_region(x.regions[dev], axis, start, stop)
        regions[dev] = got.map_axis(axis, lambda lo, hi: (lo - start, hi - start))
    c.define(
        0,
        x.dist,
        regions,
        derive_value_id("slice", c.value_of_inputs(), {"axis": axis, "start": start, "stop": stop}),
        x.tainted,
    )


def _slice_demand(c: DemandCtx) -> None:
    xid = c.node.inputs[0]
    xs = c.sym(xid)
    axis = _axis(int(c.node.attrs["axis"]), xs.ndim)
    start = int(c.node.attrs["start"])
    dem = c.demand_out(0)
    for dev in c.mesh.devices():
        if dem[dev].is_empty:
            continue
        c.need(xid, dev, dem[dev].map_axis(axis, lambda lo, hi: (lo + start, hi + start)))


register(
    OpSpec("slice", META, _slice_apply, _slice_demand, doc="Axis-interval slice; regions shift by `start`."),
    aliases=("chunk", "getitem"),
)


def _concat_apply(c: ApplyCtx) -> None:
    ys = c.out_sym(0)
    axis = _axis(int(c.node.attrs["axis"]), ys.ndim)
    regions = {d: RegionSet.empty(ys.ndim) for d in c.mesh.devices()}
    offset = 0
    dist = None
    for i, sid in enumerate(c.node.inputs):
        st = c.in_state(i)
        s = c.sym(sid)
        if dist is None:
            dist = st.dist
        for dev in c.mesh.devices():
            shifted = st.regions[dev].map_axis(axis, lambda lo, hi, o=offset: (lo + o, hi + o))
            regions[dev] = regions[dev].union(_expand_to(shifted, ys.shape))
        offset += s.shape[axis]
    c.define(
        0,
        dist or Dist.replicated(c.mesh),
        regions,
        derive_value_id("concat", c.value_of_inputs(), {"axis": axis}),
        c.tainted_inputs(),
    )


def _concat_demand(c: DemandCtx) -> None:
    ys = c.sym(c.node.outputs[0])
    axis = _axis(int(c.node.attrs["axis"]), ys.ndim)
    dem = c.demand_out(0)
    offset = 0
    for sid in c.node.inputs:
        s = c.sym(sid)
        span = (offset, offset + s.shape[axis])
        for dev in c.mesh.devices():
            if dem[dev].is_empty:
                continue
            part = _slice_region(dem[dev], axis, *span)
            c.need(sid, dev, _expand_to(part.map_axis(axis, lambda lo, hi, o=offset: (lo - o, hi - o)), s.shape))
        offset += s.shape[axis]


def _expand_to(region: RegionSet, shape: Sequence[int]) -> RegionSet:
    if region.is_empty:
        return RegionSet.empty(len(shape))
    if region.ndim == len(shape):
        return region
    return RegionSet.full(shape)


register(
    OpSpec(
        "concat",
        META,
        _concat_apply,
        _concat_demand,
        doc="Concatenate along one axis; regions shift by the running offset.",
    )
)


# -----------------------------------------------------------------------------
# attention pieces
# -----------------------------------------------------------------------------
def _qkv_layout(node: Node) -> str:
    """``per_device`` (tt_dit default) or ``global`` column ordering.

    ``ColParallelLinear`` fuses QKV with the weight pre-permuted to
    ``[n_dev][q_local | k_local | v_local][head_dim]`` (see
    ``Attention._reshape_and_merge_qkv``), so each device's own shard holds its
    local Q, K and V blocks back to back. ``global`` covers the plain
    ``concat([Q, K, V], dim=-1)`` layout.
    """
    return node.attrs.get("qkv_layout", "per_device")


def _qkv_heads_layout(node: Node) -> Tuple[int, int, int]:
    """``(q_heads, kv_heads, head_dim)`` for a split_qkv node.

    ``kv_heads`` defaults to ``q_heads`` for plain multi-head attention; it is
    smaller under grouped-query attention (Qwen3-VL / MiniMax-H3), where the fused
    tensor is ``(q_heads + 2*kv_heads)*head_dim`` wide rather than ``3*q_heads*``.
    """
    hq = int(node.attrs["heads"])
    hkv = int(node.attrs.get("kv_heads", hq))
    return hq, hkv, int(node.attrs["head_dim"])


def _shard_size_on(mesh, dist: Dist, sym_axis: int, ndim: int) -> int:
    """How many devices ``sym_axis`` is fractured across (1 if replicated on it)."""
    if dist is None:
        return 1
    for m in range(len(mesh.shape)):
        a = dist.shard[m]
        if a is not None and _axis(a, ndim) == sym_axis:
            return mesh.size(m)
    return 1


def _split_qkv_apply(c: ApplyCtx) -> None:
    """[B, S, (Hq+2*Hkv)*Dh] -> q [B,Hq,S,Dh], k,v [B,Hkv,S,Dh] (heads on axis 1).

    GQA-aware: q carries ``heads`` heads, k and v carry ``kv_heads`` (which equals
    ``heads`` for plain MHA). ``per_device`` means each device's column shard is a
    self-contained ``[q_local | k_local | v_local]`` block (nlp_create_qkv_heads);
    otherwise the fused tensor is the global ``[Q | K | V]`` concatenation.
    """
    x = c.in_state(0)
    xs = c.in_sym(0)
    hq, hkv, hd = _qkv_heads_layout(c.node)
    per_device = _qkv_layout(c.node) == "per_device"
    cols = _cols_axis(xs)
    tp = _shard_size_on(c.mesh, x.dist, cols, xs.ndim)
    counts = (hq, hkv, hkv)  # global head count of q, k, v
    loc = (hq // tp, hkv // tp, hkv // tp)  # per-device head count of q, k, v
    loc_off = (0, loc[0], loc[0] + loc[1])  # per-device head offset within a shard
    full_off = (0, hq, hq + hkv)  # global head offset within [Q|K|V]
    w_local = (hq + 2 * hkv) * hd // tp
    for i in range(len(c.node.outputs)):
        ys = c.out_sym(i)
        regions = {}
        for dev in c.mesh.devices():
            rows = x.regions[dev].bounds(_rows_axis(xs))
            b = x.regions[dev].bounds(cols)
            if b is None or rows is None:
                regions[dev] = RegionSet.empty(ys.ndim)
                continue
            if per_device:
                di = b[0] // w_local  # which device slice this column range sits in
                base = loc_off[i] * hd  # local column start of output i's head block
                s, e = max(b[0] - di * w_local, base), min(b[1] - di * w_local, base + loc[i] * hd)
                span = (di * loc[i] + (s - base) // hd, di * loc[i] + -(-(e - base) // hd)) if s < e else None
            else:
                base = full_off[i] * hd  # global column start of output i's head block
                s, e = max(b[0], base), min(b[1], base + counts[i] * hd)
                span = ((s - base) // hd, -(-(e - base) // hd)) if s < e else None
            if span is None:
                regions[dev] = RegionSet.empty(ys.ndim)
                continue
            regions[dev] = RegionSet.of(Box.full(ys.shape).replace_axis(1, *span).replace_axis(2, *rows))
        dist = Dist(
            tuple(None if a is None else (1 if _axis(a, xs.ndim) == cols else 2) for a in x.dist.shard),
            x.dist.partial,
        )
        c.define(i, dist, regions, derive_value_id("split_qkv", c.value_of_inputs(), {"i": i}), x.tainted)


def _split_qkv_demand(c: DemandCtx) -> None:
    xid = c.node.inputs[0]
    xs = c.sym(xid)
    x = c.before.get(xid)
    hq, hkv, hd = _qkv_heads_layout(c.node)
    per_device = _qkv_layout(c.node) == "per_device"
    cols = _cols_axis(xs)
    tp = _shard_size_on(c.mesh, x.dist if x else None, cols, xs.ndim)
    loc = (hq // tp, hkv // tp, hkv // tp)  # per-device head count of q, k, v
    loc_off = (0, loc[0], loc[0] + loc[1])  # per-device head offset within a shard
    full_off = (0, hq, hq + hkv)  # global head offset within [Q|K|V]
    w_local = (hq + 2 * hkv) * hd // tp
    for i in range(len(c.node.outputs)):
        dem = c.demand_out(i)
        for dev in c.mesh.devices():
            want = dem[dev]
            if want.is_empty:
                continue
            hb = want.bounds(1)
            rows = want.bounds(2)
            box = Box.full(xs.shape).replace_axis(_rows_axis(xs), *rows)
            if per_device:
                shard = x.regions[dev].bounds(cols) if x else (0, (hq + 2 * hkv) * hd)
                di = shard[0] // w_local  # which device slice this shard sits in
                lh0, lh1 = max(0, hb[0] - di * loc[i]), min(loc[i], hb[1] - di * loc[i])
                if lh0 >= lh1:
                    continue
                base = shard[0] + loc_off[i] * hd  # column start of output i's block in this shard
                box = box.replace_axis(cols, base + lh0 * hd, base + lh1 * hd)
            else:
                base = full_off[i] * hd
                box = box.replace_axis(cols, base + hb[0] * hd, base + hb[1] * hd)
            c.need(xid, dev, RegionSet.of(box))


def _split_heads_apply(c: ApplyCtx) -> None:
    """[.., N, H*Dh] -> [B, H, N, Dh]: one tensor's heads moved to axis 1."""
    x = c.in_state(0)
    xs, ys = c.in_sym(0), c.out_sym(0)
    head_dim = int(c.node.attrs["head_dim"])
    cols = _cols_axis(xs)
    regions = {}
    for dev in c.mesh.devices():
        b = x.regions[dev].bounds(cols)
        rows = x.regions[dev].bounds(_rows_axis(xs))
        if b is None or rows is None:
            regions[dev] = RegionSet.empty(ys.ndim)
            continue
        box = Box.full(ys.shape).replace_axis(1, b[0] // head_dim, -(-b[1] // head_dim)).replace_axis(2, *rows)
        regions[dev] = RegionSet.of(box)
    dist = Dist(
        tuple(None if a is None else (1 if _axis(a, xs.ndim) == cols else 2) for a in x.dist.shard),
        x.dist.partial,
    )
    c.define(0, dist, regions, derive_value_id("split_heads", c.value_of_inputs(), {}), x.tainted)


def _split_heads_demand(c: DemandCtx) -> None:
    xid = c.node.inputs[0]
    xs = c.sym(xid)
    head_dim = int(c.node.attrs["head_dim"])
    dem = c.demand_out(0)
    for dev in c.mesh.devices():
        want = dem[dev]
        if want.is_empty:
            continue
        hb = want.bounds(1)
        rows = want.bounds(2)
        box = Box.full(xs.shape).replace_axis(_rows_axis(xs), *rows)
        box = box.replace_axis(_cols_axis(xs), hb[0] * head_dim, hb[1] * head_dim)
        c.need(xid, dev, RegionSet.of(box))


register(
    OpSpec(
        "split_heads",
        META,
        _split_heads_apply,
        _split_heads_demand,
        doc="Move the head blocks of one tensor from the feature axis to axis 1 (nlp_create_qkv_heads, fused norm+split).",
    ),
    aliases=(
        "ttnn.experimental.nlp_create_qkv_heads",
        "nlp_create_qkv_heads",
        "create_heads",
    ),
)


def _permute_apply(c: ApplyCtx) -> None:
    """Axis permutation: ``perm[i]`` is the input axis that becomes output axis ``i``."""
    x = c.in_state(0)
    xs, ys = c.in_sym(0), c.out_sym(0)
    perm = [_axis(int(a), xs.ndim) for a in c.node.attrs["perm"]]
    regions = {}
    for dev in c.mesh.devices():
        boxes = []
        for b in x.regions[dev].boxes:
            boxes.append(Box(tuple(b.ranges[perm[i]] for i in range(len(perm)))))
        regions[dev] = RegionSet(ys.ndim, boxes)
    inverse = {src: dst for dst, src in enumerate(perm)}
    dist = Dist(tuple(None if a is None else inverse.get(_axis(a, xs.ndim)) for a in x.dist.shard), x.dist.partial)
    # A permute keeps the values but reorders axes, so it is deliberately *not*
    # value-preserving: equivalence claims across a rank/axis change would need
    # the region algebra to be permutation-aware too.
    c.define(0, dist, regions, derive_value_id("permute", c.value_of_inputs(), {"perm": perm}), x.tainted)


def _permute_demand(c: DemandCtx) -> None:
    xid = c.node.inputs[0]
    xs = c.sym(xid)
    perm = [_axis(int(a), xs.ndim) for a in c.node.attrs["perm"]]
    dem = c.demand_out(0)
    for dev in c.mesh.devices():
        want = dem[dev]
        if want.is_empty:
            continue
        boxes = []
        for b in want.boxes:
            ranges = [None] * xs.ndim
            for dst, src in enumerate(perm):
                ranges[src] = b.ranges[dst]
            boxes.append(Box(tuple(ranges)))
        c.need(xid, dev, RegionSet(xs.ndim, boxes))


register(
    OpSpec("permute", META, _permute_apply, _permute_demand, doc="Axis permutation; regions follow the axes."),
    aliases=("ttnn.permute", "transpose"),
)


register(
    OpSpec(
        "split_qkv_heads", META, _split_qkv_apply, _split_qkv_demand, doc="Split fused QKV and move heads to axis 1."
    ),
    aliases=("ttnn.transformer.split_query_key_value_and_split_heads", "split_query_key_value_and_split_heads"),
)


def _merge_heads_apply(c: ApplyCtx) -> None:
    """[B, H, S, Dh] -> [B, S, H*Dh]."""
    x = c.in_state(0)
    ys = c.out_sym(0)
    head_dim = int(c.node.attrs["head_dim"])
    regions = {}
    for dev in c.mesh.devices():
        hb = x.regions[dev].bounds(1)
        rows = x.regions[dev].bounds(2)
        if hb is None or rows is None:
            regions[dev] = RegionSet.empty(ys.ndim)
            continue
        box = Box.full(ys.shape).replace_axis(_rows_axis(ys), *rows)
        box = box.replace_axis(_cols_axis(ys), hb[0] * head_dim, hb[1] * head_dim)
        regions[dev] = RegionSet.of(box)
    dist = Dist(
        tuple(None if a is None else (_cols_axis(ys) if a == 1 else _rows_axis(ys)) for a in x.dist.shard),
        x.dist.partial,
    )
    c.define(0, dist, regions, derive_value_id("merge_heads", c.value_of_inputs(), {}), x.tainted)


def _merge_heads_demand(c: DemandCtx) -> None:
    xid = c.node.inputs[0]
    xs = c.sym(xid)
    head_dim = int(c.node.attrs["head_dim"])
    dem = c.demand_out(0)
    for dev in c.mesh.devices():
        want = dem[dev]
        if want.is_empty:
            continue
        cols = want.bounds(-1)
        rows = want.bounds(-2)
        box = Box.full(xs.shape).replace_axis(1, cols[0] // head_dim, -(-cols[1] // head_dim)).replace_axis(2, *rows)
        c.need(xid, dev, RegionSet.of(box))


register(
    OpSpec(
        "merge_heads",
        META,
        _merge_heads_apply,
        _merge_heads_demand,
        doc="Concatenate heads back into the feature axis.",
    ),
    aliases=("ttnn.transformer.concatenate_heads", "concatenate_heads"),
)


def _sdpa_apply(c: ApplyCtx) -> None:
    """q,k,v -> out. Each device produces the rows (queries) it holds."""
    q = c.in_state(0)
    ys = c.out_sym(0)
    regions = {}
    for dev in c.mesh.devices():
        regions[dev] = (
            q.regions[dev].intersect(ys.full_region()) if q.regions[dev].ndim == ys.ndim else ys.full_region()
        )
        # k/v must cover the full sequence axis for the heads this device owns
        for i in (1, 2):
            if i >= len(c.node.inputs):
                continue
            st = c.in_state(i)
            s = c.in_sym(i)
            b = st.regions[dev].bounds(2)
            if b is not None and b != (0, s.shape[2]):
                c.warn(
                    "SDPA_KV_COVERAGE",
                    "device %d holds keys/values for sequence %s of %d; attention needs the full sequence"
                    % (dev, list(b), s.shape[2]),
                )
    c.define(0, q.dist, regions, derive_value_id("sdpa", c.value_of_inputs(), c.node.attrs), c.tainted_inputs())


def _sdpa_demand(c: DemandCtx) -> None:
    dem = c.demand_out(0)
    qid = c.node.inputs[0]
    for dev in c.mesh.devices():
        if not dem[dev].is_empty:
            c.need(qid, dev, dem[dev])
    for i in range(1, len(c.node.inputs)):  # k, v (and joint variants): full sequence
        sid = c.node.inputs[i]
        s = c.sym(sid)
        for dev in c.mesh.devices():
            if dem[dev].is_empty:
                continue
            hb = dem[dev].bounds(1) or (0, s.shape[1])
            box = Box.full(s.shape).replace_axis(1, *hb)
            c.need(sid, dev, RegionSet.of(box))


register(
    OpSpec(
        "sdpa",
        COMPUTE,
        _sdpa_apply,
        _sdpa_demand,
        doc="Scaled dot-product attention: needs whole-sequence K/V for the heads it owns.",
    ),
    aliases=(
        "ttnn.transformer.scaled_dot_product_attention",
        "ttnn.transformer.joint_scaled_dot_product_attention",
        "joint_scaled_dot_product_attention",
        "scaled_dot_product_attention",
    ),
)


def _host_read_apply(c: ApplyCtx) -> None:
    """Readback / consumption on a subset of devices (e.g. `tensor[0]` on host)."""
    x = c.in_state(0)
    ys = c.out_sym(0)
    devs = set(int(d) for d in c.node.attrs.get("devices", c.mesh.devices()))
    regions = {d: (x.regions[d] if d in devs else RegionSet.empty(ys.ndim)) for d in c.mesh.devices()}
    c.define(0, x.dist, regions, x.value_id, x.tainted)


def _host_read_demand(c: DemandCtx) -> None:
    xid = c.node.inputs[0]
    devs = set(int(d) for d in c.node.attrs.get("devices", c.mesh.devices()))
    dem = c.demand_out(0)
    for dev in devs:
        c.need(xid, dev, dem[dev])


register(
    OpSpec(
        "host_read",
        META,
        _host_read_apply,
        _host_read_demand,
        preserves_value=True,
        doc="Value is consumed on a subset of devices only (host readback, `get_device_tensors()[0]`).",
    ),
    aliases=("to_torch", "get_device_tensors", "readback"),
)


# -----------------------------------------------------------------------------
# fallback
# -----------------------------------------------------------------------------
def _unknown_apply(c: ApplyCtx) -> None:
    c.warn("UNKNOWN_OP", "no semantics registered for '%s'; assuming it needs and produces everything" % c.node.op)
    _define_everything(c)


def _define_everything(c: ApplyCtx) -> None:
    """Pessimistic definition: full regions everywhere, fresh value, tainted."""
    for i, sid in enumerate(c.node.outputs):
        ys = c.sym(sid)
        c.define(
            i,
            Dist.replicated(c.mesh),
            {d: ys.full_region() for d in c.mesh.devices()},
            derive_value_id(c.node.op, c.value_of_inputs(), {"out": i}),
            True,
        )


def _unknown_demand(c: DemandCtx) -> None:
    for i in range(len(c.node.inputs)):
        c.need_all_local(i)


UNKNOWN = OpSpec("unknown", COMPUTE, _unknown_apply, _unknown_demand, doc="Unregistered op: pessimistic.")
REGISTRY["unknown"] = UNKNOWN


def _unregistered_apply(c: ApplyCtx) -> None:
    """A call the dry run recorded but nobody wrote semantics for.

    Distinct from ``unknown``: the shim assumed this op's output metadata equals
    its first input's, so the *shape* here is a guess, not just the dataflow. Both
    are pessimistic; only this one makes downstream findings unreportable.
    """
    call = c.node.attrs.get("call", "?")
    c.warn(
        "UNREGISTERED_OP",
        "'%s' has no op spec; its output metadata is assumed equal to input 0, so findings "
        "downstream of it are withheld. Register it (see `ditcheck ops --missing`)." % call,
    )
    _define_everything(c)


register(
    OpSpec(
        UNREGISTERED_OP,
        COMPUTE,
        _unregistered_apply,
        _unknown_demand,
        doc="A ttnn call the dry run saw with no semantics: recorded in full, never reasoned through.",
    )
)


def describe_registry() -> str:
    lines = ["Registered op semantics (Tier 1):", ""]
    for name in sorted(REGISTRY):
        spec = REGISTRY[name]
        flags = []
        if spec.is_collective:
            flags.append("collective")
        if spec.preserves_value:
            flags.append("value-preserving")
        lines.append("  %-18s %-9s %s" % (name, spec.kind, spec.doc))
        if flags:
            lines.append("  %-18s %-9s (%s)" % ("", "", ", ".join(flags)))
        seen = sorted(a for a, t in ALIASES.items() if t == name)
        if seen:
            lines.append("  %-18s %-9s aliases: %s" % ("", "", ", ".join(seen)))
        lines.append("")
    return "\n".join(lines)
