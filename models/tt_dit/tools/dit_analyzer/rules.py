# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Redundancy rules: compare available-vs-needed at every collective.

Each rule emits a :class:`Finding` carrying a machine-readable proof object, so
a reviewer can check the claim without re-deriving the analysis. Rules map 1:1
onto the finding classes in the plan:

========================  =======================================================
``dead_collective``       nothing downstream consumes the result at all
``unused_gather``         consumers only need data the device already had
``duplicate_gather``      an equivalent, uninvalidated collective already ran
``overwide_gather``       more was materialised than any consumer reads
``participant_shrink``    the group is wider than the set needing remote data
``invariant_collective``  the operand is constant across steps: hoist it
========================  =======================================================

Reported separately as *hints* (opportunities, not redundancy):

``mergeable_collectives``  independent collectives that could be issued as one
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .ir import COMM, PARAM, Graph, Node, source_chain
from .region import RegionSet
from .semantics import lookup
from .state import DemandResult, ForwardResult, SymbolState

PROVABLE = "provable"
LIKELY = "likely"
SUSPICIOUS = "suspicious"

HIGH = "high"
MEDIUM = "medium"
LOW = "low"

_SEV_ORDER = {HIGH: 0, MEDIUM: 1, LOW: 2}
_CONF_ORDER = {PROVABLE: 0, LIKELY: 1, SUSPICIOUS: 2}


@dataclass
class Finding:
    rule: str
    title: str
    severity: str
    confidence: str
    nodes: List[str]
    reason: List[str]
    suggestion: str
    proof: Dict[str, Any]
    bytes_per_call: int = 0
    calls: int = 1
    steps: int = 1
    loc: Optional[str] = None
    stack: List[str] = field(default_factory=list)  # tt_dit frames, innermost first
    scope: str = "forward"  # what `calls` counts: one forward pass, or a whole generation
    groups_total: int = 0  # participant groups this collective runs on, mesh-wide

    @property
    def source_chain(self) -> List[str]:
        """Where to point the reader: model call site first, library frame under it."""
        return source_chain(self.stack, self.loc)

    @property
    def bytes_per_forward(self) -> int:
        """Aggregate bytes crossing links, summed over all participants."""
        return self.bytes_per_call * self.calls

    @property
    def devices(self) -> int:
        groups = self.proof.get("participant_groups") or [self.proof.get("participants", [])]
        return sum(len(g) for g in groups) or 1

    @property
    def bytes_per_device(self) -> float:
        return self.bytes_per_forward / float(self.devices)

    @property
    def rank_key(self) -> Tuple:
        return (
            _SEV_ORDER.get(self.severity, 3),
            _CONF_ORDER.get(self.confidence, 3),
            -self.bytes_per_forward,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "title": self.title,
            "severity": self.severity,
            "confidence": self.confidence,
            "nodes": self.nodes,
            "loc": self.loc,
            "source_chain": self.source_chain,
            "reason": self.reason,
            "suggestion": self.suggestion,
            "bytes_per_call": self.bytes_per_call,
            "calls": self.calls,
            "scope": self.scope,
            "bytes_per_%s" % self.scope: self.bytes_per_forward,
            "bytes_per_device": self.bytes_per_device,
            "steps": self.steps,
            "proof": self.proof,
        }


@dataclass
class CollectiveView:
    """Everything the rules need about one collective on one participant group."""

    node: Node
    group: Tuple[int, ...]
    in_state: SymbolState
    out_state: SymbolState
    needed: Dict[int, RegionSet]
    local: Dict[int, RegionSet]
    gained: Dict[int, RegionSet]
    graph: Graph = field(repr=False, default=None)

    @property
    def in_sym(self):
        return self.graph.symbol(self.node.inputs[0])

    @property
    def out_sym(self):
        return self.graph.symbol(self.node.outputs[0])

    @property
    def mesh_axis(self) -> int:
        return self.node.mesh_axis

    @property
    def reduces(self) -> bool:
        """True when the collective sums partial results (RS / all-reduce)."""
        m = self.mesh_axis
        return self.in_state.dist.partial[m] and not self.out_state.dist.partial[m]

    @property
    def movement_only(self) -> bool:
        """True when input and output hold the *same* value, only placed differently.

        Only then may the analyzer treat "already local" as "equivalent", so the
        redundancy rules that rest on equivalence are gated on this.
        """
        return not self.in_state.dist.partial[self.mesh_axis]

    def moved_bytes(self) -> int:
        """First-order ring cost: bytes crossing links per call, for ranking.

        Deliberately independent of the available/needed region delta -- a
        collective whose data is already replicated still pays full fabric cost
        on hardware, and that cost is exactly what removing it recovers.

        Volumes are tile-padded (``padded_volume``): the fabric moves whole
        32x32 tiles, so a gather of a ``num_heads``-column tensor costs a full
        tile row, not its logical width. The region *algebra* stays logical.
        """
        g = len(self.group)
        if g <= 1:
            return 0
        esize = self.out_sym.elem_bytes
        share = float(g - 1) / g
        if self.node.op == "all_reduce":
            return int(sum(self.out_state.regions[d].padded_volume() for d in self.group) * share * 2 * esize)
        if self.node.op == "reduce_scatter":
            return int(sum(self.in_state.regions[d].padded_volume() for d in self.group) * share * esize)
        return int(sum(self.out_state.regions[d].padded_volume() for d in self.group) * share * esize)

    def wasted(self) -> Dict[int, RegionSet]:
        """Regions this collective materialised that nothing downstream reads."""
        return {d: self.gained[d].subtract(self.needed[d]) for d in self.group}

    def needers(self) -> List[int]:
        return [d for d in self.group if not self.local[d].covers(self.needed[d])]

    def useful_senders(self) -> List[int]:
        """Devices whose local data some *other* participant actually needs."""
        out = []
        for p in self.group:
            for d in self.group:
                if d == p:
                    continue
                if not self.needed[d].intersect(self.local[p]).subtract(self.local[d]).is_empty:
                    out.append(p)
                    break
        return out

    def tainted(self) -> bool:
        return self.in_state.tainted or self.out_state.tainted

    def unregistered(self) -> Tuple[str, ...]:
        """Unregistered calls this collective's operand or result flowed through."""
        out = list(self.in_state.unregistered)
        for call in self.out_state.unregistered:
            if call not in out:
                out.append(call)
        return tuple(out)


def _to_input_frame(graph: Graph, node: Node, region: RegionSet) -> RegionSet:
    """Map an output-frame region back to the input frame for a shape-growing collective.

    The sender/needer rules compare ``needed`` / ``gained`` (output frame) against
    ``local`` (input frame), which only aligns when the collective preserves the logical
    shape -- true for gather / reduce_scatter / mesh_partition. ``neighbor_pad`` grows each
    padded dim by ``pad_left + pad_right`` (input row ``i`` sits at output row ``i +
    pad_left``), so its frames are offset by ``pad_left``; without this shift a *symmetric*
    halo looks one-sided and is wrongly flagged ``participant_shrink``. Identity otherwise.
    """
    if node.op != "neighbor_pad" or region.is_empty:
        return region
    xs = graph.symbol(node.inputs[0])
    r = region
    for dim, pl in zip(node.attrs["dims"], node.attrs["pad_left"]):
        r = r.map_axis(dim % xs.ndim, lambda lo, hi, p=pl: (lo - p, hi - p))
    return r.intersect(RegionSet.full(xs.shape))


def collect_views(graph: Graph, fwd: ForwardResult, bwd: DemandResult) -> List[CollectiveView]:
    views: List[CollectiveView] = []
    for node in graph.nodes:
        if not lookup(node.op).is_collective or node.mesh_axis is None:
            continue
        xid, yid = node.inputs[0], node.outputs[0]
        before, after = fwd.before(node.id), fwd.after(node.id)
        if xid not in before or yid not in after:
            continue
        x, y = before[xid], after[yid]
        nd = graph.symbol(yid).ndim
        reduces = x.dist.partial[node.mesh_axis] and not y.dist.partial[node.mesh_axis]
        for group in graph.mesh_of(node).groups(node.mesh_axis):  # the node's mesh (blocker 22)
            # needed / gained are in the output frame; local is the input frame. For a
            # shape-growing halo they must be reconciled or a symmetric halo reads as
            # one-sided (the audio-vocoder false-positive storm).
            needed = {d: _to_input_frame(graph, node, bwd.of(yid, d, nd)) for d in group}
            local = {d: x.regions[d] for d in group}
            # For a reduction the pre-state is an unreduced partial sum, so none of
            # it counts as "already available"; everything the op materialises is new.
            gained = {
                d: (y.regions[d] if reduces else _to_input_frame(graph, node, y.regions[d]).subtract(local[d]))
                for d in group
            }
            views.append(
                CollectiveView(
                    node=node,
                    group=group,
                    in_state=x,
                    out_state=y,
                    needed=needed,
                    local=local,
                    gained=gained,
                    graph=graph,
                )
            )
    return views


# -----------------------------------------------------------------------------
# rules
# -----------------------------------------------------------------------------
def check_collective(graph: Graph, fwd: ForwardResult, bwd: DemandResult, v: CollectiveView) -> Optional[Finding]:
    """Single-collective rules, most conclusive first. None == necessary."""
    for rule in (_rule_dead, _rule_unused, _rule_duplicate, _rule_participants, _rule_overwide):
        finding = rule(graph, fwd, bwd, v)
        if finding is not None:
            return finding
    return None


def _base_proof(v: CollectiveView) -> Dict[str, Any]:
    return {
        "collective": v.node.id,
        "op": v.node.op,
        "fused_in": v.node.fused_in,
        "mesh_axis": v.node.mesh_axis,
        "mesh_axis_name": v.graph.mesh_of(v.node).axis_names[v.node.mesh_axis],
        "participants": list(v.group),
        "tensor": v.in_sym.id,
        "shape": list(v.in_sym.shape),
        "dtype": v.in_sym.dtype,
        "layout_before": v.in_state.dist.describe(v.graph.mesh_of(v.node)),
        "layout_after": v.out_state.dist.describe(v.graph.mesh_of(v.node)),
        "value_id": v.in_state.value_id,
        "available_before": {str(d): v.local[d].describe(v.in_sym.shape) for d in v.group},
        "materialised_after": {str(d): v.out_state.regions[d].describe(v.out_sym.shape) for d in v.group},
        "needed_downstream": {str(d): v.needed[d].describe(v.out_sym.shape) for d in v.group},
        "bytes_moved_per_call": v.moved_bytes(),
        "semantics_complete": not v.tainted(),
        "unregistered_dependencies": list(v.unregistered()),
    }


def _confidence(v: CollectiveView, base: str) -> str:
    return SUSPICIOUS if v.tainted() else base


def _consumer_summary(graph: Graph, symbol: str) -> str:
    cons = graph.consumers_of(symbol)
    if not cons:
        return "no consumers"
    return ", ".join(sorted({c.display + " (" + c.op + ")" for c in cons}))


def _rule_dead(graph, fwd, bwd, v: CollectiveView) -> Optional[Finding]:
    if any(not v.needed[d].is_empty for d in v.group):
        return None
    if v.node.outputs[0] in graph.outputs:
        return None
    proof = _base_proof(v)
    proof["conclusion"] = "no downstream consumer demands any region of the result"
    return Finding(
        rule="dead_collective",
        title="%s result is never consumed" % _name(v),
        severity=HIGH,
        confidence=_confidence(v, PROVABLE),
        nodes=[v.node.id],
        reason=[
            "Backward demand analysis reaches this collective with an empty needed set on every participant.",
            "Consumers of %s: %s." % (v.node.outputs[0], _consumer_summary(graph, v.node.outputs[0])),
        ],
        suggestion="Delete the collective (and any compute that only feeds it).",
        proof=proof,
        bytes_per_call=v.moved_bytes(),
        calls=v.node.calls,
        steps=graph.steps,
        loc=v.node.loc,
    )


def _rule_unused(graph, fwd, bwd, v: CollectiveView) -> Optional[Finding]:
    if not v.movement_only:
        return None  # a reduction changes the value; "already local" proves nothing
    if not all(v.local[d].covers(v.needed[d]) for d in v.group):
        return None
    if v.node.outputs[0] in graph.outputs:
        return None
    proof = _base_proof(v)
    proof["invalidation_check"] = "value-preserving collective: output value_id == input value_id"
    proof["conclusion"] = "every participant already holds every region its consumers read"
    suggestion = "Feed %s directly to the consumers and drop the collective." % v.node.inputs[0]
    reason = [
        "Downstream consumers (%s) demand only regions already present before the collective."
        % _consumer_summary(graph, v.node.outputs[0]),
        "Per device: needed %s vs already local %s."
        % (
            v.needed[v.group[0]].describe(v.out_sym.shape),
            v.local[v.group[0]].describe(v.in_sym.shape),
        ),
    ]
    # If an earlier collective is what made the operand complete, either one can
    # go -- and which one to keep is a scheduling question the analyzer should
    # not pretend to answer.
    upstream = graph.producer_of(v.node.inputs[0])
    if upstream is not None and lookup(upstream.op).is_collective:
        proof["redundant_pair"] = [upstream.id, v.node.id]
        reason.append("%s already made the operand complete on this axis, so the pair is redundant." % upstream.display)
        suggestion = (
            "Drop one of the two: remove %s (keep the fused/later collective, which usually overlaps "
            "communication with compute), or keep %s and drop this one." % (upstream.display, upstream.display)
        )
    return Finding(
        rule="unused_gather",
        title="%s is redundant: consumers only read data each device already had" % _name(v),
        severity=HIGH,
        confidence=_confidence(v, PROVABLE),
        nodes=[v.node.id],
        reason=reason,
        suggestion=suggestion,
        proof=proof,
        bytes_per_call=v.moved_bytes(),
        calls=v.node.calls,
        steps=graph.steps,
        loc=v.node.loc,
    )


def _rule_duplicate(graph, fwd, bwd, v: CollectiveView) -> Optional[Finding]:
    before = fwd.before(v.node.id)
    yid = v.node.outputs[0]
    for sid, st in before.items():
        if sid == v.node.inputs[0] or sid == yid:
            continue
        if st.value_id != v.in_state.value_id:
            continue
        if graph.symbol(sid).ndim != v.out_sym.ndim:
            continue  # equivalence across a rank change is not expressible in the region algebra
        if st.dist.any_partial or v.out_state.dist.any_partial:
            continue
        if not all(st.regions[d].covers(v.out_state.regions[d]) for d in v.group):
            continue
        proof = _base_proof(v)
        proof["equivalent_symbol"] = sid
        proof["equivalent_producer"] = st.producer
        proof["equivalent_regions"] = {str(d): st.regions[d].describe(v.out_sym.shape) for d in v.group}
        proof["invalidation_check"] = (
            "SSA graph: %s is still live and carries value_id %s, identical to the collective's operand; "
            "no intervening node redefines it" % (sid, st.value_id)
        )
        proof["conclusion"] = "an equivalent materialisation already exists on every participant"
        prod = st.producer or "graph input"
        return Finding(
            rule="duplicate_gather",
            title="%s duplicates data already materialised by %s" % (_name(v), prod),
            severity=HIGH,
            confidence=_confidence(v, PROVABLE),
            nodes=[v.node.id] + ([st.producer] if st.producer else []),
            reason=[
                "%s already holds every region this collective produces, on all %d participants." % (sid, len(v.group)),
                "Both carry value_id %s, so no compute between them changed the value." % st.value_id,
            ],
            suggestion="Reuse %s instead of re-communicating (CSE the collective)." % sid,
            proof=proof,
            bytes_per_call=v.moved_bytes(),
            calls=v.node.calls,
            steps=graph.steps,
            loc=v.node.loc,
        )
    return None


def _rule_overwide(graph, fwd, bwd, v: CollectiveView) -> Optional[Finding]:
    wasted = v.wasted()
    waste_vol = sum(r.volume for d, r in wasted.items())
    if waste_vol <= 0:
        return None
    moved = sum(v.gained[d].volume for d in v.group)
    frac = waste_vol / float(moved) if moved else 0.0
    proof = _base_proof(v)
    proof["unneeded"] = {str(d): wasted[d].describe(v.out_sym.shape) for d in wasted if not wasted[d].is_empty}
    proof["wasted_fraction"] = round(frac, 4)
    proof["conclusion"] = "%.0f%% of the communicated volume is never read downstream" % (100 * frac)
    return Finding(
        rule="overwide_gather",
        title="%s moves %.0f%% more data than consumers read" % (_name(v), 100 * frac),
        severity=HIGH if frac >= 0.5 else MEDIUM,
        confidence=_confidence(v, LIKELY),
        nodes=[v.node.id],
        reason=[
            "Consumers (%s) read only part of the materialised tensor." % _consumer_summary(graph, v.node.outputs[0]),
            "Unread on device %d: %s." % (v.group[0], wasted[v.group[0]].describe(v.out_sym.shape)),
        ],
        suggestion="Narrow the collective to the consumed region (gather fewer shards, or slice before gathering).",
        proof=proof,
        bytes_per_call=int(v.moved_bytes() * frac),
        calls=v.node.calls,
        steps=graph.steps,
        loc=v.node.loc,
    )


def _rule_participants(graph, fwd, bwd, v: CollectiveView) -> Optional[Finding]:
    if not v.movement_only:
        return None  # every device contributes a partial sum to a reduction
    needers = v.needers()
    senders = v.useful_senders()
    if not needers:
        return None  # handled by _rule_unused
    n = len(v.group)
    waste = sum(r.volume for r in v.wasted().values())
    if len(needers) == n and not (len(senders) < n and waste == 0):
        # A narrower participant set is not the cleanest description of the
        # problem here; _rule_overwide quantifies the wasted volume instead.
        return None

    proof = _base_proof(v)
    proof["devices_needing_remote_data"] = needers
    proof["devices_whose_data_is_read"] = senders
    proof["conclusion"] = "the participant set is wider than the consumer/producer sets require"
    if len(needers) < n:
        suggestion = (
            "Materialise on devices %s only (gather-to-subset / point-to-point) instead of an "
            "all-collective over %s." % (needers, list(v.group))
        )
        saved = v.moved_bytes() * (n - len(needers)) / float(n)
    else:
        suggestion = "Restrict the participant set to devices %s; the rest contribute nothing that is read." % senders
        saved = v.moved_bytes() * (n - len(senders)) / float(n)
    return Finding(
        rule="participant_shrink",
        title="%s spans %d devices but only %d %s remote data"
        % (_name(v), n, len(needers), "needs" if len(needers) == 1 else "need"),
        severity=MEDIUM,
        confidence=_confidence(v, LIKELY),
        nodes=[v.node.id],
        reason=[
            "Devices needing remote regions: %s." % needers,
            "Devices whose local data is actually read by a peer: %s." % senders,
            "Consumers: %s." % _consumer_summary(graph, v.node.outputs[0]),
        ],
        suggestion=suggestion,
        proof=proof,
        bytes_per_call=int(saved),
        calls=v.node.calls,
        steps=graph.steps,
        loc=v.node.loc,
    )


def check_mergeable_collectives(
    graph: Graph, fwd: ForwardResult, views: Sequence[CollectiveView]
) -> List[Tuple[Finding, CollectiveView]]:
    """Independent collectives on the same axis and group that could be one step.

    This is the plan's "gather before a fusion opportunity" class, in the shape
    that actually occurs in DiT: two neighbouring collectives with no data
    dependency between them, moving separate payloads over the same links. They
    can be served by one wider collective (concatenate the payloads, or fuse the
    consumers). It saves per-collective fixed cost -- semaphores, barrier,
    ring warm-up -- not bytes, so it is reported as a hint rather than as
    recoverable traffic.
    """
    hints: List[Tuple[Finding, CollectiveView]] = []
    order = {n.id: i for i, n in enumerate(graph.nodes)}
    by_key: Dict[Tuple[str, int, Tuple[int, ...]], List[CollectiveView]] = {}
    for v in views:
        by_key.setdefault((v.node.op, v.node.mesh_axis, v.group), []).append(v)

    for (op, mesh_axis, group), vs in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        vs = sorted(vs, key=lambda v: order[v.node.id])
        for i in range(len(vs) - 1):
            a, b = vs[i], vs[i + 1]
            if a.node.calls != b.node.calls:
                continue
            if a.node.fused_in is not None and a.node.fused_in == b.node.fused_in:
                continue  # already stages of one fused kernel
            if _depends_on(graph, b.node, a.node.outputs[0], order):
                continue  # b consumes a's result: they must stay ordered
            between = [n for n in graph.nodes if order[a.node.id] < order[n.id] < order[b.node.id]]
            if any(lookup(n.op).is_collective for n in between):
                continue  # not neighbours on this axis
            proof = _base_proof(b)
            proof["pairs_with"] = a.node.id
            proof["pair_payload_bytes_per_call"] = a.moved_bytes() + b.moved_bytes()
            proof["independent"] = "%s does not consume %s" % (b.node.display, a.node.outputs[0])
            proof["conclusion"] = "two independent %s ops on %s could be issued as one" % (
                op,
                graph.mesh_of(b.node).axis_names[mesh_axis],
            )
            hints.append(
                (
                    Finding(
                        rule="mergeable_collectives",
                        title="%s and %s are independent and share a group: one collective could serve both"
                        % (a.node.display, b.node.display),
                        severity=LOW,
                        confidence=_confidence(b, SUSPICIOUS),
                        nodes=[a.node.id, b.node.id],
                        reason=[
                            "Both are %s over %s across devices %s, with no dependency between them."
                            % (op, graph.mesh_of(b.node).axis_names[mesh_axis], list(group)),
                            "Merging saves fixed per-collective cost (semaphores, barrier, ring warm-up), not bytes.",
                        ],
                        suggestion="Concatenate the two payloads into one %s, or fuse the consumers so one collective feeds both."
                        % op,
                        proof=proof,
                        bytes_per_call=0,
                        calls=b.node.calls,
                        steps=graph.steps,
                        loc=b.node.loc,
                    ),
                    b,
                )
            )
    return hints


def _depends_on(graph: Graph, node: Node, symbol: str, order: Dict[str, int], depth: int = 24) -> bool:
    """Does ``node`` transitively consume ``symbol``?"""
    if symbol in node.inputs:
        return True
    if depth <= 0:
        return True  # give up conservatively
    for sid in node.inputs:
        prod = graph.producer_of(sid)
        if prod is not None and order.get(prod.id, 0) >= order.get(node.id, 0):
            continue
        if prod is not None and _depends_on(graph, prod, symbol, order, depth - 1):
            return True
    return False


def _steps_for(graph: Graph, node_id: str) -> int:
    """How many times this node's stage runs in one generation.

    In a linked pipeline the stages run at different rates -- H3 encodes once and evaluates the
    DiT once per denoise step -- so a graph-level `steps` is the wrong unit. `link_stages`
    records the per-stage frequency; fall back to the graph's own `steps` for a single-stage
    graph, which is what the standalone dry runs produce.
    """
    freq = graph.meta.get("stage_steps")
    if freq and "/" in node_id:
        return int(freq.get(node_id.split("/", 1)[0], graph.steps))
    return graph.steps


def _is_step_invariant(graph: Graph, sid: str, consumer_steps: int = 1, depth: int = 24) -> bool:
    """Is this value identical on every denoise step?

    Params are invariant by definition, and an entry may declare itself so. Everything derived is
    invariant exactly when all of its inputs are -- for *any* deterministic op, which is what
    separates this from `_traces_to_params`: that one asks whether the same BYTES are re-sent and
    so only follows value-preserving ops, while this asks whether the same COMPUTATION is redone.

    Undeclared entries answer False, so a graph that declares nothing produces no findings rather
    than wrong ones.
    """
    sym = graph.symbols.get(sid)
    if sym is None:
        return False
    if sym.kind == PARAM or sym.step_varying is False:
        return True
    # A value produced by a stage that runs *less often* than the consumer is, by construction,
    # constant across the consumer's evaluations: H3 encodes the prompt once above the denoise
    # loop, so everything crossing encoder -> DiT is fixed for all 49 DiT evaluations. Without
    # this, connecting the stages would hide the very redundancy that connecting them reveals,
    # because the declaration on the DiT's own prompt entry is superseded by the wiring.
    if _steps_for(graph, sid) < consumer_steps:
        return True
    prod = graph.producer_of(sid)
    if prod is None or depth <= 0:
        return False
    if sym.step_varying is True:
        return False
    return bool(prod.inputs) and all(_is_step_invariant(graph, i, consumer_steps, depth - 1) for i in prod.inputs)


def check_recomputed_across_steps(
    graph: Graph, fwd: ForwardResult, views: Sequence[CollectiveView]
) -> List[Tuple[Finding, CollectiveView]]:
    """Collectives inside a sub-graph that is recomputed identically on every denoise step.

    `invariant_collective` catches a collective whose operand is *weights*. This catches the
    larger case: a whole branch whose inputs are all step-invariant (H3's token refiner consumes
    only the prompt), so the entire computation -- collectives included -- is redone every step
    to produce the same answer. The fix is not a cheaper collective but hoisting the branch out
    of the denoise loop.
    """
    findings: List[Tuple[Finding, CollectiveView]] = []
    for v in views:
        steps = _steps_for(graph, v.node.id)
        if steps <= 1:
            continue
        if not _is_step_invariant(graph, v.node.inputs[0], steps):
            continue
        if _traces_to_params(graph, v.node.inputs[0]):
            continue  # already reported, more precisely, as invariant_collective
        proof = _base_proof(v)
        proof["operand_kind"] = "step-invariant activation (derived only from step-constant inputs)"
        proof["steps"] = steps
        proof["conclusion"] = "the identical computation is redone on every one of %d evaluations" % steps
        findings.append(
            (
                Finding(
                    rule="recomputed_stage",
                    title="%s is recomputed identically on all %d denoise evaluations" % (_name(v), steps),
                    severity=MEDIUM,
                    confidence=_confidence(v, LIKELY),
                    nodes=[v.node.id],
                    reason=[
                        "Every input of this collective traces back to values that do not change "
                        "between denoise steps, so it produces the same result each time.",
                        "Only the first of %d evaluations is doing new work; the other %d repeat it."
                        % (steps, steps - 1),
                    ],
                    suggestion=(
                        "Hoist this branch out of the denoise loop: compute it once before the loop and "
                        "reuse the result on every step."
                    ),
                    proof=proof,
                    bytes_per_call=v.moved_bytes(),
                    calls=v.node.calls * (steps - 1),
                    steps=steps,
                    loc=v.node.loc,
                    scope="generation",
                ),
                v,
            )
        )
    return findings


def check_invariant_collectives(
    graph: Graph, fwd: ForwardResult, views: Sequence[CollectiveView]
) -> List[Tuple[Finding, CollectiveView]]:
    """Collectives whose operand is constant across denoise steps."""
    if graph.steps <= 1:
        return []
    findings: List[Tuple[Finding, CollectiveView]] = []
    for v in views:
        if not _traces_to_params(graph, v.node.inputs[0]):
            continue
        proof = _base_proof(v)
        proof["operand_kind"] = "param (constant across steps)"
        proof["conclusion"] = "the same bytes are re-communicated on every one of %d denoise steps" % graph.steps
        findings.append(
            (
                Finding(
                    rule="invariant_collective",
                    title="%s re-communicates weights every step (%d steps)" % (_name(v), graph.steps),
                    severity=MEDIUM,
                    confidence=_confidence(v, LIKELY),
                    nodes=[v.node.id],
                    reason=[
                        "The operand traces back only to parameters, which do not change between denoise steps.",
                        "Cost repeats %d x %d times per generation." % (v.node.calls, graph.steps),
                    ],
                    suggestion="Gather once (at weight load / first step) and cache, or store the weight in the gathered layout.",
                    proof=proof,
                    bytes_per_call=v.moved_bytes(),
                    calls=v.node.calls * (graph.steps - 1),
                    steps=graph.steps,
                    loc=v.node.loc,
                    scope="generation",
                ),
                v,
            )
        )
    return findings


def _traces_to_params(graph: Graph, sid: str, depth: int = 12) -> bool:
    sym = graph.symbols.get(sid)
    if sym is None:
        return False
    if sym.kind == PARAM:
        return True
    if depth <= 0:
        return False
    prod = graph.producer_of(sid)
    if prod is None:
        return False
    spec = lookup(prod.op)
    if spec.kind == COMM or spec.preserves_value or spec.name in ("slice", "concat", "identity"):
        return all(_traces_to_params(graph, i, depth - 1) for i in prod.inputs)
    return False


def _name(v: CollectiveView) -> str:
    label = v.node.display
    if v.node.fused_in:
        return "%s (%s fused in %s)" % (label, v.node.op, v.node.fused_in)
    return "%s (%s)" % (label, v.node.op)


# -----------------------------------------------------------------------------
# driver
# -----------------------------------------------------------------------------
@dataclass
class Withheld:
    """A finding that was not emitted because its proof rests on assumed metadata.

    "Analysis withholds, never guesses": a redundancy claim downstream of an op
    with no semantics is not reported and not downgraded, because the shim made
    up that op's output metadata. What it *is* good for is saying which
    registrations would unlock it.
    """

    finding: Finding
    ops: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {"would_report": self.finding.to_dict(), "register_first": list(self.ops)}


@dataclass
class Report:
    graph: Graph
    forward: ForwardResult
    backward: DemandResult
    views: List[CollectiveView]
    findings: List[Finding]  # redundancy claims
    necessary: List[CollectiveView]
    hints: List[Finding] = field(default_factory=list)  # opportunities, not redundancy
    withheld: List[Withheld] = field(default_factory=list)  # blocked on op coverage

    @property
    def diagnostics(self):
        return self.forward.diagnostics + self.backward.diagnostics

    @property
    def missing_ops(self) -> List[str]:
        """Ops to register, most findings unlocked first."""
        counts: Dict[str, int] = {}
        for w in self.withheld:
            for op in w.ops:
                counts[op] = counts.get(op, 0) + 1
        return [op for op, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def run_rules(graph: Graph, fwd: ForwardResult, bwd: DemandResult) -> Report:
    views = collect_views(graph, fwd, bwd)
    groups_per_node: Dict[str, int] = {}
    for v in views:
        groups_per_node[v.node.id] = groups_per_node.get(v.node.id, 0) + 1

    raw: List[Tuple[Finding, CollectiveView]] = []
    blocked: List[Tuple[Finding, CollectiveView]] = []
    necessary: List[CollectiveView] = []
    for v in views:
        f = check_collective(graph, fwd, bwd, v)
        if f is None:
            necessary.append(v)
        elif v.unregistered():
            blocked.append((f, v))
        else:
            raw.append((f, v))
    for f, v in check_recomputed_across_steps(graph, fwd, views):
        (blocked if v.unregistered() else raw).append((f, v))
    for f, v in check_invariant_collectives(graph, fwd, views):
        (blocked if v.unregistered() else raw).append((f, v))

    findings = _merge_across_groups(raw, groups_per_node)
    findings.sort(key=lambda f: f.rank_key)
    hints = _merge_across_groups(
        [(f, v) for f, v in check_mergeable_collectives(graph, fwd, views) if not v.unregistered()], groups_per_node
    )
    withheld = [
        Withheld(finding=f, ops=list(f.proof.get("unregistered_dependencies", [])))
        for f in _merge_across_groups(blocked, groups_per_node)
    ]
    return Report(
        graph=graph,
        forward=fwd,
        backward=bwd,
        views=views,
        findings=findings,
        necessary=necessary,
        hints=hints,
        withheld=withheld,
    )


def _redundancy_signature(f: Finding, v: CollectiveView):
    """A frame-independent fingerprint of *what* is redundant, for the merge key.

    Two participant groups may only be reported as one finding when they are
    redundant in the *same way*. For the whole-value rules (a dead / duplicated
    collective) that is always true, so the signature is constant and every group
    merges (this is what lets a subset-dead collective become one ``replicated_stage``).

    But ``overwide_gather`` / ``participant_shrink`` describe a *partial* waste whose
    shape can differ per group: on a packed [text|audio|video] sequence the first
    shard's unread region is the text prefix while the last shard's is the padding
    tail — different sizes, different rows. Keyed only on (rule, node) those two
    collapse into one finding carrying the first group's region and both groups'
    bytes, over-claiming the waste. Fingerprinting the per-device wasted *volume*
    (frame-independent, unlike absolute coordinates) plus the needer/sender
    positions keeps genuinely-identical groups merged and splits the rest."""
    if f.rule not in ("overwide_gather", "participant_shrink"):
        return None  # whole-value rules: one finding across all groups, as before
    waste = v.wasted()
    vols = tuple(waste[d].padded_volume() for d in v.group)
    idx = {d: i for i, d in enumerate(v.group)}  # positions within the group, not absolute ids
    needers = tuple(sorted(idx[d] for d in v.needers()))
    senders = tuple(sorted(idx[d] for d in v.useful_senders()))
    return (vols, needers, senders)


def _merge_across_groups(
    raw: Sequence[Tuple[Finding, CollectiveView]], groups_per_node: Dict[str, int]
) -> List[Finding]:
    """One finding per (rule, nodes, redundancy-shape), covering every participant group
    it holds on identically.

    A mesh axis usually has several independent communication groups (a 2x4 mesh
    has two TP groups). A verdict that holds on only some of them is reported as
    such rather than silently generalised; and two groups whose redundancy has a
    *different shape* (see :func:`_redundancy_signature`) stay separate findings.
    """
    merged: Dict[Tuple, Finding] = {}
    order: List[Tuple] = []
    for f, v in raw:
        key = (f.rule, tuple(f.nodes), _redundancy_signature(f, v))
        if key not in merged:
            if not f.stack:
                f.stack = list(v.node.stack)
            f.proof["participant_groups"] = [list(v.group)]
            f.proof["node_name"] = _name(v)
            f.groups_total = groups_per_node.get(v.node.id, 1)
            merged[key] = f
            order.append(key)
        else:
            prev = merged[key]
            prev.bytes_per_call += f.bytes_per_call  # each group pays its own traffic
            prev.proof["participant_groups"].append(list(v.group))

    out = []
    for key in order:
        f = merged[key]
        covered = len(f.proof["participant_groups"])
        partial = f.groups_total > covered
        if f.rule == "dead_collective" and partial:
            # Dead on a *subset* of the collective's groups: the other groups still consume
            # it, so it is not deletable. Those groups recompute the same value and read none
            # of it -- a stage replicated across a mesh axis it does not shard (e.g. a
            # TP-only encoder on a TP×SP mesh: every SP row is identical, one is read back).
            # The fix is a narrower mesh, not a deletion, so this is its own rule.
            consumed = f.groups_total - covered
            f.rule = "replicated_stage"
            f.severity = MEDIUM
            f.confidence = LIKELY  # deadness is provable; "run on a submesh" needs no asymmetric consumer
            f.title = "%s is replicated across %d groups but consumed on %d — %d redundant" % (
                f.proof["node_name"],
                f.groups_total,
                consumed,
                covered,
            )
            f.suggestion = (
                "This stage does not shard across this mesh axis, so it replicates the same result "
                "on every group and %d of %d copies are computed then discarded. Run it on a submesh "
                "that omits the redundant groups rather than replicating and reading one back."
                % (covered, f.groups_total)
            )
            f.reason = [
                "The result is consumed on %d of %d participant groups; the other %d recompute the "
                "same replicated value and none of it is read." % (consumed, f.groups_total, covered),
                "Redundant groups: %s. Verify no asymmetric consumer before narrowing the mesh."
                % f.proof["participant_groups"],
            ]
        elif partial:
            f.title += " [only on %d of %d participant groups: %s]" % (
                covered,
                f.groups_total,
                f.proof["participant_groups"],
            )
            f.reason.append(
                "Holds on participant groups %s but not on the others, so check for an asymmetric "
                "consumer before changing shared code." % f.proof["participant_groups"]
            )
        out.append(f)
    return out
