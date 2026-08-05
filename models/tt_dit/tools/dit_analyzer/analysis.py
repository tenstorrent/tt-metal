# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Forward availability analysis and backward demand analysis.

A collective is redundant exactly when the backward "needed" set is already
satisfied by the forward "available" set without it, so both directions are
computed over the same IR and handed to the rules in :mod:`rules`.
"""

from __future__ import annotations

from typing import Dict, List

from .ir import Graph
from .region import RegionSet
from .semantics import ApplyCtx, DemandCtx, lookup
from .state import DemandResult, Diagnostic, ForwardResult, Snapshot, SymbolState, initial_state


def run_forward(graph: Graph) -> ForwardResult:
    """Simulate per-device availability through every node, in program order."""
    state: Dict[str, SymbolState] = initial_state(graph)
    diags: List[Diagnostic] = []
    snapshots: Dict[str, Snapshot] = {}

    for node in graph.nodes:
        before = dict(state)
        ctx = ApplyCtx(graph=graph, node=node, state=state)
        lookup(node.op).apply(ctx)
        diags.extend(ctx.diagnostics)
        for sid in node.outputs:
            if sid not in state:
                diags.append(Diagnostic(node.id, "NO_OUTPUT_STATE", "op '%s' defined no state for %s" % (node.op, sid)))
        snapshots[node.id] = Snapshot(node=node.id, before=before, after=dict(state))

    return ForwardResult(graph=graph, final=state, snapshots=snapshots, diagnostics=diags)


def run_backward(graph: Graph, forward: ForwardResult) -> DemandResult:
    """Propagate "what is actually needed" backwards from the graph outputs."""
    demand: Dict[str, Dict[int, RegionSet]] = {}
    diags: List[Diagnostic] = []

    def add(sid: str, dev: int, region: RegionSet) -> None:
        if region.is_empty:
            return
        cur = demand.setdefault(sid, {}).get(dev, RegionSet.empty(region.ndim))
        demand[sid][dev] = cur.union(region)

    # Seed: every device must hold its share of each graph output.
    for sid in graph.outputs:
        st = forward.final.get(sid)
        sym = graph.symbol(sid)
        for dev in graph.mesh_of_symbol(sid).devices():
            add(sid, dev, st.regions[dev] if st else sym.full_region())

    for node in reversed(graph.nodes):
        out_demand = {sid: dict(demand.get(sid, {})) for sid in node.outputs}
        if all(not d for d in out_demand.values()) and node.outputs:
            # Nothing downstream wants this node's results. Keep walking: the
            # rules report dead collectives, and inputs may still be needed
            # by other consumers (they add their own demand).
            pass
        ctx = DemandCtx(
            graph=graph,
            node=node,
            before=forward.before(node.id),
            out_demand=out_demand,
        )
        lookup(node.op).demand(ctx)
        diags.extend(ctx.diagnostics)
        for sid, per_dev in ctx.result().items():
            for dev, region in per_dev.items():
                add(sid, dev, region)

    return DemandResult(graph=graph, demand=demand, diagnostics=diags)


def analyze_dataflow(graph: Graph):
    forward = run_forward(graph)
    backward = run_backward(graph, forward)
    return forward, backward
