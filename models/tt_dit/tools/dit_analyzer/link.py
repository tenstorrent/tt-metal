# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Link per-stage graphs into one multi-stage pipeline graph (phase 10c).

A whole pipeline is encoder → DiT → VAE, each on its own (sub)mesh, with host
readbacks between stages. Rather than run the entire pipeline in one process
(which needs the encoder and VAE stood up under the shim), a hand-run flow can
dry-run each stage it cares about separately and link the graphs here:

* each stage keeps its own mesh, registered under the stage name (blocker 22);
* stages are separated by a readback boundary, so ``Graph.segments()`` sees the
  pipeline as a sequence of device segments (blocker 43);
* symbol / node / value ids are namespaced by stage so nothing collides and no
  cross-stage value is mistaken for an equivalence;
* every stage's outputs are graph outputs -- each stage was dry-run
  independently, so its results are read back at its boundary rather than fed as
  the next stage's device input. That keeps a stage's collectives *demanded* (no
  spurious ``dead_collective`` from the artificial disconnection). Data-connecting
  stages (the DiT output *is* the VAE input) needs the real pipeline and is the
  rest of 10c.

The result analyses like any graph: ``analyze_graph(link_stages(...))`` reports
per-stage findings, each resolved on its stage's mesh. ``steps`` and carried
state stay per-stage (deriving them across the denoise loop is the rest of 10c).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .ir import PARAM, Graph, Node, Placement, TensorSymbol


def _primary_input(g: Graph, shape) -> Optional[str]:
    """A stage's handoff input: an activation symbol consumed but produced by nothing
    (a graph entry), preferring one whose shape matches the previous stage's output."""
    produced = {o for n in g.nodes for o in n.outputs}
    consumed = {x for n in g.nodes for x in n.inputs}
    entries = [s for s in consumed if s not in produced and g.symbols[s].kind != PARAM]
    return next((s for s in entries if g.symbols[s].shape == shape), entries[0] if entries else None)


def link_stages(stages: Sequence[Tuple[str, Graph]], connect: bool = False) -> Graph:
    """Merge ``(stage_name, graph)`` pairs into one linked multi-stage graph.

    ``connect=False`` (default): each stage's outputs stay graph outputs, separated by a
    decorative readback boundary — per-stage findings, no demand across the boundary.

    ``connect=True`` **data-connects** consecutive stages (phase 10c): stage N's output is
    read back at the boundary and fed as stage N+1's matching input, and only the *last*
    stage's outputs seed demand. Backward demand then crosses the boundary, so a collective
    at the end of one stage is judged by what the next stage actually consumes — e.g. a
    gather that replicates to every device feeding a boundary that reads back only device 0
    surfaces as redundant, which neither stage reveals alone.
    """
    if not stages:
        raise ValueError("link_stages needs at least one stage")

    first_name, first = stages[0]
    linked = Graph(name=" -> ".join(n for n, _ in stages), mesh=first.mesh, provenance=first.provenance)

    prev_out: Optional[str] = None
    prev_mesh: Optional[str] = None
    outputs: List[str] = []
    for i, (name, g) in enumerate(stages):
        pfx = name + "/"
        mesh_id = None if i == 0 else name  # first stage is the primary mesh
        if mesh_id is not None:
            linked.meshes[mesh_id] = g.mesh

        def rid(x: str) -> str:
            return pfx + x

        # a readback boundary separates this stage from the previous one
        redirect: Optional[Tuple[str, str]] = None
        if prev_out is not None:
            hs = pfx + "__enter"
            src = linked.symbols[prev_out]
            # when connected the handoff lands on this stage's mesh and carries the dist its
            # matching entry expected, so this stage analyses exactly as it did standalone
            wired = _primary_input(g, src.shape) if connect else None
            hmesh = mesh_id if connect else prev_mesh
            linked.symbols[hs] = TensorSymbol(id=hs, shape=src.shape, dtype=src.dtype, value_id=hs, mesh=hmesh)
            if wired is not None and wired in g.placements:
                linked.placements[hs] = Placement(dist=g.placements[wired].dist, region=g.placements[wired].region)
            linked.nodes.append(
                Node(
                    id=pfx + "__boundary",
                    op="host_read",
                    inputs=[prev_out],
                    outputs=[hs],
                    attrs={"devices": [0], "boundary": True},
                    label="stage boundary: %s" % name,
                    mesh=prev_mesh,
                )
            )
            if wired is not None:
                redirect = (rid(wired), hs)  # this stage's consumers read the handoff, not a fresh entry

        for sid, s in g.symbols.items():
            linked.symbols[rid(sid)] = TensorSymbol(
                id=rid(sid),
                shape=s.shape,
                dtype=s.dtype,
                kind=s.kind,
                value_id=pfx + (s.value_id or sid),
                note=s.note,
                mesh=mesh_id,
            )
        for sid, p in g.placements.items():
            linked.placements[rid(sid)] = Placement(dist=p.dist, region=p.region)
        for n in g.nodes:
            ins = [rid(x) for x in n.inputs]
            if redirect is not None:
                ins = [redirect[1] if x == redirect[0] else x for x in ins]
            linked.nodes.append(
                Node(
                    id=rid(n.id),
                    op=n.op,
                    inputs=ins,
                    outputs=[rid(x) for x in n.outputs],
                    attrs=dict(n.attrs),
                    mesh_axis=n.mesh_axis,
                    loc=n.loc,
                    stack=list(n.stack),
                    label=n.label,
                    calls=n.calls,
                    fused_in=n.fused_in,
                    mesh=mesh_id,
                )
            )

        if not connect:
            outputs.extend(rid(o) for o in g.outputs)  # disconnected: every stage seeds demand
        prev_out = rid(g.outputs[0]) if g.outputs else None
        prev_mesh = mesh_id

    if connect:  # only the final stage seeds demand; upstream stages are pulled through the boundaries
        last_name, last_g = stages[-1]
        outputs = [last_name + "/" + o for o in last_g.outputs]
    linked.outputs = outputs
    linked.meta["stages"] = [n for n, _ in stages]
    linked.meta["connected"] = connect
    return linked
