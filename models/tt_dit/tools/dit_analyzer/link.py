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

from .ir import Graph, Node, Placement, TensorSymbol


def link_stages(stages: Sequence[Tuple[str, Graph]]) -> Graph:
    """Merge ``(stage_name, graph)`` pairs into one linked multi-stage graph."""
    if not stages:
        raise ValueError("link_stages needs at least one stage")

    first_name, first = stages[0]
    linked = Graph(name=" -> ".join(n for n, _ in stages), mesh=first.mesh, provenance=first.provenance)

    prev_out: Optional[str] = None
    prev_mesh: Optional[str] = None
    outputs: List[str] = []  # every stage's outputs are read back at its boundary
    for i, (name, g) in enumerate(stages):
        pfx = name + "/"
        mesh_id = None if i == 0 else name  # first stage is the primary mesh
        if mesh_id is not None:
            linked.meshes[mesh_id] = g.mesh

        def rid(x: str) -> str:
            return pfx + x

        # a readback boundary separates this stage from the previous one
        if prev_out is not None:
            hs = pfx + "__enter"
            src = linked.symbols[prev_out]
            linked.symbols[hs] = TensorSymbol(id=hs, shape=src.shape, dtype=src.dtype, value_id=hs, mesh=prev_mesh)
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
            linked.nodes.append(
                Node(
                    id=rid(n.id),
                    op=n.op,
                    inputs=[rid(x) for x in n.inputs],
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

        outputs.extend(rid(o) for o in g.outputs)
        prev_out = rid(g.outputs[0]) if g.outputs else None
        prev_mesh = mesh_id

    linked.outputs = outputs
    linked.meta["stages"] = [n for n, _ in stages]
    return linked
