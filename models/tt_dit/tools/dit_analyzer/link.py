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

from typing import Dict, List, Optional, Sequence, Tuple

from .ir import PARAM, Graph, Node, Placement, TensorSymbol


def _primary_input(g: Graph, shape) -> Optional[str]:
    """A stage's handoff input: an activation symbol consumed but produced by nothing
    (a graph entry), preferring one whose shape matches the previous stage's output."""
    produced = {o for n in g.nodes for o in n.outputs}
    consumed = {x for n in g.nodes for x in n.inputs}
    entries = [s for s in consumed if s not in produced and g.symbols[s].kind != PARAM]
    return next((s for s in entries if g.symbols[s].shape == shape), entries[0] if entries else None)


def link_stages(stages: Sequence[Tuple], connect: bool = False, stage_steps: Optional[Dict[str, int]] = None) -> Graph:
    """Merge stage graphs into one linked multi-stage graph.

    Each entry is ``(name, graph)``, ``(name, graph, in_sym)``, or
    ``(name, graph, in_sym, source)``:

    * ``in_sym`` wires the handoff explicitly (needed when the pipeline reshapes between
      stages — DiT → VAE unpatchify — so the producer output and this input differ in shape);
    * ``source = (producer_name, output_index)`` names *which* upstream output feeds this
      stage. It defaults to the previous stage's output 0, which reproduces a linear chain.
      A non-default source lets one producer **fan out** to several stages — the MiniMax-H3
      DiT emits ``[video, audio]`` and feeds the video VAE from output 0 and the audio VAE
      from output 1, a DAG rather than a chain.

    ``connect=False`` (default): stages are separated by a decorative readback boundary —
    per-stage findings, no demand across it, every stage seeds demand.

    ``connect=True`` **data-connects** each stage to its source across a demand-carrying
    boundary, and only **terminal** stages (those no other stage draws from) seed demand.
    Backward demand then crosses the boundaries, so a collective at the end of one stage is
    judged by what its consumer actually reads — e.g. the DiT audio output is dead only until
    the audio VAE is wired in as its consumer.
    """
    if not stages:
        raise ValueError("link_stages needs at least one stage")

    first = stages[0][1]
    linked = Graph(name=" -> ".join(s[0] for s in stages), mesh=first.mesh, provenance=first.provenance)

    out_syms: Dict[str, List[str]] = {}  # stage name -> its namespaced output symbol ids
    mesh_of: Dict[str, Optional[str]] = {}
    # a stage that appears as some other stage's source is not a pipeline sink
    drawn_from = {
        (entry[3][0] if len(entry) > 3 else (stages[i - 1][0] if i > 0 else None)) for i, entry in enumerate(stages)
    }
    outputs: List[str] = []
    for i, entry in enumerate(stages):
        name, g = entry[0], entry[1]
        in_override = entry[2] if len(entry) > 2 else None
        source = entry[3] if len(entry) > 3 else ((stages[i - 1][0], 0) if i > 0 else None)
        pfx = name + "/"
        mesh_id = None if i == 0 else name  # first stage is the primary mesh
        if mesh_id is not None:
            linked.meshes[mesh_id] = g.mesh

        def rid(x: str, pfx: str = pfx) -> str:
            return pfx + x

        # a readback boundary separates this stage from the source output it consumes
        redirect: Optional[Tuple[str, str]] = None
        if source is not None:
            src_name, src_idx = source
            prev_out = out_syms[src_name][src_idx]
            prev_mesh = mesh_of[src_name]
            hs = pfx + "__enter"
            src = linked.symbols[prev_out]
            # when connected the handoff lands on this stage's mesh and carries the dist its
            # matching entry expected, so this stage analyses exactly as it did standalone. Its
            # *shape* is the input's (a reshaping handoff differs from the source output).
            wired = (in_override or _primary_input(g, src.shape)) if connect else None
            hshape = g.symbols[wired].shape if wired is not None else src.shape
            hmesh = mesh_id if connect else prev_mesh
            linked.symbols[hs] = TensorSymbol(id=hs, shape=hshape, dtype=src.dtype, value_id=hs, mesh=hmesh)
            if wired is not None and wired in g.placements:
                linked.placements[hs] = Placement(dist=g.placements[wired].dist, region=g.placements[wired].region)
            # connect=True: a demand-carrying readback+re-upload that reads each region from
            # the fewest devices (so a replicated stage output is read once). Disconnected: a
            # decorative device-0 readback that just marks the segment split.
            linked.nodes.append(
                Node(
                    id=pfx + "__boundary",
                    op="stage_boundary" if connect else "host_read",
                    inputs=[prev_out],
                    outputs=[hs],
                    attrs={"boundary": True} if connect else {"devices": [0], "boundary": True},
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

        out_syms[name] = [rid(o) for o in g.outputs]
        mesh_of[name] = mesh_id
        if not connect:
            outputs.extend(out_syms[name])  # disconnected: every stage seeds demand

    if connect:  # only terminal stages seed demand; upstream stages are pulled through the boundaries
        for name in (s[0] for s in stages):
            if name not in drawn_from:
                outputs.extend(out_syms[name])
    linked.outputs = outputs
    linked.meta["stages"] = [s[0] for s in stages]
    linked.meta["connected"] = connect
    # How often each stage runs in ONE generation. Stages in a pipeline do not run at the same
    # rate -- MiniMax-H3 encodes the prompt once and then evaluates the DiT once per denoise
    # step -- so a single per-forward total silently adds quantities with different frequencies.
    # Recording it per stage lets the report state per-generation cost without pretending the
    # graph itself is unrolled (the loop dimension proper is still blocker 24).
    if stage_steps:
        unknown = set(stage_steps) - {s[0] for s in stages}
        if unknown:
            raise ValueError("stage_steps names a stage that was not linked: %s" % ", ".join(sorted(unknown)))
        linked.meta["stage_steps"] = dict(stage_steps)
    return linked
