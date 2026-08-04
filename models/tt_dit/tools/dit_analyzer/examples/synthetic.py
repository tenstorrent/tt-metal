# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""A small graph that exercises the remaining finding classes.

Each pattern here is drawn from something that exists in the tree:

* dead gather -- a gathered tensor kept "for debugging" that nothing reads
* over-wide gather -- gather the whole feature axis, then slice half of it
  (chunked-linear / swiglu paths do this)
* participant shrink -- gather across the mesh, then read device 0 only
  (``pipelines/ideogram4/pipeline_ideogram4.py:88``)
* invariant collective -- FSDP weight gather inside the block, repeated on every
  denoise step (``layers/linear.py:230``)
"""

from __future__ import annotations

from ..builder import GraphBuilder
from ..ir import Graph, Mesh

SP, TP = 0, 1
S, D = 1024, 2048
STEPS = 20
BLOCKS = 4


def synthetic_redundancy() -> Graph:
    b = GraphBuilder(
        "synthetic_redundancy",
        Mesh(shape=(2, 4), axis_names=("sp", "tp")),
        steps=STEPS,
        note="hand-written patterns, one per finding class",
    )
    x = b.input("x", [1, S, D], shard={SP: 1, TP: 2})
    u = b.input("u", [1, S, D], shard={SP: 1, TP: 2})
    probe = b.input("probe", [1, S, D], shard={SP: 1, TP: 2})
    w_half = b.param("w_half.weight", [D // 2, D], shard={TP: 1})
    # FSDP weight: fractured on the SP axis for memory, gathered before use.
    w_fsdp = b.param("ff1.weight", [D, D], shard={SP: 0, TP: 1})

    with b.block(calls=BLOCKS, loc="example.py"):
        # 1. dead: gathered and never consumed
        b.all_gather(probe, dim=2, mesh_axis=TP, label="ag_debug_unused")

        # 2. over-wide: gathers D, consumer reads D/2
        wide = b.all_gather(x, dim=2, mesh_axis=TP, label="ag_before_half_matmul")
        half = b.slice(wide, axis=2, start=0, stop=D // 2, label="first_half")
        y = b.matmul(half, w_half, label="half_matmul")

        # 3. participant shrink: all-gather, then one device per SP row is read
        gathered = b.all_gather(y, dim=2, mesh_axis=TP, label="ag_before_readback")
        readback = b.host_read(gathered, devices=[0, 4], label="latents_to_host")

        # 4. invariant: FSDP weight gather, same bytes on every denoise step
        wg = b.all_gather(w_fsdp, dim=0, mesh_axis=SP, label="ag_fsdp_weight", loc="models/tt_dit/layers/linear.py:230")
        ug = b.all_gather(u, dim=2, mesh_axis=TP, label="ag_before_fsdp_matmul")  # necessary
        z = b.matmul(ug, wg, label="fsdp_matmul")

    return b.finish([readback, z])
