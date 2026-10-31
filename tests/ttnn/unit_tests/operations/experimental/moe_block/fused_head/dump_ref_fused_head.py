# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reference outputs of the fused head (models/demos/moe_block/moe_block.py FusedHead on mstaletovic/moe-trim) for the
C++ op's bit-exact device test. Run by the lead on a Galaxy in the reference tree's environment (it opens the mesh):

    cd /data/mstaletovic/tt-metal-trimb && python <this tree>/tests/ttnn/unit_tests/operations/experimental/moe_block/\
fused_head/dump_ref_fused_head.py [--mesh 8x4] [--fabric FABRIC_2D_TORUS_XY] [--models glm53,k3] [--chunk 5120]

Each case (op "fused_head", case "<model>-<R>x<C>-c<chunk>-uniform-s<seed>"): the head built as MoEBlock builds it
(FusedHead.plan with the planner's accumulators, deterministic router, the DRAM fallback allowed), run once on x
from the seed. Saved per chip: input x [1, 1, S/SP, H/TP]; outputs se (the shared expert's reduced output), rows (the
owners' joint routing rows), xbuf (the dispatch buffer: only this chip's owned slots are the head's), route, latent
(K3), and the routing plan tensors counts / regions / token_index / table / plan. The weights are not saved: the
test rebuilds them from the seed with weights_of() below (torch's generator, as the reference's random_weights).
The planned layout and accumulators go into meta.json (config) so the C++ test plans the same head."""

import argparse
import dataclasses
import importlib.util
import sys
from pathlib import Path

import torch

import ttnn

HERE = Path(__file__).resolve().parent


def _harness():
    spec = importlib.util.spec_from_file_location("ref_harness", HERE.parent / "ref_harness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def weights_of(model, seed):
    """The reference's test weights (models/demos/moe_block/tests/test_moe_block.py random_weights)."""
    from models.demos.moe_block.tests.test_moe_block import random_weights

    return random_weights(model, seed)


def x_of(T, H, seed):
    g = torch.Generator().manual_seed(seed + 1)
    return torch.randn(T, H, generator=g)


def dump(mesh, R, C, name, chunk, seed, A, hz):
    from models.demos.moe_block.moe_block import MODELS, FusedHead, Geometry, MoEDtypes

    model = {"glm53": MODELS["glm53"], "k3": MODELS["kimi_k3"], "k27": MODELS["kimi_k27"]}.get(name) or MODELS[name]
    g = Geometry.of(mesh, model, chunk, 2, False)
    dtypes = MoEDtypes()
    fit = FusedHead.plan(g, model, dtypes, mesh=mesh, accumulators=A)
    assert fit.ok, fit.why
    w = weights_of(model, seed)
    head = FusedHead(
        mesh,
        g,
        model,
        w,
        [],
        deterministic=True,
        accumulators=A,
        fidelity=dtypes.shared_fidelity,
        w_dtype=dtypes.shared_weights,
    )
    H = model.emb_dim
    x = x_of(chunk, H, seed).bfloat16()
    # (chip (r, c) holds rows r of the chunk's SP split, columns c of H's TP split)
    xt = ttnn.from_torch(
        x.reshape(1, 1, chunk, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(2, 3)),
    )
    if head.head.xdram:
        head.gather_x(xt, ttnn.Topology.Ring if g.tp_wraps else ttnn.Topology.Linear)
    se, out = head(xt)
    outs = dict(se=se, **out, **head.head.routing.outs)
    per_chip = {}
    for nm, t in [("x", xt)] + list(outs.items()):
        for rc, v in hz.from_mesh(t, (R, C)).items():
            per_chip.setdefault(rc, {})[nm] = v
    lay = head.head.planned.lay if head.head.planned is not None else None
    hz.save_case(
        "fused_head",
        f"{name}-{R}x{C}-c{chunk}-uniform-s{seed}",
        (R, C),
        per_chip,
        inputs=["x"],
        outputs=list(outs),
        seed=seed,
        model=name,
        config=dict(
            chunk_tokens=chunk,
            num_links=2,
            deterministic=True,
            head_dram_fallback=True,
            accumulators=A,
            tp_wraps=g.tp_wraps,
            sp_wraps=g.sp_wraps,
            layout=dataclasses.asdict(lay) if lay is not None else None,
            weights="weights_of(model, seed): test_moe_block.random_weights",
            x="x_of(chunk, H, seed): torch.randn(chunk, H, Generator(seed + 1)), bfloat16",
        ),
    )
    for t in outs.values():
        ttnn.deallocate(t)
    ttnn.deallocate(xt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="8x4")
    ap.add_argument("--fabric", default="FABRIC_2D_TORUS_XY")
    ap.add_argument("--models", default="glm53,k3")
    ap.add_argument("--chunk", type=int, default=5120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--accumulators", type=int, default=12)
    a = ap.parse_args()
    R, C = (int(v) for v in a.mesh.split("x"))
    hz = _harness()
    ttnn.set_fabric_config(getattr(ttnn.FabricConfig, a.fabric))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(R, C))
    try:
        for name in a.models.split(","):
            dump(mesh, R, C, name, a.chunk, a.seed, a.accumulators, hz)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    sys.exit(main())
