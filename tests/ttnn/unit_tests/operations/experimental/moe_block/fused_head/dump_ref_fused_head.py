# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reference outputs of the fused head (models/demos/moe_block/moe_block.py FusedHead on mstaletovic/moe-trim) for the
C++ op's bit-exact device test. Run by the lead on a Galaxy in the reference tree's environment (it opens the mesh):

    cd /data/mstaletovic/tt-metal-trimb && MOE_BLOCK_REFS=<refs root> python <this tree>/tests/ttnn/unit_tests/operations/experimental/moe_block/\
fused_head/dump_ref_fused_head.py [--mesh 8x4] [--fabric FABRIC_2D_TORUS_XY] [--models glm53,k3] [--chunk 5120]

Each case (op "fused_head", case "<model>-<R>x<C>-c<chunk>-uniform-s<seed>"): the head built as MoEBlock builds it
(FusedHead.plan with the planner's accumulators, deterministic router, the DRAM fallback allowed), run once on x
from the seed. Saved per chip: input x [1, 1, S/SP, H/TP]; outputs se (the shared expert's reduced output), rows (the
owners' joint routing rows), xbuf (the dispatch buffer: only this chip's owned slots are the head's), route, latent
(K3), and the routing plan tensors counts / regions / token_index / table / plan; the weights in weights.pt.
The shared expert's and the latent down's weights are put on the bf8 grid first (ref_harness.bf8_grid): the
reference uploads them through ttnn.from_torch, which rounds fp32 to bf16 before packing a sharded bf8 tensor, the C++
op packs fp32 straight to bf8 - values already on the grid come out the same bits both ways (the C++ op's packing).
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


def weights_of(model, seed, hz=None):
    """The reference's test weights (models/demos/moe_block/tests/test_moe_block.py random_weights), the bf8 ones
    (the shared expert's gate_up [2 Is, H] / down [H, Is], the latent down [Hr, H]) on the bf8 grid along their
    output dim (dim 0) when hz (ref_harness) is given."""
    from models.demos.moe_block.tests.test_moe_block import random_weights

    w = random_weights(model, seed)
    if hz is not None:
        for d, k in ((w.shared, "gate_up"), (w.shared, "down"), (w.latent, "down")):
            if d is not None and k in d:
                d[k] = hz.bf8_grid(d[k], dim=0)
    return w


def x_of(T, H, seed):
    g = torch.Generator().manual_seed(seed + 1)
    return torch.randn(T, H, generator=g)


def dump(mesh, R, C, name, chunk, seed, A, hz, fabric):
    from models.demos.moe_block.moe_block import MODELS, FusedHead, Geometry, MoEDtypes

    model = {"glm53": MODELS["glm53"], "k3": MODELS["kimi_k3"], "k27": MODELS["kimi_k27"]}.get(name) or MODELS[name]
    g = Geometry.of(mesh, model, chunk, 2, False)
    dtypes = MoEDtypes()
    fit = FusedHead.plan(g, model, dtypes, mesh=mesh, accumulators=A)
    assert fit.ok, fit.why
    w = weights_of(model, seed, hz)
    # (the free L1 the head's build planned on - HeadSE3 refits on the device's: the C++ op is given the same)
    from models.demos.moe_block.tt import se_engine3

    real, seen = se_engine3.l1_free_bytes, []
    se_engine3.l1_free_bytes = lambda m: seen.append(real(m)) or seen[-1]
    try:
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
    finally:
        se_engine3.l1_free_bytes = real
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
            fabric=fabric,
            l1_free=seen[-1],
            weights="weights.pt: gate, up [Is, H], down [H, Is], latent_down [Hr, H], router_weight [E, H], router_bias [E]",
            x="x_of(chunk, H, seed): torch.randn(chunk, H, Generator(seed + 1)), bfloat16",
        ),
    )
    from models.demos.moe_block.moe_block import split_gate_up

    gate, up = split_gate_up(w.shared["gate_up"].float())
    host = dict(gate=gate, up=up, down=w.shared["down"].float(), router_weight=w.router["weight"].float())
    host["router_bias"] = w.router["bias"].float()
    if model.latent_dim:
        host["latent_down"] = w.latent["down"].float()
    torch.save(
        {k: v.contiguous() for k, v in host.items()},
        hz.case_dir("fused_head", f"{name}-{R}x{C}-c{chunk}-uniform-s{seed}") / "weights.pt",
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
    # (a fresh mesh a model, as the block builds its head on a fresh device: a previous model's head keeps its
    # persistent L1 - K3's least arena, 1199 KB, does not fit in what GLM-5.3's head leaves, 1192 KB)
    for name in a.models.split(","):
        ttnn.set_fabric_config(getattr(ttnn.FabricConfig, a.fabric))
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(R, C), l1_small_size=0)  # (as the block tests)
        try:
            dump(mesh, R, C, name, a.chunk, a.seed, a.accumulators, hz, a.fabric)
        finally:
            ttnn.close_mesh_device(mesh)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    sys.exit(main())
