# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Saves fused_tail reference cases from the Python reference (mstaletovic/moe-trim: models/demos/moe_block/tt/
tail_a2a.py TailA2A, tail_engine.py TailEngine), for test_fused_tail.py to compare the C++ op against bit for bit.

Run in the reference tree's environment, on the machine the cases are for (a Galaxy for 8x4):

    cd /data/mstaletovic/tt-metal-trimb && source python_env/bin/activate
    python <this file> [--mesh 8x4] [--fabric torus_xy] [--chunk 5120] [--models glm53 k3 ...] [--root DIR]

Cases are named <model>_<R>x<C>_c<chunk> (e.g. glm53_8x4_c5120, k3_8x4_c5120).

Per case: the TP group is the mesh row (cluster_axis 1), rows = chunk / SP a mesh row, S = rows / TP owned a chip.
  * a model without a latent (TailA2A): y_own [1, 1, S, H] ROW_MAJOR, shared_expert [1, 1, S, H] TILE -> output
    [1, 1, rows, H / TP];
  * K3 (TailEngine): y_own [1, 1, S, Hr] ROW_MAJOR, shared_expert [1, 1, rows, H / TP] TILE, W_up [H, Hr] and the
    norm weight [Hr] (fp32, in weights.pt next to the chips: the same on every chip) -> output [1, 1, rows, H / TP].
The tail's inputs are not routed (no combine_snap applies): seeded normal rows, as the reference's own tests use. Each
case runs the reference twice on the same inputs and checks the two outputs agree before saving.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # ref_harness

import ttnn  # noqa: E402
from ref_harness import case_dir, from_mesh, save_case  # noqa: E402

OP = "fused_tail"
# model -> its key in moe_block.MODELS; a case is "<model>_<R>x<C>_c<chunk>"
MODELS_BY_NAME = {
    "glm53": "glm53",
    "k3": "kimi_k3",
    "k27": "kimi_k27",
    "glm53_flash": "glm53_flash",
    "dsv41_flash": "dsv41_flash",
}
FABRICS = {"torus_xy": "FABRIC_2D_TORUS_XY", "torus_x": "FABRIC_2D_TORUS_X", "2d": "FABRIC_2D"}


def model_dict(m):
    return dict(
        num_experts=m.num_experts,
        top_k=m.top_k,
        emb_dim=m.emb_dim,
        routed_expert_hidden_dim=m.routed_expert_hidden_dim,
        latent_dim=m.latent_dim,
        latent_norm=m.latent_norm,
        norm_eps=m.norm_eps,
    )


def run_case(mesh, case, model_key, chunk, fabric, payload, root):
    from models.demos.moe_block.moe_block import MODELS
    from models.demos.moe_block.tt.tail_a2a import TailA2A
    from models.demos.moe_block.tt.tail_engine import TailEngine

    m = MODELS[model_key]
    R, C = tuple(mesh.shape)
    rows, H = chunk // R, m.emb_dim
    S, Ho = rows // C, H // C
    ring = fabric in ("FABRIC_2D_TORUS_XY", "FABRIC_2D_TORUS_X") and C > 2
    topology = ttnn.Topology.Ring if ring else ttnn.Topology.Linear
    seed = 0
    g = torch.Generator().manual_seed(seed)
    mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(0, 1))
    put = lambda t, layout: ttnn.from_torch(  # noqa: E731
        t, dtype=ttnn.bfloat16, layout=layout, device=mesh, mesh_mapper=mapper
    )
    extra = {}
    if m.latent_dim is None:
        y = torch.randn(R, C, S, H, generator=g).to(torch.bfloat16)
        se = torch.randn(R, C, S, H, generator=g).to(torch.bfloat16)
        yt, st = put(y, ttnn.ROW_MAJOR_LAYOUT), put(se, ttnn.TILE_LAYOUT)
        tail = TailA2A(mesh, rows=rows, H=H, topology=topology, num_links=2)
        runs = [tail.run(yt, st) for _ in range(2)]
    else:
        Hr = m.latent_dim
        w_up = torch.randn(H, Hr, generator=g) * 0.02
        gn = 1.0 + torch.randn(Hr, generator=g) * 0.1
        y = torch.randn(R, C, S, Hr, generator=g).to(torch.bfloat16)
        se = (torch.randn(R, C, rows, Ho, generator=g) * 0.5).to(torch.bfloat16)
        yt, st = put(y, ttnn.ROW_MAJOR_LAYOUT), put(se, ttnn.TILE_LAYOUT)
        eng = TailEngine(
            mesh,
            rows=rows,
            Hr=Hr,
            Ho=Ho,
            chunks=C,
            eps=m.norm_eps,
            topology=topology,
            num_links=2,
            fidelity=ttnn.MathFidelity.HiFi2,
        )
        w = eng.upload([w_up.T[:, (d % C) * Ho : (d % C + 1) * Ho].contiguous() for d in range(R * C)], gn)
        runs = [eng.run(yt, w, st) for _ in range(2)]
        extra = dict(weights="weights.pt", nslot=eng.NSLOT, engine=f"{eng.GX}x{eng.GY}")
        d = case_dir(OP, case, root)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(dict(latent_up=w_up.contiguous(), latent_norm=gn.contiguous()), d / "weights.pt")
    ttnn.synchronize_device(mesh)
    outs = [from_mesh(o, (R, C)) for o in runs]
    for ch in outs[0]:
        assert torch.equal(
            outs[0][ch].view(torch.int16), outs[1][ch].view(torch.int16)
        ), f"{case}: calls differ at {ch}"
    ins = dict(y_own=from_mesh(yt, (R, C)), shared_expert=from_mesh(st, (R, C)))
    per_chip = {
        ch: dict(y_own=ins["y_own"][ch], shared_expert=ins["shared_expert"][ch], output=outs[0][ch]) for ch in outs[0]
    }
    path = save_case(
        OP,
        case,
        (R, C),
        per_chip,
        inputs=["y_own", "shared_expert"],
        outputs=["output"],
        seed=seed,
        model=model_key,
        config=dict(
            chunk_tokens=chunk,
            num_links=2,
            fabric_config=fabric,
            fabric_payload_bytes=payload,
            math_fidelity="HiFi2",
            weights_dtype="bfloat8_b",
            dims=model_dict(m),
        ),
        root=root,
        extra=extra,
    )
    print(f"{case}: saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", default="8x4")
    ap.add_argument("--fabric", default="torus_xy", choices=sorted(FABRICS))
    ap.add_argument("--models", nargs="*", default=list(MODELS_BY_NAME), choices=list(MODELS_BY_NAME))
    ap.add_argument("--chunk", type=int, default=5120)
    ap.add_argument("--root", type=Path, default=None, help="refs root (default $MOE_BLOCK_REFS)")
    args = ap.parse_args()
    R, C = (int(v) for v in args.mesh.split("x"))
    from models.demos.deepseek_v3_d_p.tests.fabric_profiles import create_fabric_router_config
    from models.demos.moe_block.moe_block import MODELS
    from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import fabric_payload

    # (the block's payload: the combine's row of the widest routed width among the cases)
    keys = [MODELS_BY_NAME[n] for n in args.models]
    payload = max(fabric_payload(MODELS[k].latent_dim or MODELS[k].emb_dim) for k in keys)
    fabric = FABRICS[args.fabric]
    ttnn.set_fabric_config(
        getattr(ttnn.FabricConfig, fabric),
        ttnn.FabricReliabilityMode.RELAXED_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        create_fabric_router_config(max_payload_size=payload),
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(R, C), l1_small_size=0)
    try:
        for name, key in zip(args.models, keys):
            run_case(mesh, f"{name}_{R}x{C}_c{args.chunk}", key, args.chunk, fabric, payload, args.root)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
