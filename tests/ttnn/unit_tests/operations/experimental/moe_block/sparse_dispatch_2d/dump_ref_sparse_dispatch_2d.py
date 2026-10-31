# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Save sparse_dispatch_2d reference cases from the Python reference (ttnn.bringup.combine_rf.sparse_all_gather_2d,
as tt_k3_token_split_moe.dispatch calls it). Run in the reference tree's environment (mstaletovic/moe-trim):

    cd /data/mstaletovic/tt-metal-trimb && python <this tree>/tests/ttnn/unit_tests/operations/experimental/moe_block/\
sparse_dispatch_2d/dump_ref_sparse_dispatch_2d.py [--cases name,...] [--root DIR]

Cases run on the machine's mesh (Galaxy: the 8x4 ones on FABRIC_2D_TORUS_XY; LoudBox: the 2x4 ones on FABRIC_2D),
chunk 5120 tokens (S = 5120 / N a chip), uniform routing, the MoE block's fabric payload for the row width. Each case
saves per chip xbuf_in (the input: random stale contents, the own rows in place), route (the route words) and xbuf
(the reference's result), and checks the result against the torch model first.
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import sparse_dispatch_2d_inputs as inputs  # noqa: E402
from ref_harness import from_mesh, save_case, to_mesh  # noqa: E402

import ttnn  # noqa: E402
from ttnn.bringup.combine_rf.sparse_all_gather_2d import (  # noqa: E402
    manifest_spec,
    span_entry,
    sparse_all_gather_2d,
)

OP = "sparse_dispatch_2d"
CHUNK = 5120
# name: (mesh, torus (X, Y), model, width (routed_width), num_experts, top_k)
CASES = {
    "glm53-8x4-5120": ((8, 4), (True, True), "GLM-5.3", 6144, 256, 8),
    "k3-8x4-5120": ((8, 4), (True, True), "Kimi-K3", 3584, 896, 16),
    "glm53flash-2x4-5120": ((2, 4), (False, False), "GLM-5.3-Flash", 4096, 288, 8),
    "k3-2x4-5120": ((2, 4), (False, False), "Kimi-K3", 3584, 896, 16),
}
SEED = 2026
NUM_LINKS = 2
MANIFEST_BATCH = 16


def run_case(name, root):
    (R, C), torus, model, W, E, K = CASES[name]
    N = R * C
    S = CHUNK // N
    payload = inputs.fabric_payload(W)
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = payload
    fabric = ttnn.FabricConfig.FABRIC_2D_TORUS_XY if any(torus) else ttnn.FabricConfig.FABRIC_2D
    ttnn.set_fabric_config(fabric, ttnn.FabricReliabilityMode.RELAXED_INIT, router_config=router)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(R, C))
    try:
        gen = torch.Generator().manual_seed(SEED)
        wrap = (torus[0] and C > 2, torus[1] and R > 2)
        route = inputs.route_words(R, C, S, E, K, NUM_LINKS, wrap[0], wrap[1], span_entry, gen)
        xbuf_in = inputs.xbuf_inputs(R, C, S, W, gen)
        per_chip = {ch: {"xbuf_in": xbuf_in[ch], "route": route[ch]} for ch in route}
        xbuf = to_mesh(per_chip, "xbuf_in", mesh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        route_t = to_mesh(per_chip, "route", mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        group = 8 if 2 * W <= 7168 else 2  # (tt_k3_token_split_moe.dispatch)
        man = ttnn.allocate_tensor_on_device(*manifest_spec(mesh, S, num_links=NUM_LINKS, torus=torus))
        sparse_all_gather_2d(
            xbuf, route_t, num_links=NUM_LINKS, manifest=man, torus=torus, group=group, manifest_batch=MANIFEST_BATCH
        )
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(man)
        out = from_mesh(xbuf, (R, C))
        inputs.assert_dispatched(out, xbuf_in, route, R, C, S, name)
        for ch in out:
            per_chip[ch]["xbuf"] = out[ch]
        d = save_case(
            OP,
            name,
            (R, C),
            per_chip,
            inputs=["xbuf_in", "route"],
            outputs=["xbuf"],
            seed=SEED,
            model=model,
            config=dict(
                width=W,
                S=S,
                chunk=CHUNK,
                num_experts=E,
                top_k=K,
                torus=list(torus),
                fabric=str(fabric),
                fabric_payload=payload,
                num_links=NUM_LINKS,
                group=group,
                manifest_batch=MANIFEST_BATCH,
                routing="uniform",
            ),
            root=root,
        )
        print(f"{name}: saved {d}")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=None, help="comma-separated case names (default: those fitting this machine)")
    ap.add_argument("--root", default=None, help="output root (default: $MOE_BLOCK_REFS or ref_harness's)")
    a = ap.parse_args()
    n = ttnn.get_num_devices()
    names = a.cases.split(",") if a.cases else [k for k, v in CASES.items() if v[0][0] * v[0][1] == n]
    if not names:
        sys.exit(f"no case for a {n}-chip machine; cases: {', '.join(CASES)}")
    for name in names:
        run_case(name, Path(a.root) if a.root else None)


if __name__ == "__main__":
    main()
