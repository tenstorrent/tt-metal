# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Galaxy mesh smoke test — confirm the mesh opens at the right shape and CCL works,
BEFORE running any MiniMax-M2 model code.

Why this exists: the single_bh_galaxy mesh-graph-descriptor is [8, 4] (8 rows x 4 cols).
The mesh MUST be opened with the SAME shape as the MGD or device open can hang. DeepSeek
D/P runs prefill at (SP=8, TP=4) = (8,4) which matches. minimax_m2/config.mesh_4x8()
uses (4,8)/TP=8 — that is the orientation to NOT use against this MGD.

Run (on the Galaxy host, with the env activated):

  export TT_METAL_HOME=/data/vmelnykov/tt-metal
  export PYTHONPATH=$TT_METAL_HOME
  source python_env/bin/activate
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto

  # Stage A — open only, per-chip compute, no fabric:
  python3 models/demos/minimax_m2/tests/galaxy_mesh_smoke.py --rows 8 --cols 4
  # Stage B — add fabric + one all_gather across the TP axis:
  python3 models/demos/minimax_m2/tests/galaxy_mesh_smoke.py --rows 8 --cols 4 --collective
"""

import argparse
import sys

import torch

import ttnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8, help="mesh rows (SP axis) — must match MGD")
    ap.add_argument("--cols", type=int, default=4, help="mesh cols (TP axis) — must match MGD")
    ap.add_argument("--collective", action="store_true", help="also exercise fabric + all_gather")
    ap.add_argument(
        "--topology",
        choices=["linear", "ring"],
        default="linear",
        help="linear -> FABRIC_1D + plain MESH MGD (this box); ring -> FABRIC_1D_RING + torus MGD",
    )
    args = ap.parse_args()

    shape = (args.rows, args.cols)
    print(f"[smoke] opening mesh {shape}  collective={args.collective}", flush=True)

    # Stage B needs fabric configured BEFORE the mesh opens. minimax CCL uses the ring
    # topology (see tests/test_factory.py); FABRIC_1D_RING matches DeepSeek's sp<=8 path.
    ring = args.topology == "ring"
    if args.collective:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if ring else ttnn.FabricConfig.FABRIC_1D)

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*shape))
    print(f"[smoke] mesh opened OK: shape={tuple(mesh.shape)} num_devices={mesh.get_num_devices()}", flush=True)

    try:
        # ---- Stage A: per-chip compute, replicated, no inter-chip traffic ----
        rows, cols = shape
        x = torch.randn(1, 1, 32, 64)
        tt = ttnn.from_torch(
            x,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        y = ttnn.add(tt, tt)
        y_host = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float()
        err = (y_host - (x + x)).abs().max().item()
        print(f"[smoke] STAGE A per-chip eltwise OK  max_abs_err={err:.4f}", flush=True)
        assert err < 0.5, f"per-chip compute wrong: {err}"

        # ---- Stage B: fabric all_gather across the TP (col) axis ----
        if args.collective:
            # Shard a tensor along the last dim across the TP axis (cols), then all_gather
            # it back — round-trips through the fabric and must reconstruct the input.
            w = torch.randn(1, 1, 32, 64 * cols)
            sharded = ttnn.from_torch(
                w,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(None, -1)),
            )
            topo = ttnn.Topology.Ring if ring else ttnn.Topology.Linear
            gathered = ttnn.all_gather(sharded, dim=3, cluster_axis=1, topology=topo)
            g_host = ttnn.to_torch(ttnn.get_device_tensors(gathered)[0]).float()
            cerr = (g_host - w).abs().max().item()
            print(f"[smoke] STAGE B all_gather(TP axis) OK  max_abs_err={cerr:.4f}", flush=True)
            assert cerr < 0.5, f"all_gather wrong: {cerr}"

        print("[smoke] PASS", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
