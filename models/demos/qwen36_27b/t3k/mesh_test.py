# SPDX-License-Identifier: Apache-2.0
"""Gating test for 4-chip TP: open a 2x2 mesh and run a CCL all-gather (needs eth)."""
import sys, time
import torch
import ttnn


def try_open(shape):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)  # enable fabric for CCL
    print(f"[open] mesh {shape} (fabric=1D) ...", flush=True)
    t0 = time.time()
    md = ttnn.open_mesh_device(ttnn.MeshShape(*shape))
    print(f"[open] OK in {time.time()-t0:.1f}s, devices={md.get_num_devices()}", flush=True)
    return md


def main():
    shape = (2, 2) if len(sys.argv) < 2 else tuple(int(x) for x in sys.argv[1].split("x"))
    md = try_open(shape)
    try:
        nd = md.get_num_devices()
        # replicate a [1,1,32, 32*nd] tensor; shard last dim across devices, all-gather back
        full = torch.arange(32 * 32 * nd, dtype=torch.float32).reshape(1, 1, 32, 32 * nd)
        t = ttnn.from_torch(
            full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=shape),
        )
        print(f"[shard] per-device shard shape {t.shape}", flush=True)
        print("[ccl] all_gather dim=3 ...", flush=True)
        t0 = time.time()
        g = ttnn.all_gather(t, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(md)
        print(f"[ccl] all_gather OK in {time.time()-t0:.2f}s, gathered shape {g.shape}", flush=True)
        print("MESH+CCL WORKS", flush=True)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()
