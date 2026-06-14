"""Phase 7 foundation smoke test: open a (1,N) MeshDevice across N Blackhole cards
and check whether tt_ccl collectives (all_gather / reduce_scatter) work between them.

This determines whether tensor-parallel is viable (needs fabric/ethernet links).
Run: TT_DEVICES=0,1 /home/yito/work/run_zaya_multi.sh python models/demos/zaya1_8b/tests/run_mesh_smoke.py 2
"""
import sys
import torch
import ttnn


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f"[mesh] enabling FABRIC_1D + opening (1,{N}) mesh...", flush=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, N)))
    try:
        print(f"[mesh] OK: shape={mesh.shape}, num_devices={mesh.get_num_devices()}", flush=True)

        # replicate a tensor across all devices, then all-gather along the mesh dim
        x = torch.arange(N * 32 * 32, dtype=torch.float32).reshape(N, 1, 32, 32)
        # shard one [1,1,32,32] slice to each device along mesh dim 1
        t = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 0), mesh_shape=(1, N)),
        )
        print("[mesh] sharded tensor placed; attempting all_gather...", flush=True)
        try:
            g = ttnn.all_gather(t, dim=0, num_links=1)
            ttnn.synchronize_device(mesh)
            gt = ttnn.to_torch(g, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            print(f"[mesh] all_gather OK; gathered shape={tuple(gt.shape)}", flush=True)
            print("[mesh] CCL WORKS -> tensor-parallel is viable", flush=True)
        except Exception as e:
            print(f"[mesh] all_gather FAILED: {type(e).__name__}: {e}", flush=True)
            print("[mesh] CCL unavailable (likely no inter-card fabric/ethernet links)", flush=True)
            # fall back: confirm independent per-device compute at least works
            y = ttnn.add(t, t)
            ttnn.synchronize_device(mesh)
            print(f"[mesh] per-device elementwise OK: {tuple(y.shape)} (data-parallel only)", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass


if __name__ == "__main__":
    main()
