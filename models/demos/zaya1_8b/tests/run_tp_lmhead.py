"""Phase 7: tensor-parallel lm_head across a (1,N) mesh, validated token-exact.

lm_head weight is [2048, vocab=262272] (~1GB bf16). Shard the vocab dim across N
cards: each computes [1,1,vocab/N], all_gather(dim=-1) -> full [1,1,vocab], argmax.
Compare the TP argmax to a single-device full-matmul argmax (must match) and time both.

Run: TT_DEVICES=0,1 /home/yito/work/run_zaya_multi.sh \
        python models/demos/zaya1_8b/tests/run_tp_lmhead.py 2
"""
import sys
import time
import torch
import ttnn

from models.demos.zaya1_8b.tt.model_args import ZayaWeights

CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    w = ZayaWeights()
    embed = w.embed()                          # [vocab, 2048]
    vocab, dim = embed.shape
    wT = embed.t().contiguous()                # [2048, vocab]  (tied lm_head)
    hidden = torch.randn(1, 1, dim) * 0.1

    # ---- single-device golden ----
    print(f"[tp] vocab={vocab} dim={dim}; single-device golden...", flush=True)
    d0 = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 1)))
    try:
        wd = ttnn.from_torch(wT, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d0)
        hd = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d0)
        logits = ttnn.matmul(hd, wd, compute_kernel_config=CKC)
        gold = int(ttnn.to_torch(ttnn.argmax(logits, dim=-1)).reshape(-1)[0])
        t = time.time()
        for _ in range(10):
            lg = ttnn.matmul(hd, wd, compute_kernel_config=CKC)
            _ = int(ttnn.to_torch(ttnn.argmax(lg, dim=-1)).reshape(-1)[0])
        t1 = (time.time() - t) / 10
        print(f"[tp] golden token={gold}; single-device lm_head+argmax {t1*1000:.2f} ms", flush=True)
    finally:
        ttnn.close_mesh_device(d0)

    # ---- N-way tensor-parallel ----
    print(f"[tp] FABRIC_1D + (1,{N}) mesh; sharding vocab across {N} cards...", flush=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, N)))
    try:
        # weight sharded along vocab (dim=1); hidden replicated
        wsh = ttnn.from_torch(
            wT, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 1), mesh_shape=(1, N)))
        hrep = ttnn.from_torch(
            hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def tp_token():
            part = ttnn.matmul(hrep, wsh, compute_kernel_config=CKC)   # [1,1,vocab/N] per card
            full = ttnn.all_gather(part, dim=-1, num_links=1)          # [1,1,vocab] on every card
            # argmax on-device, read one device's result
            am = ttnn.to_torch(ttnn.argmax(full, dim=-1), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            return int(am.reshape(-1)[0])

        tok = tp_token()
        ttnn.synchronize_device(mesh)
        t = time.time()
        for _ in range(10):
            tok = tp_token()
        t2 = (time.time() - t) / 10
        ok = (tok == gold)
        print(f"[tp] TP token={tok}  [{'PASS' if ok else 'FAIL'}] vs golden {gold}", flush=True)
        print(f"[tp] TP({N}) lm_head+all_gather+argmax {t2*1000:.2f} ms  (single {t1*1000:.2f} ms)", flush=True)
        print(f"[tp] speedup x{t1/t2:.2f}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass


if __name__ == "__main__":
    main()
