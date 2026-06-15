"""Phase 2 follow-on: batch-size sweep of the N-chip EP expert-MLP to find the largest
batch B where throughput still scales (before the collective / memory ceiling degrades it).
Proxy for the multi-user serving ceiling (the 16GB experts are the capacity-limiting part;
sharding them N-ways frees DRAM for bigger batch). Correctness validated separately
(run_ep_moe.py, PCC 0.999996); here we use random hidden/gate and only time ep_forward.

Run: TT_DEVICES=0,1,2,3       /home/yito/work/run_zaya_multi.sh python .../tests/run_ep_moe_batch.py 4
     TT_DEVICES=0,1,2,3,4,5,6,7 /home/yito/work/run_zaya_multi.sh python .../tests/run_ep_moe_batch.py 8
"""
import sys
import time
import torch
import ttnn

from models.demos.zaya1_8b.tt.model_args import ZayaConfig, ZayaWeights

C = ZayaConfig
LAYER = 1
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
BSWEEP = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def reorder_fc1(fc1, n):
    d = C.dim
    w = d // n
    order = []
    for c in range(n):
        order += list(range(c * w, c * w + w))
        order += list(range(d + c * w, d + c * w + w))
    return fc1[:, :, order].contiguous()


def main():
    w_ = C.dim // N
    n_e = C.n_experts
    w = ZayaWeights()
    p = f"model.layers.{LAYER}.zaya_block.experts.local_experts"
    fc1 = torch.stack([w.get(f"{p}.{e}.linear_fc1.weight").t().contiguous() for e in range(n_e)], 0)
    fc2 = torch.stack([w.get(f"{p}.{e}.linear_fc2.weight").t().contiguous() for e in range(n_e)], 0)

    print(f"[epb] FABRIC_1D + (1,{N}) mesh; expert MLP batch sweep (w={w_}/chip)...", flush=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, N)), l1_small_size=32768)
    try:
        fc1_sh = ttnn.from_torch(reorder_fc1(fc1, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                 mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 2), mesh_shape=(1, N)))
        fc2_sh = ttnn.from_torch(fc2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                 mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 1), mesh_shape=(1, N)))

        def ep_forward(hmesh, gate_m, B):
            h16 = ttnn.repeat(hmesh, ttnn.Shape([n_e, 1, 1]))
            hh = ttnn.matmul(h16, fc1_sh, compute_kernel_config=CKC)
            a = ttnn.slice(hh, [0, 0, 0], [n_e, B, w_])
            b = ttnn.slice(hh, [0, 0, w_], [n_e, B, 2 * w_])
            g = ttnn.mul(ttnn.silu(a), b)
            eo_part = ttnn.matmul(g, fc2_sh, compute_kernel_config=CKC)
            gN = ttnn.all_gather(ttnn.unsqueeze(eo_part, 0), dim=0, num_links=1)
            eo = ttnn.sum(gN, dim=0)
            gate_e = ttnn.permute(ttnn.slice(gate_m, [0, 0, 0], [1, B, n_e]), [2, 1, 0])
            out = ttnn.sum(ttnn.mul(eo, gate_e), dim=0, keepdim=True)
            gskip = ttnn.slice(gate_m, [0, 0, n_e], [1, B, n_e + 1])
            return ttnn.add(out, ttnn.mul(hmesh, gskip))

        best_tput, best_B = 0.0, 0
        print(f"{'B':>6} | {'ms/step':>9} | {'MoE tok/s':>10} | note")
        for B in BSWEEP:
            try:
                hmesh = ttnn.from_torch(torch.randn(1, B, C.dim) * 0.1, dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                gate_m = ttnn.from_torch(torch.softmax(torch.randn(1, B, 17), -1), dtype=ttnn.bfloat16,
                                         layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                for _ in range(2):
                    _ = ep_forward(hmesh, gate_m, B)
                ttnn.synchronize_device(mesh)
                t = time.time()
                for _ in range(5):
                    _ = ep_forward(hmesh, gate_m, B)
                ttnn.synchronize_device(mesh)
                dt = (time.time() - t) / 5
                tput = B / dt
                note = ""
                if tput > best_tput * 1.02:
                    best_tput, best_B = tput, B
                elif tput < best_tput * 0.97:
                    note = "<- throughput degraded"
                print(f"{B:>6} | {dt*1000:9.2f} | {tput:10.1f} | {note}", flush=True)
                if note:
                    break
            except Exception as e:
                print(f"{B:>6} |    -      |     -      | FAILED: {type(e).__name__} (memory ceiling)", flush=True)
                break
        print(f"\n[epb] N={N}: peak MoE-block throughput {best_tput:.0f} tok/s at B={best_B}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass


if __name__ == "__main__":
    main()
