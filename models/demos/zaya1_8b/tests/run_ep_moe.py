"""Phase 2 primitive: 4-chip tensor-parallel expert MLP (token-exact vs single chip).

Shards the 16-expert swiglu MLP across a (1,4) mesh by splitting the FFN width: each chip
holds, per expert, a 512-wide (gate,up) slice of fc1 and the matching 512 input rows of fc2;
partial fc2 outputs are summed across chips (all_reduce emulated via all_gather+sum). The
router/gate is computed once on the reference chip and replicated (this unit-tests the EP
expert compute, not the router). SPMD-clean: no per-chip data-dependent routing.

Run: TT_DEVICES=0,1,2,3 /home/yito/work/run_zaya_multi.sh \
        python models/demos/zaya1_8b/tests/run_ep_moe.py
"""
import sys
import time
import torch
import ttnn

from models.demos.zaya1_8b.tt.model_args import ZayaConfig, ZayaWeights
from models.demos.zaya1_8b.tt.moe import ZayaMoEBlock
from models.demos.zaya1_8b.tt.standard import to_dev

C = ZayaConfig
LAYER = 1
B = 8
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4    # number of chips (FFN-width TP shards)
CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def reorder_fc1(fc1, n):
    """fc1 [16,2048,4096] (cols 0:2048=gate 'a', 2048:4096=up 'b') -> reorder cols so each
    contiguous 2w block = chip c's (a[cw:(c+1)w], b[cw:(c+1)w]) where w=2048/n."""
    d = C.dim
    w = d // n
    order = []
    for c in range(n):
        order += list(range(c * w, c * w + w))                  # a slice c
        order += list(range(d + c * w, d + c * w + w))          # b slice c
    return fc1[:, :, order].contiguous()


def main():
    w = ZayaWeights()
    p = f"model.layers.{LAYER}.zaya_block.experts.local_experts"
    n_e = C.n_experts
    fc1 = torch.stack([w.get(f"{p}.{e}.linear_fc1.weight").t().contiguous() for e in range(n_e)], 0)  # [16,2048,4096]
    fc2 = torch.stack([w.get(f"{p}.{e}.linear_fc2.weight").t().contiguous() for e in range(n_e)], 0)  # [16,2048,2048]
    hidden = torch.randn(1, B, C.dim) * 0.1

    print("[ep] single-chip golden...", flush=True)
    d0 = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 1)))
    try:
        blk = ZayaMoEBlock(d0, w, LAYER)
        hd = to_dev(hidden, d0)
        out, _, gate = blk.forward(hd, None, return_gate=True)
        golden = ttnn.to_torch(out).float()
        gate_t = ttnn.to_torch(gate).float()
        t = time.time()
        for _ in range(10):
            _ = blk.forward(hd, None)
        t1 = (time.time() - t) / 10
        print(f"[ep] golden ready; single-chip MoE {t1*1000:.2f} ms", flush=True)
    finally:
        ttnn.close_mesh_device(d0)

    print(f"[ep] FABRIC_1D + (1,{N}) mesh; sharded expert MLP...", flush=True)
    w = C.dim // N                                                         # gate/up slice width per chip
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, N)))
    try:
        fc1r = reorder_fc1(fc1, N)
        fc1_sh = ttnn.from_torch(fc1r, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                 mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 2), mesh_shape=(1, N)))  # [16,2048,2w]/chip
        fc2_sh = ttnn.from_torch(fc2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                 mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 1), mesh_shape=(1, N)))  # [16,w,2048]/chip
        hmesh = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        gate_m = ttnn.from_torch(gate_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def ep_forward():
            h16 = ttnn.repeat(hmesh, ttnn.Shape([n_e, 1, 1]))              # [16,B,2048]
            hh = ttnn.matmul(h16, fc1_sh, compute_kernel_config=CKC)       # [16,B,2w] (w a | w b)
            a = ttnn.slice(hh, [0, 0, 0], [n_e, B, w])
            b = ttnn.slice(hh, [0, 0, w], [n_e, B, 2 * w])
            g = ttnn.mul(ttnn.silu(a), b)                                  # [16,B,w]
            eo_part = ttnn.matmul(g, fc2_sh, compute_kernel_config=CKC)    # [16,B,2048] partial
            gN = ttnn.all_gather(ttnn.unsqueeze(eo_part, 0), dim=0, num_links=1)  # [N,16,B,2048]
            eo = ttnn.sum(gN, dim=0)                                       # [16,B,2048] full
            gate_e = ttnn.permute(ttnn.slice(gate_m, [0, 0, 0], [1, B, n_e]), [2, 1, 0])  # [16,B,1]
            out = ttnn.sum(ttnn.mul(eo, gate_e), dim=0, keepdim=True)      # [1,B,2048]
            gskip = ttnn.slice(gate_m, [0, 0, n_e], [1, B, n_e + 1])
            return ttnn.add(out, ttnn.mul(hmesh, gskip))

        res = ep_forward()
        ttnn.synchronize_device(mesh)
        got = ttnn.to_torch(res, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1].float()
        pcc = torch.corrcoef(torch.stack([golden.flatten(), got.flatten()]))[0, 1].item()
        maxerr = (golden - got).abs().max().item()
        print(f"[ep] PCC vs single-chip = {pcc:.6f}, max|err| = {maxerr:.4g}", flush=True)
        t = time.time()
        for _ in range(10):
            _ = ep_forward()
        ttnn.synchronize_device(mesh)
        t4 = (time.time() - t) / 10
        ok = pcc > 0.999
        print(f"[ep] [{'PASS' if ok else 'FAIL'}] PCC>0.999 | {N}-chip {t4*1000:.2f} ms (single {t1*1000:.2f} ms, {t1/t4:.2f}x)", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass


if __name__ == "__main__":
    main()
