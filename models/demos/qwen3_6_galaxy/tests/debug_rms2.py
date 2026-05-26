"""Test rms_norm with n_heads=1, T=1 vs T=32.
Run: source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
     python3 models/demos/qwen3_6_galaxy/tests/debug_rms2.py
"""
import sys

import torch

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")
import ttnn

TILE = 32
HEAD_DIM = 256
EPS = 1e-6


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def rms_norm_ref(x, w, eps=1e-6):
    xf = x.float()
    norm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (norm * (1.0 + w.float())).to(x.dtype)


def main():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    try:
        _run(mesh)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def to_host(tt_t, mesh):
    return ttnn.to_torch(tt_t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0:1]


def _run(mesh):
    compute_kernel = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    torch.manual_seed(0)
    w = torch.randn(HEAD_DIM)

    # Weight format A: [1, 1, HEAD_DIM//TILE, TILE] (current code)
    w_4d = (1.0 + w).reshape(1, 1, HEAD_DIM // TILE, TILE).bfloat16()
    w_tt_4d = ttnn.from_torch(
        w_4d,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    print("Testing n_heads=1, T=32 (prefill-like)")
    x32 = torch.randn(1, 32, 1, HEAD_DIM, dtype=torch.bfloat16)
    ref32 = rms_norm_ref(x32, w)
    x32_tt = ttnn.from_torch(
        x32,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    out32_tt = ttnn.rms_norm(
        x32_tt, weight=w_tt_4d, epsilon=EPS, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=compute_kernel
    )
    out32 = to_host(out32_tt, mesh)
    p32 = pcc(out32.float(), ref32.float())
    print(f"  4D weight, [1,32,1,256]: PCC={p32:.6f}")
    out32_tt.deallocate(True)
    x32_tt.deallocate(True)

    print("Testing n_heads=1, T=1 (decode-like)")
    x1 = torch.randn(1, 1, 1, HEAD_DIM, dtype=torch.bfloat16)
    ref1 = rms_norm_ref(x1, w)
    x1_tt = ttnn.from_torch(
        x1,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    print(f"  x1_tt.shape = {list(x1_tt.shape)}")
    out1_tt = ttnn.rms_norm(
        x1_tt, weight=w_tt_4d, epsilon=EPS, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=compute_kernel
    )
    out1 = to_host(out1_tt, mesh)
    p1 = pcc(out1.float(), ref1.float())
    print(f"  4D weight, [1,1,1,256]: PCC={p1:.6f}")
    out1_tt.deallocate(True)
    x1_tt.deallocate(True)

    w_tt_4d.deallocate(True)


if __name__ == "__main__":
    main()
