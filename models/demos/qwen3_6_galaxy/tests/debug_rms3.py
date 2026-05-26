"""Test rms_norm on [rows, 1, hd] with weight [1, hd//32, 32].

Run:
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python3 models/demos/qwen3_6_galaxy/tests/debug_rms3.py
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

    # 3D weight: [1, hd//TILE, TILE]
    w_3d = (1.0 + w).reshape(1, HEAD_DIM // TILE, TILE).bfloat16()
    w_tt = ttnn.from_torch(
        w_3d,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    print(f"Weight 3D shape: {list(w_tt.shape)}")

    # Test [1, 1, 256] - should work
    print("\nTest [1, 1, HEAD_DIM=256]:")
    x = torch.randn(1, 1, HEAD_DIM, dtype=torch.bfloat16)
    ref = rms_norm_ref(x, w)
    x_tt = ttnn.from_torch(
        x,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    print(f"  x_tt.shape = {list(x_tt.shape)}")
    out_tt = ttnn.rms_norm(
        x_tt, weight=w_tt, epsilon=EPS, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=compute_kernel
    )
    out = to_host(out_tt, mesh)
    p = pcc(out.float(), ref.float())
    print(f"  PCC = {p:.6f}")
    out_tt.deallocate(True)
    x_tt.deallocate(True)

    print("\nTest done!")
    w_tt.deallocate(True)


if __name__ == "__main__":
    main()
