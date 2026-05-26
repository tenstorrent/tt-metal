"""Test rms_norm on [rows, 1, hd] for various rows values.

Run:
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python3 models/demos/qwen3_6_galaxy/tests/debug_rms4.py
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


def _run(mesh):
    compute_kernel = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    torch.manual_seed(0)
    w = torch.randn(HEAD_DIM)

    w_3d = (1.0 + w).reshape(1, HEAD_DIM // TILE, TILE).bfloat16()
    w_tt = ttnn.from_torch(
        w_3d,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    for rows in [1, 6, 32, 192]:
        print(f"\nTest [{rows}, 1, {HEAD_DIM}]:")
        x = torch.randn(rows, 1, HEAD_DIM, dtype=torch.bfloat16)
        ref = rms_norm_ref(x, w)  # [rows, 1, HEAD_DIM]
        x_tt = ttnn.from_torch(
            x,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        print(f"  x_tt logical shape: {list(x_tt.shape)}")
        out_tt = ttnn.rms_norm(
            x_tt, weight=w_tt, epsilon=EPS, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=compute_kernel
        )
        # ConcatMeshToTensor on dim=0 with 32 devices, each holding [rows, 1, head_dim]
        # → [32*rows, 1, head_dim]
        out_host = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        # Take device 0's contribution (first `rows` rows)
        out_dev0 = out_host[:rows, :, :]  # [rows, 1, head_dim]
        p = pcc(out_dev0.float(), ref.float())
        print(f"  out_host.shape = {out_host.shape}, out_dev0.shape = {out_dev0.shape}")
        print(f"  PCC = {p:.6f}")
        out_tt.deallocate(True)
        x_tt.deallocate(True)

    w_tt.deallocate(True)
    print("\nAll tests done!")


if __name__ == "__main__":
    main()
