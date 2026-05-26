"""Test rms_norm behavior with 4D weight for different n_heads values.

Run:
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python3 models/demos/qwen3_6_galaxy/tests/debug_rms_norm.py
"""
import sys

import torch

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")
import ttnn

TILE = 32
HEAD_DIM = 256


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def rms_norm_ref(x, w, eps=1e-6):
    """Zero-centered: output = (1+w) * norm(x)"""
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
    w = torch.randn(HEAD_DIM)  # weight
    eps = 1e-6

    # Create TTNN weight in 4D format used by the code
    w_tt_4d = (1.0 + w.float()).reshape(1, 1, HEAD_DIM // TILE, TILE)
    w_tt = ttnn.from_torch(
        w_tt_4d.bfloat16(),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    print(f"Weight 4D shape: {list(w_tt.shape)}")

    # Also create TTNN weight in 2D format [1, head_dim]
    w_tt_2d = (1.0 + w.float()).reshape(1, HEAD_DIM)
    w_tt2 = ttnn.from_torch(
        w_tt_2d.bfloat16(),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    for n_heads in [1, 6]:
        for T in [1, 32]:
            x = torch.randn(1, T, n_heads, HEAD_DIM, dtype=torch.bfloat16)
            x_tt = ttnn.from_torch(
                x,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

            ref = rms_norm_ref(x, w, eps)

            # 4D weight
            out_tt_4d = ttnn.rms_norm(
                x_tt,
                weight=w_tt,
                epsilon=eps,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=compute_kernel,
            )
            out_4d = to_host(out_tt_4d, mesh)[:, :T, :, :]
            p_4d = pcc(out_4d.float(), ref.float())

            # 2D weight
            out_tt_2d = ttnn.rms_norm(
                x_tt,
                weight=w_tt2,
                epsilon=eps,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=compute_kernel,
            )
            out_2d = to_host(out_tt_2d, mesh)[:, :T, :, :]
            p_2d = pcc(out_2d.float(), ref.float())

            print(f"  [n_heads={n_heads}, T={T}] 4D-weight PCC={p_4d:.6f}, 2D-weight PCC={p_2d:.6f}")

            out_tt_4d.deallocate(True)
            out_tt_2d.deallocate(True)
            x_tt.deallocate(True)


if __name__ == "__main__":
    main()
