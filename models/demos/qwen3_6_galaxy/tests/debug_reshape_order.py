"""Check what data ordering TTNN reshape produces for [1, 32, 1536] → [1, 6, 32, 256].

Run:
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python3 models/demos/qwen3_6_galaxy/tests/debug_reshape_order.py
"""
import sys

import torch

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")
import ttnn


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


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
    # Use a simple tensor with recognizable values
    B, T, N, HD = 1, 32, 6, 256
    x_cpu = torch.arange(B * T * N * HD, dtype=torch.float32).reshape(B, T, N * HD).bfloat16()

    # Upload as [1, 32, 1536]
    x_tt = ttnn.from_torch(
        x_cpu,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    print(f"x_tt.shape = {list(x_tt.shape)}")

    # Reshape to [1, 6, 32, 256]
    x_bht = ttnn.reshape(x_tt, [B, N, T, HD], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    print(f"x_bht.shape = {list(x_bht.shape)}")

    # Gather
    out = ttnn.to_torch(x_bht, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:B]
    print(f"out.shape = {out.shape}")

    # What does PyTorch say the reshape should be?
    # Option 1: [B, T, N, HD] then permute to [B, N, T, HD] (transpose)
    expected_1 = x_cpu.reshape(B, T, N, HD).permute(0, 2, 1, 3)
    p1 = pcc(out.float(), expected_1.float())
    print(f"PCC vs [T, N, HD].permute(N, T, HD) = {p1:.6f}")

    # Option 2: direct reshape [B, N, T, HD]
    expected_2 = x_cpu.reshape(B, N, T, HD)
    p2 = pcc(out.float(), expected_2.float())
    print(f"PCC vs direct reshape [N, T, HD] = {p2:.6f}")

    # Check first few values
    print(f"\nOut [0,0,0,:5] = {out[0,0,0,:5].float()}")
    print(f"Expected_1 [0,0,0,:5] = {expected_1[0,0,0,:5].float()}")
    print(f"Expected_2 [0,0,0,:5] = {expected_2[0,0,0,:5].float()}")

    # x_cpu[0, t=0, hd=0:1536]: elements 0..1535
    # expected_1 = x_cpu.reshape(B, T, N, HD).permute(0, 2, 1, 3)
    #   = [B, N, T, HD] where [0, h, t, d] = x_cpu[0, t, h*256+d]
    #   So [0, 0, 0, 0:5] = x_cpu[0, 0, 0:5] = [0, 1, 2, 3, 4]
    # expected_2 = x_cpu.reshape(B, N, T, HD)
    #   = [B, N, T, HD] where [0, n, t, d] = x_cpu.flatten()[n*T*HD + t*HD + d]
    #   = x_cpu[0, n*T + t, ?]... Actually it does row-major reshape
    #   [0, n, t, d] = x_cpu.flatten()[n*T*HD + t*HD + d]
    #   x_cpu.flatten()[k] = x_cpu[0, k//1536, k%1536]
    #   So [0, 0, 0, 0:5] = x_cpu.flatten()[0:5] = [0, 1, 2, 3, 4]
    # Both options give [0, 1, 2, 3, 4] for [0, 0, 0, 0:5]... but they differ elsewhere

    print(f"\nOut [0,1,0,:5] = {out[0,1,0,:5].float()}")
    print(f"Expected_1 [0,1,0,:5] = {expected_1[0,1,0,:5].float()}")
    print(f"Expected_2 [0,1,0,:5] = {expected_2[0,1,0,:5].float()}")

    # expected_1[0,1,0,:5] = x_cpu[0, t=0, h=1, d=0:5] = x_cpu.flatten()[0*1536 + 1*256 + 0:5] = [256, 257, 258, 259, 260]
    # expected_2[0,1,0,:5] = x_cpu.flatten()[1*32*256 + 0*256 + 0:5] = [8192, 8193, 8194, 8195, 8196]

    x_bht.deallocate(True)
    x_tt.deallocate(True)


if __name__ == "__main__":
    main()
