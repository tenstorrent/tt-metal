"""Debug: test if NOT deallocating slice before reshape fixes concat.

The hypothesis: ttnn.reshape creates a view (shares buffer with input).
Deallocating the original slice while the reshaped view is still live corrupts data.
"""
import sys

import torch

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")
import ttnn

TILE = 32
HEAD_DIM = 256
B = 1
T = 32
N = 6


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


def to_host(t, mesh):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:B]


def _run(mesh):
    torch.manual_seed(42)

    flat_cpu = torch.zeros(B, T, N * HEAD_DIM, dtype=torch.bfloat16)
    for h in range(N):
        flat_cpu[:, :, h * HEAD_DIM : (h + 1) * HEAD_DIM] = float(h + 1)

    flat_tt = ttnn.from_torch(
        flat_cpu,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    print("=== Test A: Deallocate slice BEFORE concat (original broken approach) ===")
    heads_a = []
    slices_a = []  # keep slices alive too
    for h in range(N):
        s = ttnn.slice(flat_tt, [0, 0, h * HEAD_DIM], [B, T, (h + 1) * HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        s_4d = ttnn.reshape(s, [B, 1, T, HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        s.deallocate(True)  # ← deallocate slice before concat
        heads_a.append(s_4d)

    out_a = ttnn.concat(heads_a, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_a_host = to_host(out_a, mesh)
    for h in range(N):
        actual = out_a_host[0, h, 0, 0].item()
        expected = float(h + 1)
        match = "OK" if abs(actual - expected) < 0.1 else "MISMATCH"
        print(f"  Head {h}: expected={expected}, actual={actual} [{match}]")
    for t in heads_a:
        t.deallocate(True)
    out_a.deallocate(True)

    print("\n=== Test B: Keep slice alive until after concat ===")
    heads_b = []
    slices_b = []
    for h in range(N):
        s = ttnn.slice(flat_tt, [0, 0, h * HEAD_DIM], [B, T, (h + 1) * HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        s_4d = ttnn.reshape(s, [B, 1, T, HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        slices_b.append(s)  # keep slice alive
        heads_b.append(s_4d)

    out_b = ttnn.concat(heads_b, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_b_host = to_host(out_b, mesh)
    for h in range(N):
        actual = out_b_host[0, h, 0, 0].item()
        expected = float(h + 1)
        match = "OK" if abs(actual - expected) < 0.1 else "MISMATCH"
        print(f"  Head {h}: expected={expected}, actual={actual} [{match}]")
    for t in heads_b:
        t.deallocate(True)
    for s in slices_b:
        s.deallocate(True)
    out_b.deallocate(True)

    print("\n=== Test C: ttnn.clone the reshape to get independent buffer ===")
    heads_c = []
    for h in range(N):
        s = ttnn.slice(flat_tt, [0, 0, h * HEAD_DIM], [B, T, (h + 1) * HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        s_4d = ttnn.reshape(s, [B, 1, T, HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        s_4d_clone = ttnn.clone(s_4d, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # deep copy
        s.deallocate(True)
        s_4d.deallocate(True)
        heads_c.append(s_4d_clone)

    out_c = ttnn.concat(heads_c, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_c_host = to_host(out_c, mesh)
    for h in range(N):
        actual = out_c_host[0, h, 0, 0].item()
        expected = float(h + 1)
        match = "OK" if abs(actual - expected) < 0.1 else "MISMATCH"
        print(f"  Head {h}: expected={expected}, actual={actual} [{match}]")
    for t in heads_c:
        t.deallocate(True)
    out_c.deallocate(True)

    flat_tt.deallocate(True)
    print("\nDone!")


if __name__ == "__main__":
    main()
