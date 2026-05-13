"""Debug: ttnn.concat along dim=1 for [1, 1, T, hd] tensors.

Tests whether concat([1,1,32,256] × 6, dim=1) → [1,6,32,256] is correct.
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

    # Create 6 distinct [1, 1, 32, 256] tensors with recognizable values
    head_cpus = [torch.zeros(B, 1, T, HEAD_DIM, dtype=torch.bfloat16) for _ in range(N)]
    for h in range(N):
        # Fill with unique pattern: all elements = h+1 (i.e. head 0 = all 1s, head 1 = all 2s, ...)
        head_cpus[h].fill_(float(h + 1))

    # Upload each head to device as [1, 1, 32, 256] TILE_LAYOUT
    head_tts = []
    for h in range(N):
        t = ttnn.from_torch(
            head_cpus[h],
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        head_tts.append(t)

    print(f"Each head shape: {list(head_tts[0].shape)}")

    # Concat along dim=1
    out_tt = ttnn.concat(head_tts, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    print(f"Concat output shape: {list(out_tt.shape)}")

    out_host = to_host(out_tt, mesh)
    print(f"out_host.shape = {out_host.shape}")

    # Expected: out[0, h, :, :] = h+1
    for h in range(N):
        actual = out_host[0, h, 0, 0].item()
        expected = float(h + 1)
        match = "OK" if abs(actual - expected) < 0.1 else "MISMATCH"
        print(f"  Head {h}: expected={expected}, actual={actual} [{match}]")

    # Build expected [1, 6, 32, 256]
    ref = torch.cat(head_cpus, dim=1)  # [1, 6, 32, 256]
    p = pcc(out_host.float(), ref.float())
    print(f"\nPCC(concat_out, expected) = {p:.6f}")

    # Cleanup
    for t in head_tts:
        t.deallocate(True)
    out_tt.deallocate(True)

    # Now test concat from the slice path: start with [1, 32, 1536]
    print("\n=== Test concat from slice path ===")
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

    heads_from_slice = []
    for h in range(N):
        s = ttnn.slice(flat_tt, [0, 0, h * HEAD_DIM], [B, T, (h + 1) * HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Check the slice
        s_host = to_host(s, mesh)
        print(f"  Slice head {h}: [0,0,0]={s_host[0,0,0].item():.1f} (expected {h+1})")
        s_4d = ttnn.reshape(s, [B, 1, T, HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        s.deallocate(True)
        heads_from_slice.append(s_4d)

    out2 = ttnn.concat(heads_from_slice, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out2_host = to_host(out2, mesh)

    print("\n  After concat:")
    for h in range(N):
        actual = out2_host[0, h, 0, 0].item()
        expected = float(h + 1)
        match = "OK" if abs(actual - expected) < 0.1 else "MISMATCH"
        print(f"  Head {h}: expected={expected}, actual={actual} [{match}]")

    for t in heads_from_slice:
        t.deallocate(True)
    out2.deallocate(True)
    flat_tt.deallocate(True)
    print("\nDone!")


if __name__ == "__main__":
    main()
