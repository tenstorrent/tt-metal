"""Step-by-step debug of the PCC=0.163 issue.

Isolate which step (slice, rms_norm, reshape, concat) is wrong.
"""
import sys

import torch

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")
import ttnn

TILE = 32
HEAD_DIM = 256
EPS = 1e-6
B = 1
T = 32
N = 6


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def rms_norm_ref(x, w, eps=1e-6):
    """x: [..., hd], w: [hd]. Returns same shape."""
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


def to_host(t, mesh):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:B]


def _run(mesh):
    compute_kernel = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    torch.manual_seed(0)
    w = torch.randn(HEAD_DIM)

    # Build weight [1, hd//32, 32] ROW_MAJOR (1+w baked in)
    w_3d = (1.0 + w).reshape(1, HEAD_DIM // TILE, TILE).bfloat16()
    w_tt = ttnn.from_torch(
        w_3d,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    # Input: [1, 32, 1536]
    x_flat_cpu = torch.randn(B, T, N * HEAD_DIM, dtype=torch.bfloat16)

    x_flat_tt = ttnn.from_torch(
        x_flat_cpu,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    print(f"\n=== Step 1: Verify slice for head 0 ===")
    # Slice head 0: [0, 0, 0] to [1, 32, 256]
    h0 = ttnn.slice(x_flat_tt, [0, 0, 0], [B, T, HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    h0_host = to_host(h0, mesh)  # [1, 32, 256]
    ref_h0 = x_flat_cpu[:, :, :HEAD_DIM]  # [1, 32, 256]
    p0 = pcc(h0_host.float(), ref_h0.float())
    print(f"  PCC(slice_h0, ref_h0) = {p0:.6f}")
    print(f"  h0_host[0, 0, :5] = {h0_host[0, 0, :5].float()}")
    print(f"  ref_h0[0, 0, :5] = {ref_h0[0, 0, :5].float()}")

    print(f"\n=== Step 2: Verify slice for head 1 ===")
    # Slice head 1: [0, 0, 256] to [1, 32, 512]
    h1 = ttnn.slice(x_flat_tt, [0, 0, HEAD_DIM], [B, T, 2 * HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    h1_host = to_host(h1, mesh)  # [1, 32, 256]
    ref_h1 = x_flat_cpu[:, :, HEAD_DIM : 2 * HEAD_DIM]
    p1 = pcc(h1_host.float(), ref_h1.float())
    print(f"  PCC(slice_h1, ref_h1) = {p1:.6f}")
    print(f"  h1_host[0, 0, :5] = {h1_host[0, 0, :5].float()}")
    print(f"  ref_h1[0, 0, :5] = {ref_h1[0, 0, :5].float()}")

    print(f"\n=== Step 3: Verify rms_norm on head 0 slice ===")
    h0_normed_tt = ttnn.rms_norm(
        h0, weight=w_tt, epsilon=EPS, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=compute_kernel
    )
    h0_normed_host = to_host(h0_normed_tt, mesh)
    ref_h0_normed = rms_norm_ref(ref_h0, w)
    p2 = pcc(h0_normed_host.float(), ref_h0_normed.float())
    print(f"  PCC(rms_norm_h0, ref) = {p2:.6f}")

    print(f"\n=== Step 4: Reshape head 0 normed [1,32,256] → [1,1,32,256] ===")
    h0_4d = ttnn.reshape(h0_normed_tt, [B, 1, T, HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    h0_4d_host = to_host(h0_4d, mesh)  # [1, 1, 32, 256]
    ref_h0_4d = ref_h0_normed.reshape(B, 1, T, HEAD_DIM)
    p3 = pcc(h0_4d_host.float(), ref_h0_4d.float())
    print(f"  PCC(reshape_h0_4d, ref) = {p3:.6f}")
    print(f"  h0_4d_host.shape = {h0_4d_host.shape}")

    print(f"\n=== Step 5: Full _qknorm_flat_to_heads result vs reference ===")
    # Build full reference
    x_heads_cpu = x_flat_cpu.reshape(B, T, N, HEAD_DIM)  # [B, T, N, hd]
    ref_normed_all = rms_norm_ref(x_heads_cpu, w)  # [B, T, N, hd]
    ref_bht = ref_normed_all.permute(0, 2, 1, 3).contiguous()  # [B, N, T, hd]

    # TTNN: process all heads
    head_tensors = []
    for h_idx in range(N):
        hh = ttnn.slice(
            x_flat_tt, [0, 0, h_idx * HEAD_DIM], [B, T, (h_idx + 1) * HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        hn = ttnn.rms_norm(
            hh, weight=w_tt, epsilon=EPS, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=compute_kernel
        )
        hh.deallocate(True)
        hn_4d = ttnn.reshape(hn, [B, 1, T, HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hn.deallocate(True)
        head_tensors.append(hn_4d)

    out_concat = ttnn.concat(head_tensors, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for t in head_tensors:
        t.deallocate(True)

    out_host = to_host(out_concat, mesh)  # [1, 6, 32, 256]
    p_full = pcc(out_host.float(), ref_bht.float())
    print(f"  Full concat PCC = {p_full:.6f}")

    # Check each head independently
    for h_idx in range(N):
        p_h = pcc(out_host[:, h_idx : h_idx + 1, :, :].float(), ref_bht[:, h_idx : h_idx + 1, :, :].float())
        print(f"  Head {h_idx} PCC = {p_h:.6f}")

    # Cleanup
    h0.deallocate(True)
    h1.deallocate(True)
    h0_normed_tt.deallocate(True)
    h0_4d.deallocate(True)
    out_concat.deallocate(True)
    x_flat_tt.deallocate(True)
    w_tt.deallocate(True)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
