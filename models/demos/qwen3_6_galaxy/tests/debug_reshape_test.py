"""Test the _qknorm_flat_to_heads reshape approach.

Verifies that reshape [B, T, n*hd] → [B, n, T, hd] works correctly
for both prefill (T=32) and decode (T=1) cases.

Run:
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python3 models/demos/qwen3_6_galaxy/tests/debug_reshape_test.py
"""
import sys

import torch

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")
import ttnn

TILE = 32
HEAD_DIM = 256
EPS = 1e-6
B = 1
N_Q_PC = 6
N_KV_PC = 1


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

    # Weight [1, hd//32, 32] in ROW_MAJOR (as in _make_qknorm_weight_tt)
    w_3d = (1.0 + w).reshape(1, HEAD_DIM // TILE, TILE).bfloat16()
    w_tt = ttnn.from_torch(
        w_3d,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    def test_case(T, n_heads, label):
        """Test: flat [B, T, n*hd] → [B, n, T, hd] via slice+rms_norm+concat."""
        from models.demos.qwen3_6_galaxy.tt.llama_attention import _qknorm_flat_to_heads

        flat_dim = n_heads * HEAD_DIM
        x_flat_cpu = torch.randn(B, T, flat_dim, dtype=torch.bfloat16)

        # CPU reference: rms_norm per-head, result in [B, n_heads, T, hd] format
        x_heads = x_flat_cpu.reshape(B, T, n_heads, HEAD_DIM)  # [B, T, n_heads, hd]
        ref_normed = rms_norm_ref(x_heads, w)  # [B, T, n_heads, hd]
        ref_bht = ref_normed.permute(0, 2, 1, 3).contiguous()  # [B, n_heads, T, hd]

        # TTNN: upload flat, use _qknorm_flat_to_heads (slice per head, rms_norm, concat)
        x_flat_tt = ttnn.from_torch(
            x_flat_cpu,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        print(f"  x_flat_tt.shape = {list(x_flat_tt.shape)}")

        compute_kernel_loc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        x_bht = _qknorm_flat_to_heads(x_flat_tt, w_tt, EPS, B, n_heads, T, HEAD_DIM, compute_kernel_loc)
        print(f"  x_bht.shape = {list(x_bht.shape)}")

        # Gather from device
        out_host = ttnn.to_torch(x_bht, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:B]
        x_bht.deallocate(True)
        x_flat_tt.deallocate(True)
        print(f"  out_host.shape = {out_host.shape}")

        p = pcc(out_host.float(), ref_bht.float())
        print(f"  PCC = {p:.6f}")
        assert p > 0.99, f"FAILED: PCC={p:.6f} < 0.99"

    print("\n=== Test 1: Q prefill [1, 32, 1536] → [1, 6, 32, 256] ===")
    test_case(T=32, n_heads=N_Q_PC, label="Q prefill")

    print("\n=== Test 2: K prefill [1, 32, 256] → [1, 1, 32, 256] ===")
    test_case(T=32, n_heads=N_KV_PC, label="K prefill")

    print("\n=== Test 3: Q decode [1, 1, 1536] → [1, 6, 1, 256] ===")
    test_case(T=1, n_heads=N_Q_PC, label="Q decode")

    print("\n=== Test 4: K decode [1, 1, 256] → [1, 1, 1, 256] ===")
    test_case(T=1, n_heads=N_KV_PC, label="K decode")

    w_tt.deallocate(True)
    print("\n=== All reshape tests passed! ===")


if __name__ == "__main__":
    main()
