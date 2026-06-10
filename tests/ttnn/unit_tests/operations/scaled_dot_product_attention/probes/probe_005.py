import math
import torch
import ttnn

device = ttnn.open_device(device_id=0)


def torch_sdpa(q, k, v, scale=None):
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    s = scale if scale is not None else 1.0 / math.sqrt(qf.shape[-1])
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * s
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, vf).to(q.dtype)


try:
    # ------------------------------------------------------------------
    # Patch SUPPORTED to allow non-aligned for this probe; then call
    # the program descriptor directly using the existing kernel.
    # ------------------------------------------------------------------
    from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention as sdpa_mod
    from ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention import (
        SUPPORTED,
        scaled_dot_product_attention,
    )

    # Add the non-aligned values temporarily
    SUPPORTED["alignment"].extend(["w_non_aligned", "h_non_aligned"])
    print("Patched SUPPORTED alignment:", SUPPORTED["alignment"])

    # CASE 1: W non-aligned (D=50)
    torch.manual_seed(0)
    Q = torch.randn(1, 1, 32, 50, dtype=torch.bfloat16)
    K = torch.randn(1, 1, 32, 50, dtype=torch.bfloat16)
    V = torch.randn(1, 1, 32, 50, dtype=torch.bfloat16)
    expected = torch_sdpa(Q, K, V)
    print(f"\nCASE 1 — W non-aligned (D=50):")
    print(f"  expected shape: {expected.shape}")
    try:
        ttnn_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_out = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V)
        torch_out = ttnn.to_torch(ttnn_out)
        diff = (torch_out.float() - expected.float()).abs()
        print(f"  output shape: {torch_out.shape}")
        print(f"  max diff: {diff.max().item():.6f}")
        print(f"  mean diff: {diff.mean().item():.6f}")
        from tests.ttnn.utils_for_testing import comp_pcc

        passed, pcc_val = comp_pcc(torch_out.float(), expected.float())
        print(f"  PCC: {pcc_val}")
    except Exception as e:
        print(f"  RUNTIME ERROR: {type(e).__name__}: {e}")

    # CASE 2: H non-aligned in S_q only (S_q=47, D=64)
    torch.manual_seed(0)
    Q = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)
    K = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)
    V = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)
    expected = torch_sdpa(Q, K, V)
    print(f"\nCASE 2 — H non-aligned (S=47, D=64):")
    print(f"  expected shape: {expected.shape}")
    try:
        ttnn_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_out = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V)
        torch_out = ttnn.to_torch(ttnn_out)
        diff = (torch_out.float() - expected.float()).abs()
        print(f"  output shape: {torch_out.shape}")
        print(f"  max diff: {diff.max().item():.6f}")
        print(f"  mean diff: {diff.mean().item():.6f}")
        passed, pcc_val = comp_pcc(torch_out.float(), expected.float())
        print(f"  PCC: {pcc_val}")
    except Exception as e:
        print(f"  RUNTIME ERROR: {type(e).__name__}: {e}")
finally:
    ttnn.close_device(device)
