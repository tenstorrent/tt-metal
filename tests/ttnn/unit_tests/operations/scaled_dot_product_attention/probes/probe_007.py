import math
import torch
import ttnn

device = ttnn.open_device(device_id=0)


def torch_sdpa(Q, K, V, mask=None):
    qf = Q.to(torch.float32)
    kf = K.to(torch.float32)
    vf = V.to(torch.float32)
    s = 1.0 / math.sqrt(qf.shape[-1])
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * s
    if mask is not None:
        scores = scores + mask.to(torch.float32)
    return torch.matmul(torch.softmax(scores, dim=-1), vf).to(Q.dtype)


try:
    from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
    from tests.ttnn.utils_for_testing import comp_pcc

    # Case A: broadcast mask + S_kv non-aligned, S_q non-aligned, H=4
    B, H, S_q, S_kv, D = 1, 4, 47, 47, 64
    print(f"\n=== A: B={B} H={H} S_q={S_q} S_kv={S_kv} D={D}, BROADCAST mask ===")
    torch.manual_seed(0)
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    mask_bcast = torch.randn(B, 1, S_q, S_kv, dtype=torch.bfloat16)
    expected = torch_sdpa(Q, K, V, mask=mask_bcast)
    ttnn_out = scaled_dot_product_attention(
        ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        attention_mask=ttnn.from_torch(mask_bcast, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )
    actual = ttnn.to_torch(ttnn_out)
    diff = (actual.float() - expected.float()).abs()
    p, pcc = comp_pcc(actual.float(), expected.float())
    print(f"  PCC={pcc} max_diff={diff.max().item():.4f}")

    # Case B: per-head mask + S_kv non-aligned, S_q non-aligned, H=4
    print(f"\n=== B: B={B} H={H} S_q={S_q} S_kv={S_kv} D={D}, PER-HEAD mask ===")
    torch.manual_seed(0)
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    mask_perhead = torch.randn(B, H, S_q, S_kv, dtype=torch.bfloat16)
    expected = torch_sdpa(Q, K, V, mask=mask_perhead)
    ttnn_out = scaled_dot_product_attention(
        ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        attention_mask=ttnn.from_torch(mask_perhead, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )
    actual = ttnn.to_torch(ttnn_out)
    diff = (actual.float() - expected.float()).abs()
    p, pcc = comp_pcc(actual.float(), expected.float())
    print(f"  PCC={pcc} max_diff={diff.max().item():.4f}")

    # Case C: per-head mask + ALIGNED shapes — should pass (regression)
    B, H, S_q, S_kv, D = 1, 4, 32, 32, 64
    print(f"\n=== C: B={B} H={H} S_q={S_q} S_kv={S_kv} D={D}, PER-HEAD mask, ALIGNED ===")
    torch.manual_seed(0)
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    mask_perhead = torch.randn(B, H, S_q, S_kv, dtype=torch.bfloat16)
    expected = torch_sdpa(Q, K, V, mask=mask_perhead)
    ttnn_out = scaled_dot_product_attention(
        ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        attention_mask=ttnn.from_torch(mask_perhead, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )
    actual = ttnn.to_torch(ttnn_out)
    diff = (actual.float() - expected.float()).abs()
    p, pcc = comp_pcc(actual.float(), expected.float())
    print(f"  PCC={pcc} max_diff={diff.max().item():.4f}")

    # Case D: per-head + ONLY S_kv non-aligned (S_q aligned)
    B, H, S_q, S_kv, D = 1, 4, 32, 47, 64
    print(f"\n=== D: B={B} H={H} S_q={S_q} S_kv={S_kv} D={D}, PER-HEAD mask, only S_kv non-aligned ===")
    torch.manual_seed(0)
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    mask_perhead = torch.randn(B, H, S_q, S_kv, dtype=torch.bfloat16)
    expected = torch_sdpa(Q, K, V, mask=mask_perhead)
    ttnn_out = scaled_dot_product_attention(
        ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        attention_mask=ttnn.from_torch(mask_perhead, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )
    actual = ttnn.to_torch(ttnn_out)
    diff = (actual.float() - expected.float()).abs()
    p, pcc = comp_pcc(actual.float(), expected.float())
    print(f"  PCC={pcc} max_diff={diff.max().item():.4f}")

finally:
    ttnn.close_device(device)
