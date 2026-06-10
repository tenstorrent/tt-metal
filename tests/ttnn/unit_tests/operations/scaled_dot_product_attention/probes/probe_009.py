"""Quick sanity check — minimal SDPA call with bf16 inputs after R4 changes."""
import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

torch.manual_seed(0)
B, H, S, D = 1, 1, 32, 32
Q = torch.randn(B, H, S, D, dtype=torch.bfloat16) * 0.3
K = torch.randn(B, H, S, D, dtype=torch.bfloat16) * 0.3
V = torch.randn(B, H, S, D, dtype=torch.bfloat16) * 0.3

device = ttnn.open_device(device_id=0)
try:
    q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(q, k, v)
    actual = ttnn.to_torch(out)
    print("OK shape:", tuple(actual.shape))

    # Reference
    qf, kf, vf = Q.float(), K.float(), V.float()
    scale = 1.0 / (D**0.5)
    scores = (qf @ kf.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    expected = (attn @ vf).to(torch.bfloat16)
    diff = (actual.float() - expected.float()).abs()
    print(f"max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}")
finally:
    ttnn.close_device(device)
