import math, torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)
try:
    B, H, Sq, Skv, D = 1, 1, 32, 256, 32
    q = torch.zeros(B, H, Sq, D)
    q[..., 0] = 1.0
    k = torch.zeros(B, H, Skv, D)
    k[:, :, 128:, 0] = math.sqrt(D)  # scores: 0 for n<128, 1.0 for n>=128 after scale
    v = torch.zeros(B, H, Skv, D)
    v[:, :, 128:, :] = 1.0
    # expected: e/(1+e) = 0.731059 everywhere
    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv))
    print("expected 0.731; got min/max/mean:", out.min().item(), out.max().item(), out.mean().item())
finally:
    ttnn.close_device(device)
