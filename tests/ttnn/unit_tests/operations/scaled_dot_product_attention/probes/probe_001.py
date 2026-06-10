import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)
try:
    B, H, Sq, Skv, D = 1, 1, 32, 256, 32
    q = torch.ones(B, H, Sq, D)
    k = torch.ones(B, H, Skv, D)
    v = torch.zeros(B, H, Skv, D)
    for n in range(Skv):
        v[:, :, n, :] = n / 100.0
    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv))
    # uniform softmax -> O = mean(v) = 1.275 everywhere
    print("expected 1.275; got min/max/mean:", out.min().item(), out.max().item(), out.mean().item())
    print(out[0, 0, :3, :4])
finally:
    ttnn.close_device(device)
