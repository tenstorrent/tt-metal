import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

dev = ttnn.open_device(device_id=0)
try:
    # All-ones causal self-attn: query i attends uniformly to keys 0..i, output = mean(ones) = 1.
    q = torch.ones(1, 1, 64, 64, dtype=torch.bfloat16)
    k = torch.ones(1, 1, 64, 64, dtype=torch.bfloat16)
    v = torch.ones(1, 1, 64, 64, dtype=torch.bfloat16)

    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    out = scaled_dot_product_attention(tq, tk, tv, is_causal=True)
    res = ttnn.to_torch(out).float()
    print("shape", list(out.shape))
    print("min", res.min().item(), "max", res.max().item(), "mean", res.mean().item())
    print("all close to 1?", torch.allclose(res, torch.ones_like(res), atol=0.05))

    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
    print("max diff vs torch", (res - ref).abs().max().item())
finally:
    ttnn.close_device(dev)
