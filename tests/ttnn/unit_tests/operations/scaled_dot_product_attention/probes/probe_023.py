import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    S, D = 64, 64
    q = torch.randn(1, 1, S, D)
    k = torch.randn(1, 1, S, D)
    v = torch.randn(1, 1, S, D)
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
    tri = torch.zeros(1, 1, S, S)
    tri.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), 1), float("-inf"))
    tq = ttnn.from_torch(q, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    tm = ttnn.from_torch(tri, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    oc = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv, is_causal=True)).float()
    ou = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv, attn_mask=tm)).float()
    print("causal: nan?", torch.isnan(oc).any().item(), "inf?", torch.isinf(oc).any().item())
    print("causal row0 [:4]", oc[0, 0, 0, :4].tolist())
    print("custom row0 [:4]", ou[0, 0, 0, :4].tolist())
    print("ref    row0 [:4]", ref[0, 0, 0, :4].tolist())
    # per-row abs error, to find where causal diverges
    e_causal = (oc - ref).abs().mean(dim=-1)[0, 0]  # (S,)
    e_custom = (ou - ref).abs().mean(dim=-1)[0, 0]
    print("causal per-row err rows 0,1,2,31,32,63:", [round(e_causal[i].item(), 3) for i in [0, 1, 2, 31, 32, 63]])
    print("custom per-row err rows 0,1,2,31,32,63:", [round(e_custom[i].item(), 3) for i in [0, 1, 2, 31, 32, 63]])
finally:
    ttnn.close_device(dev)
