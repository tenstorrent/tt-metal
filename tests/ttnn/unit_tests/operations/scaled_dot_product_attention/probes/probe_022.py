import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

dev = ttnn.open_device(device_id=0)


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


try:
    torch.manual_seed(0)
    B, H, Hkv, S, D = 2, 8, 1, 128, 64
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, Hkv, S, D)
    v = torch.randn(B, Hkv, S, D)
    # references (head-broadcast for MQA)
    kb = k.repeat_interleave(H // Hkv, dim=1)
    vb = v.repeat_interleave(H // Hkv, dim=1)
    ref_causal = torch.nn.functional.scaled_dot_product_attention(q.float(), kb.float(), vb.float(), is_causal=True)
    tri = torch.zeros(B, 1, S, S)
    tri.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), 1), float("-inf"))
    ref_custom = torch.nn.functional.scaled_dot_product_attention(
        q.float(), kb.float(), vb.float(), attn_mask=tri.float()
    )
    print("ref custom vs causal identical?", torch.allclose(ref_causal, ref_custom, atol=1e-4))

    for dt in [ttnn.bfloat16, ttnn.bfloat8_b]:
        tq = ttnn.from_torch(q, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        tk = ttnn.from_torch(k, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        tv = ttnn.from_torch(v, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        tm = ttnn.from_torch(tri, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        o_causal = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv, is_causal=True))
        o_custom = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv, attn_mask=tm))
        print(
            f"{dt}: causal pcc={pcc(o_causal, ref_causal):.4f}  custom pcc={pcc(o_custom, ref_custom):.4f}  causal-vs-custom pcc={pcc(o_causal, o_custom):.4f}"
        )
finally:
    ttnn.close_device(dev)
