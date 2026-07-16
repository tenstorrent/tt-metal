import math, torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def ref(q, k, v, m, scale):
    q, k, v = q.float(), k.float(), v.float()
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    if m is not None:
        s = s + m.float()
    return torch.matmul(torch.softmax(s, dim=-1), v)


def run(tag, qs, ks, custom):
    torch.manual_seed(1234)
    dev = ttnn.open_device(device_id=0)
    try:
        q = torch.randn(qs)
        k = torch.randn(ks)
        v = torch.randn(ks)
        scale = 1.0 / math.sqrt(qs[-1])
        m = None
        tm = None
        if custom:
            m = torch.randn(qs[0], 1, qs[-2], ks[-2]) * 2.0
            tm = ttnn.from_torch(m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        r = ref(q, k, v, m, scale)
        tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        o = scaled_dot_product_attention(tq, tk, tv, attn_mask=tm, scale=scale)
        ot = ttnn.to_torch(o).float()
        pcc = torch.corrcoef(torch.stack([r.flatten(), ot.flatten()]))[0, 1].item()
        print(f"{tag}: PCC={pcc:.5f}")
    finally:
        ttnn.close_device(dev)


# sq_skv=12 WITHOUT a partial KV width of 3 (Skv_t=4 full chunk, sq_valid=3):
run("sqvalid3_skvfull4_custom", (1, 1, 96, 64), (1, 1, 128, 64), True)
