import math, torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)


def ref(q, k, v):
    D = q.shape[-1]
    s = 1.0 / math.sqrt(D)
    sc = torch.matmul(q.float(), k.float().transpose(-2, -1)) * s
    return torch.matmul(torch.softmax(sc, -1), v.float())


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def run(qs, ks, tag):
    torch.manual_seed(3)
    q = torch.randn(*qs, dtype=torch.bfloat16)
    k = torch.randn(*ks, dtype=torch.bfloat16)
    v = torch.randn(*ks, dtype=torch.bfloat16)
    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(tq, tk, tv)
    r = ttnn.to_torch(out).float()
    rf = ref(q, k, v)
    print(f"{tag} q{qs} k{ks}: max diff {(r-rf).abs().max().item():.4f} pcc {pcc(r,rf):.5f}")


run((1, 1, 64, 32), (1, 1, 256, 32), "2kvchunk_Sq2")
run((1, 1, 128, 64), (1, 1, 256, 64), "2kvchunk_Dt2")
run((1, 2, 512, 64), (1, 2, 512, 64), "longer_self")
ttnn.close_device(device)
