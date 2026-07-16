import math, torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)


def ref(q, k, v):
    D = q.shape[-1]
    s = 1.0 / math.sqrt(D)
    sc = torch.matmul(q.float(), k.float().transpose(-2, -1)) * s
    return torch.matmul(torch.softmax(sc, -1), v.float())


def run(shape, tag):
    torch.manual_seed(3)
    q = torch.randn(*shape, dtype=torch.bfloat16)
    k = torch.randn(*shape, dtype=torch.bfloat16)
    v = torch.randn(*shape, dtype=torch.bfloat16)
    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(tq, tk, tv)
    r = ttnn.to_torch(out).float()
    rf = ref(q, k, v)
    d = (r - rf).abs().max().item()
    print(f"{tag} {shape}: max diff {d:.4f}  {'OK' if d<0.2 else 'FAIL'}")


run((1, 1, 128, 64), "128x64_Dt2")
run((1, 1, 64, 64), "64x64_MNK2")
run((1, 1, 32, 64), "32x64")
run((1, 1, 128, 32), "128x32")
ttnn.close_device(device)
