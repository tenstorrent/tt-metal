import math, torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def ref(Q, K, V, mask=None, scale=None):
    s = scale if scale is not None else 1.0 / math.sqrt(Q.shape[-1])
    sc = Q.float() @ K.float().transpose(-2, -1) * s
    if mask is not None:
        sc = sc + mask.float()
    return torch.softmax(sc, -1) @ V.float()


device = ttnn.open_device(device_id=0)
try:
    for shape in [(1, 1, 47, 64), (1, 1, 32, 50), (1, 1, 50, 50)]:
        torch.manual_seed(0)
        Q = torch.randn(shape, dtype=torch.bfloat16)
        K = torch.randn(shape, dtype=torch.bfloat16)
        V = torch.randn(shape, dtype=torch.bfloat16)
        tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv)).float()
        exp = ref(Q, K, V)
        d = out - exp
        rms = d.pow(2).mean().sqrt() / exp.std()
        pcc = torch.corrcoef(torch.stack([out.flatten(), exp.flatten()]))[0, 1]
        print(f"{shape}: pcc={pcc:.6f} rel_rms={rms:.5f} maxabs={d.abs().max():.5f}")
finally:
    ttnn.close_device(device)
