import torch, ttnn, math
from ttnn.operations import scaled_dot_product_attention as sdpa_mod
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# widen SUPPORTED so validate() lets non-aligned through
sdpa_mod.SUPPORTED["alignment"] = ["tile_aligned", "w_non_aligned", "h_non_aligned"]


def ref(Q, K, V, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    s = scale or 1.0 / math.sqrt(Qf.shape[-1])
    sc = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    w = torch.softmax(sc, dim=-1)
    return torch.matmul(w, Vf)


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


dev = ttnn.GetDefaultDevice()
torch.manual_seed(0)
for shp in [(1, 1, 32, 50), (1, 1, 47, 64), (1, 1, 50, 50)]:
    B, H, S, D = shp
    Q = torch.randn(shp, dtype=torch.bfloat16)
    K = torch.randn(shp, dtype=torch.bfloat16)
    V = torch.randn(shp, dtype=torch.bfloat16)
    exp = ref(Q, K, V)
    qt = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    kt = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    vt = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    try:
        ot = scaled_dot_product_attention(qt, kt, vt)
        out = ttnn.to_torch(ot)
        print(shp, "out.shape", list(out.shape), "PCC", round(pcc(out, exp), 5))
    except Exception as e:
        print(shp, "EXC", type(e).__name__, str(e)[:120])
