import torch, ttnn, math
import ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention as submod
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# Temporarily drop the bf8b+h_non_aligned exclusion to probe whether it works.
submod.EXCLUSIONS = [e for e in submod.EXCLUSIONS if e != {"dtype": ttnn.bfloat8_b, "alignment": "h_non_aligned"}]


def ref(Q, K, V, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    s = scale or 1.0 / math.sqrt(Qf.shape[-1])
    sc = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    return torch.matmul(torch.softmax(sc, dim=-1), Vf)


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


dev = ttnn.GetDefaultDevice()
torch.manual_seed(0)
# h_non_aligned: S non-aligned, D aligned. (also include both-non-aligned check)
for shp in [(1, 1, 47, 64), (1, 4, 47, 64), (2, 4, 100, 64)]:
    Q = torch.randn(shp, dtype=torch.bfloat16)
    K = torch.randn(shp, dtype=torch.bfloat16)
    V = torch.randn(shp, dtype=torch.bfloat16)
    exp = ref(Q, K, V)
    qt = ttnn.from_torch(Q, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    kt = ttnn.from_torch(K, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    vt = ttnn.from_torch(V, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    try:
        ot = scaled_dot_product_attention(qt, kt, vt)
        out = ttnn.to_torch(ot)
        print(shp, "bf8b h_non_aligned PCC", round(pcc(out, exp), 5))
    except Exception as e:
        print(shp, "EXC", type(e).__name__, str(e)[:100])
