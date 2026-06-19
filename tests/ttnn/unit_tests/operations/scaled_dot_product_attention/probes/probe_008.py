import torch, ttnn, math
import ttnn.operations.scaled_dot_product_attention as sdpa_mod
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# Monkeypatch SUPPORTED so validate() permits non-aligned alignment values.
sdpa_mod.SUPPORTED["alignment"] = ["tile_aligned", "w_non_aligned", "h_non_aligned"]


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return 1.0 if denom == 0 else float((a * b).sum() / denom)


def ref(Q, K, V, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    D = Qf.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(D)
    sc = (Qf @ Kf.transpose(-2, -1)) * s
    return torch.softmax(sc, dim=-1) @ Vf


def run(name, qshape, kshape):
    torch.manual_seed(0)
    Q = torch.randn(qshape, dtype=torch.bfloat16)
    K = torch.randn(kshape, dtype=torch.bfloat16)
    V = torch.randn(kshape, dtype=torch.bfloat16)
    exp = ref(Q, K, V)
    qt = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    try:
        out = scaled_dot_product_attention(qt, kt, vt)
        res = ttnn.to_torch(out)
        p = pcc(res, exp)
        maxd = float((res.float() - exp.float()).abs().max())
        print(f"{name}: shape={qshape} PCC={p:.5f} max_abs={maxd:.4f} out_shape={tuple(res.shape)}")
    except Exception as e:
        print(f"{name}: shape={qshape} EXCEPTION {type(e).__name__}: {e}")


device = ttnn.open_device(device_id=0)
try:
    run("aligned_baseline", (1, 1, 32, 64), (1, 1, 32, 64))
    run("w_non_aligned_D50", (1, 1, 32, 50), (1, 1, 32, 50))
    run("w_non_aligned_D47_mh", (1, 8, 64, 47), (1, 8, 64, 47))
    run("h_non_aligned_S47", (1, 1, 47, 64), (1, 1, 47, 64))
    run("h_non_aligned_S100", (2, 4, 100, 64), (2, 4, 100, 64))
    run("both_50_50", (1, 1, 50, 50), (1, 1, 50, 50))
finally:
    ttnn.close_device(device)
