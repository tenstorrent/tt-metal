import torch, ttnn, math
import ttnn.operations.scaled_dot_product_attention as sdpa_mod
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# Allow non-aligned through validate()
sdpa_mod.SUPPORTED["alignment"] = ["tile_aligned", "w_non_aligned", "h_non_aligned"]


def ref(Q, K, V, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    D = Qf.shape[-1]
    s = scale or 1.0 / math.sqrt(D)
    sc = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    w = torch.softmax(sc, dim=-1)
    return torch.matmul(w, Vf)


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


device = ttnn.open_device(device_id=0)
try:
    cases = [
        ("w_nonaligned D=50 Saligned", (1, 1, 32, 50), (1, 1, 32, 50)),
        ("h_nonaligned S=47 Daligned", (1, 1, 47, 64), (1, 1, 47, 64)),
        ("both S=50 D=50", (1, 1, 50, 50), (1, 1, 50, 50)),
        ("D=47 multihead", (1, 8, 64, 47), (1, 8, 64, 47)),
        ("cross S_q=100 S_kv=47 D=50", (1, 4, 100, 50), (1, 4, 47, 50)),
    ]
    for name, qs, ks in cases:
        torch.manual_seed(0)
        Q = torch.randn(qs, dtype=torch.bfloat16)
        K = torch.randn(ks, dtype=torch.bfloat16)
        V = torch.randn(ks, dtype=torch.bfloat16)
        exp = ref(Q, K, V)
        tQ = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tK = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tV = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        try:
            out = scaled_dot_product_attention(tQ, tK, tV)
            r = ttnn.to_torch(out)
            print(f"{name}: PCC={pcc(r,exp):.5f} maxabs={(r.float()-exp).abs().max().item():.4f}")
        except Exception as e:
            print(f"{name}: EXCEPTION {type(e).__name__}: {str(e)[:120]}")
finally:
    ttnn.close_device(device)
