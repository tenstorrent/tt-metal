import torch, ttnn

torch.manual_seed(0)


def torch_ref(Q, K, V, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    return torch.nn.functional.scaled_dot_product_attention(Qf, Kf, Vf, scale=scale)


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if a.std() == 0 or b.std() == 0:
        return float(torch.allclose(a, b))
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# Check zero-padding assumption: create non-aligned tile tensor, read back padded region
t = torch.arange(1 * 1 * 47 * 50).reshape(1, 1, 47, 50).float()
tt = ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
# read raw padded via to_torch with pad? Just check logical round trips
back = ttnn.to_torch(tt).float()
print(
    "roundtrip shape",
    back.shape,
    "max diff logical",
    (back[..., :47, :50] - t[..., :47, :50]).abs().max().item() if False else "n/a",
)

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

shapes = [
    ("w_D50_Saligned", (1, 1, 32, 50)),
    ("w_D47_multih", (1, 8, 64, 47)),
    ("h_S47", (1, 1, 47, 64)),
    ("h_S100_batch", (2, 4, 100, 64)),
    ("both_50_50", (1, 1, 50, 50)),
]
for name, s in shapes:
    B, H, S, D = s
    Q = torch.randn(s, dtype=torch.bfloat16)
    K = torch.randn(s, dtype=torch.bfloat16)
    V = torch.randn(s, dtype=torch.bfloat16)
    exp = torch_ref(Q, K, V)
    td = lambda x: ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    try:
        o = scaled_dot_product_attention(td(Q), td(K), td(V))
        out = ttnn.to_torch(o).float()
        print(f"{name:20s} {s} -> PCC={pcc(exp,out):.5f} shape={tuple(out.shape)}")
    except Exception as e:
        print(f"{name:20s} {s} -> ERROR {type(e).__name__}: {str(e)[:80]}")
