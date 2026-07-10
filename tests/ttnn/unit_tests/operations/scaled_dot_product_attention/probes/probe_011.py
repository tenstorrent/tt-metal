import torch, ttnn

torch.manual_seed(0)
dev = ttnn.open_device(device_id=0)


def torch_ref(Q, K, V, mask=None, scale=None):
    Hq, Hk = Q.shape[1], K.shape[1]
    Kf, Vf = K.float(), V.float()
    if Hq != Hk:
        r = Hq // Hk
        Kf = Kf.repeat_interleave(r, 1)
        Vf = Vf.repeat_interleave(r, 1)
    m = mask.float() if mask is not None else None
    return torch.nn.functional.scaled_dot_product_attention(Q.float(), Kf, Vf, attn_mask=m, scale=scale)


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if a.std() == 0 or b.std() == 0:
        return float(torch.allclose(a, b))
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def relrms(ref, got):
    ref = ref.flatten().float()
    got = got.flatten().float()
    return (((got - ref) ** 2).mean().sqrt() / (ref.std() + 1e-12)).item()


def cmask(B, Sq, Skv, dt):
    m = torch.zeros(B, 1, Sq, Skv, dtype=dt)
    m.masked_fill_(torch.triu(torch.ones(Sq, Skv, dtype=torch.bool), 1), float("-inf"))
    return m


from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

cases = [
    ("aligned_noreg", (1, 2, 128, 64), (1, 2, 128, 64), False),
    ("w_D50", (1, 1, 32, 50), (1, 1, 32, 50), False),
    ("h_S47", (1, 1, 47, 64), (1, 1, 47, 64), False),
    ("h_S100_batch", (2, 4, 100, 64), (2, 4, 100, 64), False),
    ("both_50_50", (1, 1, 50, 50), (1, 1, 50, 50), False),
    ("both_33_50_mh", (1, 12, 33, 50), (1, 12, 33, 50), False),
    ("h_S47_gqa", (1, 8, 47, 64), (1, 2, 47, 64), False),
    ("h_S47_mqa", (1, 8, 47, 64), (1, 1, 47, 64), False),
    ("cross_100_47_50", (1, 4, 100, 50), (1, 4, 47, 50), False),
    ("h_S47_mask", (1, 1, 47, 64), (1, 1, 47, 64), True),
    ("both_cross_mask", (1, 4, 100, 50), (1, 4, 47, 50), True),
]
fails = 0
try:
    for name, qs, ks, use_mask in cases:
        Q = torch.randn(qs, dtype=torch.bfloat16)
        K = torch.randn(ks, dtype=torch.bfloat16)
        V = torch.randn(ks, dtype=torch.bfloat16)
        tm = cmask(qs[0], qs[2], ks[2], torch.bfloat16) if use_mask else None
        exp = torch_ref(Q, K, V, mask=tm)
        td = lambda x: ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        o = scaled_dot_product_attention(td(Q), td(K), td(V), attn_mask=td(tm) if tm is not None else None)
        out = ttnn.to_torch(o).float()
        p, r = pcc(exp, out), relrms(exp, out)
        ok = p >= 0.995 and r <= 0.05
        if not ok:
            fails += 1
        print(f"{'OK' if ok else 'FAIL':4s} {name:16s} q={qs} k={ks} mask={use_mask} PCC={p:.5f} relRMS={r:.4f}")
    print("TOTAL_FAILS", fails)
finally:
    ttnn.close_device(dev)
