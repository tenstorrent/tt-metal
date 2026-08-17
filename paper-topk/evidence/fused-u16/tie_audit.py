import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    k, rows, n, vl = 1536, 2, 102400, 56320
    x = torch.randn(rows, n, dtype=torch.bfloat16)
    xd = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    xs = ttnn.from_torch(x[:, :vl].contiguous(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    b = ttnn.to_torch(ttnn.experimental.topk_large_indices(xd, k=k, valid_length=vl), dtype=torch.uint32).to(
        torch.int64
    )
    s = ttnn.to_torch(ttnn.experimental.topk_large_indices(xs, k=k), dtype=torch.uint32).to(torch.int64)
    xf = x.to(torch.float32)
    total_diff = nontie = 0
    for r in range(rows):
        d = (b[r] != s[r]).nonzero().flatten()
        total_diff += len(d)
        vb, vs = xf[r, b[r][d]], xf[r, s[r][d]]
        nontie += int((vb != vs).sum())
        gb, _ = torch.sort(xf[r, b[r]], descending=True)
        gs, _ = torch.sort(xf[r, s[r]], descending=True)
        assert torch.equal(gb, gs), f"row {r}: value multiset mismatch!"
    print(f"diffs={total_diff} non_tie={nontie} value_multisets=BIT-EXACT")
finally:
    ttnn.close_device(device)
