import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("GRID", g.x, g.y)
    SPECIAL = (0x8000, 0x0001, 0x8008, 0x7F80, 0xFF80, 0x7FC0, 0xFFFF, 0x7F81, 0x007F, 0x0080)
    for W in (64, 1280):
        torch.manual_seed(0)
        t = torch.randn(32, W, dtype=torch.bfloat16)
        bits = torch.tensor([b - 0x10000 if b >= 0x8000 else b for b in SPECIAL], dtype=torch.int16)
        pat = bits[(torch.arange(W).unsqueeze(0) + torch.arange(32).unsqueeze(1)) % len(bits)]
        v = t.view(torch.int16)
        v[::2] = pat[::2]
        rm = ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tl = ttnn.tilize(rm)
        back = ttnn.to_torch(tl).contiguous().view(torch.int16)
        want = t.contiguous().view(torch.int16)
        diff = back != want
        print("W", W, "mismatches", int(diff.sum()), "of", diff.numel())
        if diff.any():
            seen = {}
            for i in diff.nonzero()[:2000]:
                a = int(want[tuple(i)]) & 0xFFFF
                b = int(back[tuple(i)]) & 0xFFFF
                seen.setdefault(hex(a), set()).add(hex(b))
            print({k: sorted(v)[:4] for k, v in seen.items()})
finally:
    ttnn.close_device(device)
