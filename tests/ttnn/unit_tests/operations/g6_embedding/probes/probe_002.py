import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    SPECIAL = (0x8000, 0x0001, 0x8008, 0x7F80, 0xFF80, 0x7FC0, 0xFFFF, 0x7F81, 0x007F, 0x0080)
    for W in (64, 1280):
        torch.manual_seed(0)
        t = torch.randn(32, W, dtype=torch.bfloat16)
        bits = torch.tensor([b - 0x10000 if b >= 0x8000 else b for b in SPECIAL], dtype=torch.int16)
        pat = bits[(torch.arange(W).unsqueeze(0) + torch.arange(32).unsqueeze(1)) % len(bits)]
        v = t.view(torch.int16)
        v[::2] = pat[::2]
        want = t.contiguous().view(torch.int16)
        # as uint16 integers: values 0..65535 carried in int32 for the host conversion
        u = want.to(torch.int32) & 0xFFFF
        try:
            rm = ttnn.from_torch(
                u, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            chk = ttnn.to_torch(rm).to(torch.int32) & 0xFFFF
            print("host roundtrip uint16 ok:", torch.equal(chk, u))
            tl = ttnn.tilize(rm)
            back = ttnn.to_torch(tl).to(torch.int32) & 0xFFFF
            diff = back != u
            print("uint16 tilize W", W, "mismatches", int(diff.sum()), "of", diff.numel())
            if diff.any():
                seen = {}
                for i in diff.nonzero()[:2000]:
                    seen.setdefault(hex(int(u[tuple(i)])), set()).add(hex(int(back[tuple(i)])))
                print({k: sorted(v)[:4] for k, v in list(seen.items())[:12]})
        except Exception as e:
            print("uint16 path error:", repr(e)[:300])
finally:
    ttnn.close_device(device)
