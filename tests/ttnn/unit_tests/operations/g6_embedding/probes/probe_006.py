import torch, ttnn

# Every bf16 bit pattern once: does the tilize helper path keep each bit?
device = ttnn.open_device(device_id=0)
try:
    bits = torch.arange(65536, dtype=torch.int32)
    as_i16 = torch.where(bits >= 32768, bits - 65536, bits).to(torch.int16).reshape(64, 1024)
    x_bf16 = as_i16.view(torch.bfloat16)
    t = ttnn.from_torch(
        x_bf16, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    back_rm = ttnn.to_torch(t).contiguous().view(torch.int16)
    print("bf16 host->device->host RM identical:", torch.equal(back_rm, as_i16))
    tt = ttnn.tilize(t)
    got = ttnn.to_torch(tt).contiguous().view(torch.int16)
    diff = got != as_i16
    print("bf16 tilize: differing patterns:", int(diff.sum()))
    if diff.any():
        idx = diff.nonzero()
        sample = [(hex(int(as_i16[i, j]) & 0xFFFF), hex(int(got[i, j]) & 0xFFFF)) for i, j in idx[:12].tolist()]
        print("  samples (in -> out):", sample)
        ins = as_i16[diff].to(torch.int32) & 0xFFFF
        exp = (ins >> 7) & 0xFF
        print(
            "  differing with exp==0:",
            int((exp == 0).sum()),
            " exp==255:",
            int((exp == 255).sum()),
            " other:",
            int(((exp != 0) & (exp != 255)).sum()),
        )
    for name, dt in (("uint16", ttnn.uint16),):
        try:
            u = ttnn.from_torch(
                bits.reshape(64, 1024),
                dtype=dt,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tu = ttnn.tilize(u)
            gu = ttnn.to_torch(tu).to(torch.int64) & 0xFFFF
            d = gu != bits.reshape(64, 1024).to(torch.int64)
            print(f"{name} tilize: dtype out {tu.dtype}, differing patterns:", int(d.sum()))
            if d.any():
                idx = d.nonzero()
                print(
                    "  samples:",
                    [(hex(int(bits.reshape(64, 1024)[i, j])), hex(int(gu[i, j]))) for i, j in idx[:12].tolist()],
                )
        except Exception as e:
            print(f"{name} tilize failed:", repr(e)[:400])
finally:
    ttnn.close_device(device)
