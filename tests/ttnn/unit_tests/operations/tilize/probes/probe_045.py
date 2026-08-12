import torch, ttnn
from ttnn.operations.tilize.tilize import _dispatch

device = ttnn.open_device(device_id=0)
try:
    fails = []
    cases = [
        ((1, 1, 256, 512), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        ((1, 1, 128, 96), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),  # Wt=3 -> tail width
        ((1, 1, 64, 2048), ttnn.uint8, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        ((1, 1, 128, 256), ttnn.float32, ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ]
    for shape, dt, im, om in cases:
        if dt == ttnn.uint8:
            src = torch.randint(0, 256, shape, dtype=torch.uint8)
        elif dt == ttnn.float32:
            src = torch.randn(shape, dtype=torch.float32)
        else:
            src = torch.randn(shape).bfloat16()
        t = ttnn.from_torch(src, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=im)
        for name, lv in [
            ("base", dict()),
            ("vc", dict(per_core_vc=1)),
            ("row", dict(block_order=1)),
            ("both", dict(per_core_vc=1, block_order=1)),
        ]:
            out = _dispatch(t, om, use_multicore=True, levers=lv)
            got = ttnn.to_torch(out)
            ok = torch.equal(got, src)
            print(f"{shape} {dt} {name}: bit-exact={ok}")
            if not ok:
                fails.append((shape, dt, name, (got.float() - src.float()).abs().max().item()))
    print("FAILURES:", fails)
finally:
    ttnn.close_device(device)
