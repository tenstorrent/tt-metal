import torch, ttnn
from ttnn.operations.g6_reshape import g6_reshape, describe_plan

dev = ttnn.open_device(device_id=0)
cases = [
    ((1, 4096), (4096, 1), ttnn.int32),
    ((16,), (16, 1), ttnn.uint32),
    ((8, 1), (8,), ttnn.int32),
    ((8, 1), (1, 1, 1, 8), ttnn.uint32),
    ((1, 1, 1, 32), (32, 1), ttnn.uint32),
    ((16,), (4, 4), ttnn.int32),
    ((1, 31), (31, 1), ttnn.int32),
    ((1, 3844), (3844, 1), ttnn.int32),
    ((32, 64), (64, 32), ttnn.bfloat16),
    ((3, 5, 7), (7, 15), ttnn.bfloat16),
    ((4096, 1), (1, 4096), ttnn.int32),
    ((2, 33, 40), (66, 40), ttnn.float32),
    ((1, 5120), (5120,), ttnn.bfloat16),
]
tmap = {ttnn.int32: torch.int32, ttnn.uint32: torch.uint32, ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
for i, o, dt in cases:
    if dt in (ttnn.int32, ttnn.uint32):
        t = torch.randint(0, 2**31 - 1, i, dtype=torch.int64).to(tmap[dt])
    else:
        t = torch.randn(i).to(tmap[dt])
    x = ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    p = describe_plan(x, list(o))
    y = ttnn.to_torch(g6_reshape(x, list(o)))
    ref = t.reshape(o)
    ok = y.shape == ref.shape and torch.equal(
        y.to(torch.int64) if dt in (ttnn.int32, ttnn.uint32) else y.float(),
        ref.to(torch.int64) if dt in (ttnn.int32, ttnn.uint32) else ref.float(),
    )
    print(i, o, dt, "OK" if ok else "FAIL", p["num_cores"], p["block_atoms"], p["l1_bytes_per_core"])
ttnn.close_device(dev)
