import torch, ttnn
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)
try:
    for shape in (
        [1, 1, 2048, 2048],
        [1, 1, 32, 16384],
        [1, 1, 8192, 1024],
        [1, 1, 32, 64],
        [1, 1, 96, 128],
        [1, 1, 64, 96],
    ):
        for dt, td in ((ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)):
            t = torch.randn(shape).to(td)
            x = ttnn.from_torch(
                t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            assert torch.equal(ttnn.to_torch(tilize(x)), t), f"MISMATCH {shape} {dt}"
            print(f"{shape} {dt} exact")
    t = torch.randn([1, 1, 256, 256]).to(torch.bfloat16)
    x = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for kw in (
        dict(use_double_buffer=False),
        dict(use_multicore=False),
        dict(use_multicore=False, use_double_buffer=False),
    ):
        assert torch.equal(ttnn.to_torch(tilize(x, **kw)), t), f"MISMATCH {kw}"
        print(f"{kw} exact")
    print("ALL EXACT")
finally:
    ttnn.close_device(dev)
