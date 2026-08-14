import torch, ttnn
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)
try:
    for shape in ([1, 1, 2048, 2048], [1, 1, 32, 16384], [1, 1, 8192, 1024], [1, 1, 32, 64], [1, 1, 96, 128]):
        for dt, td in ((ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)):
            t = torch.randn(shape).to(td)
            x = ttnn.from_torch(
                t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            got = ttnn.to_torch(tilize(x))
            ok = torch.equal(got, t)
            print(f"{shape} {dt} exact={ok}")
            assert ok, f"MISMATCH {shape} {dt}"
    print("ALL EXACT")
finally:
    ttnn.close_device(dev)
