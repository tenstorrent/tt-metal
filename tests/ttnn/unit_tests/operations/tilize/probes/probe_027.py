import torch, ttnn
from ttnn.operations.tilize.tilize import _dispatch

dev = ttnn.open_device(device_id=0)
try:
    t = torch.arange(32 * 64, dtype=torch.int32).remainder(251).reshape(1, 1, 32, 64)
    x = ttnn.from_torch(
        t, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    y = _dispatch(x)
    print("mismatches", int((ttnn.to_torch(y).to(torch.int32) != t).sum()), flush=True)
finally:
    ttnn.close_device(dev)
