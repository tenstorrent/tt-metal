import torch, ttnn
from ttnn.operations.tilize.tilize import _dispatch

dev = ttnn.open_device(device_id=0)
try:
    tf = torch.randn((1, 1, 64, 128), dtype=torch.float32)
    x = ttnn.from_torch(
        tf, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    y = _dispatch(x)
    print("MISMATCH", int((ttnn.to_torch(y) != tf).sum()))
finally:
    ttnn.close_device(dev)
