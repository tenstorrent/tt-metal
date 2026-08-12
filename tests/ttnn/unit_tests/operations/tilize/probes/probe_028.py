import torch, ttnn
from ttnn.operations.tilize.tilize import _dispatch

dev = ttnn.open_device(device_id=0)
try:
    # words 0..3 of row 0 are bytes 0,1,2,3.. -> uint8: 0x03020100 = 50462976
    t = torch.arange(32 * 64, dtype=torch.int32).remainder(251).reshape(1, 1, 32, 64)
    for dt in (ttnn.uint16, ttnn.uint8):
        print(f"=== {dt}", flush=True)
        x = ttnn.from_torch(
            t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        y = _dispatch(x, levers=dict(stagger_reads=1))
        print("mismatches", int((ttnn.to_torch(y).to(torch.int32) != t).sum()), flush=True)
        ttnn.deallocate(x)
        ttnn.deallocate(y)
finally:
    ttnn.close_device(dev)
