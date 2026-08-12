# R7 probe F: is the uint8 narrow-stick failure a DRAM-alignment gap?
# Control = the same shape with an L1 source (16 B alignment).
import torch, ttnn
from ttnn.operations.tilize.tilize import _dispatch

dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(3)
    for shape in ((1, 1, 32, 32), (1, 1, 32, 96), (1, 1, 32, 64), (1, 1, 32, 128)):
        t = torch.randint(0, 251, shape, dtype=torch.int32)
        row_bytes = shape[-1]  # uint8
        print(f"=== uint8 {shape}  row_bytes={row_bytes}  %64={row_bytes % 64}")
        for mc, name in ((ttnn.DRAM_MEMORY_CONFIG, "dram_src"), (ttnn.L1_MEMORY_CONFIG, "l1_src")):
            x = ttnn.from_torch(t, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
            y = _dispatch(x, memory_config=mc)
            got = ttnn.to_torch(y).to(torch.int32)
            print(f"  {name:9s} mismatches={int((got != t).sum()):5d}/{t.numel()}")
            ttnn.deallocate(x)
            ttnn.deallocate(y)
finally:
    ttnn.close_device(dev)
