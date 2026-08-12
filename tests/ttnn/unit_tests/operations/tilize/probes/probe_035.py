# R7 probe G: the A5b narrow-stick reader — uint8 at every W phase, DRAM and L1.
import torch, ttnn
from ttnn.operations.tilize.tilize import _dispatch

dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(3)
    for shape in ((1, 1, 32, 32), (1, 1, 32, 96), (1, 1, 32, 64), (1, 1, 32, 128), (1, 1, 64, 160), (2, 3, 32, 1056)):
        t = torch.randint(0, 251, shape, dtype=torch.int32)
        print(f"=== uint8 {shape}  row_bytes={shape[-1]}  %64={shape[-1] % 64}")
        for mc, name in ((ttnn.DRAM_MEMORY_CONFIG, "dram_src"), (ttnn.L1_MEMORY_CONFIG, "l1_src")):
            x = ttnn.from_torch(t, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
            y = _dispatch(x, memory_config=mc)
            got = ttnn.to_torch(y).to(torch.int32)
            print(f"  {name:9s} mismatches={int((got != t).sum()):5d}/{t.numel()}")
            ttnn.deallocate(x)
            ttnn.deallocate(y)
finally:
    ttnn.close_device(dev)
