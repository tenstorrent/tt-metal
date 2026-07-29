import torch, ttnn
import ttnn.operations.tilize as T
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)
try:
    T.SUPPORTED["rank"] = [2, 3, 4, 5, 6]
    for shape in [(2, 2, 2, 32, 64), (2, 2, 2, 2, 32, 32)]:
        t = torch.randn(shape).bfloat16()
        x = ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for mc in (False, True):
            y = tilize(x, use_multicore=mc)
            out = ttnn.to_torch(y)
            ok = torch.equal(out, t)
            print(f"rank={len(shape)} shape={shape} multicore={mc} layout={y.layout} exact={ok}")
finally:
    ttnn.close_device(dev)
