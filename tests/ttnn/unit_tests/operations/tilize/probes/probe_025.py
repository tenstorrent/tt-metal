# R7 probe C: reader-vs-compute discriminator for uint8. Position-encoded input;
# print the first 4 L1 words of the input block (compute side) and of the packed
# output tile (writer side).  uint16 runs first as the known-good control.
import torch, ttnn
from ttnn.operations.tilize.tilize import _dispatch

dev = ttnn.open_device(device_id=0)
try:
    t = torch.arange(32 * 64, dtype=torch.int32).remainder(251).reshape(1, 1, 32, 64)
    for dt in (ttnn.uint16, ttnn.uint8):
        print(f"=== {dt}")
        x = ttnn.from_torch(
            t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        y = _dispatch(x)
        out = ttnn.to_torch(y).to(torch.int32)
        print("mismatches", int((out != t).sum()))
        ttnn.deallocate(x)
        ttnn.deallocate(y)
finally:
    ttnn.close_device(dev)
