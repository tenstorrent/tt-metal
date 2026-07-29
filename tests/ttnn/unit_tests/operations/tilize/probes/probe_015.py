import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
try:
    for shape, mc in [((1, 1, 32, 16384), True), ((1, 1, 512, 512), False)]:
        t = torch.randn(shape).bfloat16()
        d = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out = tilize(d, use_multicore=mc)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)
        ttnn.deallocate(d)
finally:
    ttnn.close_device(device)
