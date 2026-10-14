import torch, ttnn
from ttnn.operations.tilize.tilize import tilize

dev = ttnn.open_device(device_id=0)
torch.manual_seed(0)
for shape in [(1, 1, 32, 32), (1, 1, 64, 64)]:
    xf = torch.randn(shape).to(torch.float8_e4m3fn).to(torch.float32)
    t = ttnn.from_torch(xf, dtype=ttnn.fp8_e4m3, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    # host-free check of the input itself: typecast via to_layout reference not allowed; just look at tilize output
    y = ttnn.to_torch(tilize(t, dtype=ttnn.float32)).float()
    bad = (xf != y).nonzero()
    print("RES", shape, "nbad", len(bad))
    for idx in bad[:12]:
        i = tuple(idx.tolist())
        print("RES   ", i, xf[i].item(), y[i].item())
ttnn.close_device(dev)
