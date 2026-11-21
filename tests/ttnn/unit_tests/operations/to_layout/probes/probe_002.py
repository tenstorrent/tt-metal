import torch, ttnn
from ttnn.operations.to_layout import to_layout
dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    for shape in [(1,1,32,64), (1,1,50,128)]:
        x = torch.randn(shape, dtype=torch.bfloat16)
        for src, dst in [(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT), (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT), (ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT), (ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT)]:
            t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=src, device=dev)
            o = to_layout(t, dst)
            y = ttnn.to_torch(o)
            print(shape, src, "->", dst, o.layout, "exact:", torch.equal(y, x), flush=True)
finally:
    ttnn.close_device(dev)
