import torch, ttnn
from ttnn.operations.concat import concat
dev = ttnn.open_device(device_id=0)
try:
    for dim, shapes in [(-1, [(1,1,64,64),(1,1,64,80)]), (-1, [(1,1,32,50),(1,1,32,14)]), (1, [(1,64,144),(1,16,144),(1,4,144)])]:
        ts = [torch.randn(*s, dtype=torch.bfloat16) for s in shapes]
        dts = [ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16) for t in ts]
        out = ttnn.to_torch(concat(dts, dim=dim))
        ref = torch.cat(ts, dim=dim)
        print(dim, shapes, out.shape == ref.shape, torch.equal(out, ref), (out.float()-ref.float()).abs().max().item() if out.shape==ref.shape else None)
finally:
    ttnn.close_device(dev)
