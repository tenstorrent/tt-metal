import torch, ttnn
from ttnn.operations.concat.concat import _dispatch
dev = ttnn.open_device(device_id=0)
cases = [(-1, [(1,1,32,64)]), (-2, [(4,128,47)]), (-1, [(1,1,32,32)]*5), (-1, [(1,1,32,50),(1,1,32,14),(1,1,32,20),(1,1,32,7),(1,1,32,40)]),
         (-1, [(32,64),(32,64)]), (-1, [(32,17),(32,47)]), (-2, [(17,64),(17,64),(30,64)])]
for dim, shapes in cases:
    torch.manual_seed(0)
    host = [torch.randn(*s, dtype=torch.bfloat16) for s in shapes]
    d = [ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16) for t in host]
    try:
        out = ttnn.to_torch(_dispatch(d, dim))
        print("PROBE", dim, shapes, "exact" if torch.equal(out, torch.cat(host, dim)) else "MISMATCH")
    except Exception as e:
        print("PROBE", dim, shapes, "ERROR", type(e).__name__, str(e)[:200])
ttnn.close_device(dev)
