import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
_L1 = ttnn.BufferType.L1
_ROW = ttnn.ShardOrientation.ROW_MAJOR


def crs(e):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(*e))})


ok = True
try:
    grid4 = crs((3, 0))
    nd = ttnn.NdShardSpec(ttnn.Shape([2, 64, 64]), grid4, _ROW)
    mcnd = ttnn.MemoryConfig(_L1, nd)
    xn = torch.arange(4 * 128 * 128).reshape(4, 128, 128).float().bfloat16()
    tin = ttnn.from_torch(xn, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mcnd)
    rn = ttnn.to_torch(tilize(tin, memory_config=mcnd))
    en = torch.equal(xn, rn)
    ok &= en
    print("nd multishard [4,128,128]/[2,64,64]:", en, "max", (xn.float() - rn.float()).abs().max().item())
    # single-shard nd regression
    nds = ttnn.NdShardSpec(ttnn.Shape([1, 64, 64]), crs((1, 0)), _ROW)
    mcs = ttnn.MemoryConfig(_L1, nds)
    xs = torch.arange(2 * 64 * 64).reshape(2, 64, 64).float().bfloat16()
    ts = ttnn.from_torch(xs, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mcs)
    rs = ttnn.to_torch(tilize(ts, memory_config=mcs))
    es = torch.equal(xs, rs)
    ok &= es
    print("single-shard nd regression:", es)
finally:
    ttnn.close_device(device)
print("ALL_OK", ok)
assert ok
