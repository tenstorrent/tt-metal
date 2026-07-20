import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
_L1 = ttnn.BufferType.L1
_ROW = ttnn.ShardOrientation.ROW_MAJOR
_H = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def height_mc(g, s):
    return ttnn.MemoryConfig(_H, _L1, ttnn.ShardSpec(g, s, _ROW))


def crs(e):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(*e))})


ok = True
try:
    grid4 = crs((3, 0))
    # multi-shard same-spec nd [4,128,128]/[2,64,64] = 8 shards/4 cores
    nd = ttnn.NdShardSpec(ttnn.Shape([2, 64, 64]), grid4, _ROW)
    mcnd = ttnn.MemoryConfig(_L1, nd)
    xn = torch.arange(4 * 128 * 128).reshape(4, 128, 128).float().bfloat16()
    tin = ttnn.from_torch(xn, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mcnd)
    rn = ttnn.to_torch(tilize(tin, memory_config=mcnd))
    en = torch.equal(xn, rn)
    ok &= en
    print("multishard nd [4,128,128]/[2,64,64]:", en, (xn.float() - rn.float()).abs().max().item())
    # multi-shard same-spec legacy HEIGHT [1,1,512,64]/[128,64] on 2 cores = 4 shards/2 cores
    grid2 = crs((1, 0))
    mc2 = height_mc(grid2, (128, 64))
    x = torch.arange(512 * 64).reshape(1, 1, 512, 64).float().bfloat16()
    tih = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc2)
    rh = ttnn.to_torch(tilize(tih, memory_config=mc2))
    eh = torch.equal(x, rh)
    ok &= eh
    print("multishard legacy HEIGHT 4sh/2co:", eh, (x.float() - rh.float()).abs().max().item())
    # regression: single-shard nd still works
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
