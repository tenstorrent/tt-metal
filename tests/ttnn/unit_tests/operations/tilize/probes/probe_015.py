import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
_L1 = ttnn.BufferType.L1
_ROW = ttnn.ShardOrientation.ROW_MAJOR
_H = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def height_mc(grid, shard):
    return ttnn.MemoryConfig(_H, _L1, ttnn.ShardSpec(grid, shard, _ROW))


def crs(e):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(*e))})


ok = True
try:
    # ---- Crossover dir2: interleaved DRAM RM -> HEIGHT sharded TILE ----
    grid4 = crs((3, 0))  # 4 cores
    shape = [1, 1, 512, 64]
    mc = height_mc(grid4, (128, 64))  # 4 shards of 128x64, 1/core
    x = torch.arange(512 * 64).reshape(1, 1, 512, 64).float().bfloat16()
    ti = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    to = tilize(ti, memory_config=mc)
    r = ttnn.to_torch(to)
    e = torch.equal(x, r)
    ok &= e
    print("dir2 interleaved->HEIGHT sharded identity:", e, "max_diff", (x.float() - r.float()).abs().max().item())

    # ---- Crossover dir1: HEIGHT sharded RM -> interleaved DRAM TILE ----
    ti2 = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    to2 = tilize(ti2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    r2 = ttnn.to_torch(to2)
    e2 = torch.equal(x, r2)
    ok &= e2
    print("dir1 HEIGHT sharded->interleaved identity:", e2, "max_diff", (x.float() - r2.float()).abs().max().item())

    # ---- Multi-shard same-spec even: nd [4,128,128]/[2,64,64] = 8 shards / 4 cores ----
    nd = ttnn.NdShardSpec(ttnn.Shape([2, 64, 64]), grid4, _ROW)
    mcnd = ttnn.MemoryConfig(_L1, nd)
    xn = torch.arange(4 * 128 * 128).reshape(4, 128, 128).float().bfloat16()
    tin = ttnn.from_torch(xn, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mcnd)
    ton = tilize(tin, memory_config=mcnd)
    rn = ttnn.to_torch(ton)
    en = torch.equal(xn, rn)
    ok &= en
    print(
        "multi-shard same-spec nd [4,128,128]/[2,64,64] identity:",
        en,
        "max_diff",
        (xn.float() - rn.float()).abs().max().item(),
    )

    # ---- Multi-shard same-spec even: legacy HEIGHT [1,1,512,64]/[128,64] on 2 cores = 4 shards/2 cores ----
    grid2 = crs((1, 0))
    mc2 = height_mc(grid2, (128, 64))
    tih = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc2)
    toh = tilize(tih, memory_config=mc2)
    rh = ttnn.to_torch(toh)
    eh = torch.equal(x, rh)
    ok &= eh
    print(
        "multi-shard same-spec legacy HEIGHT 4shards/2cores identity:",
        eh,
        "max_diff",
        (x.float() - rh.float()).abs().max().item(),
    )
finally:
    ttnn.close_device(device)
print("ALL_OK", ok)
assert ok
