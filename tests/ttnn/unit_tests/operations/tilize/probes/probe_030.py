import sys, torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)


def sh(layout, gx, gy, shard):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    return ttnn.MemoryConfig(layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR))


W, H = ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED
cases = [
    ((1, 1, 32, 2048), sh(W, 8, 1, (32, 256)), ttnn.DRAM_MEMORY_CONFIG),
    ((1, 1, 2048, 64), sh(H, 8, 8, (32, 64)), ttnn.DRAM_MEMORY_CONFIG),
    ((1, 1, 2048, 64), sh(H, 8, 8, (32, 64)), ttnn.L1_MEMORY_CONFIG),
    ((1, 1, 128, 64), sh(H, 4, 1, (32, 64)), ttnn.DRAM_MEMORY_CONFIG),
    ((1, 1, 128, 64), ttnn.DRAM_MEMORY_CONFIG, sh(H, 4, 1, (32, 64))),
]
for shape, mc, omc in cases:
    x = torch.randn(shape).to(torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    out = tilize(t, memory_config=omc)
    print("R", shape, torch.equal(ttnn.to_torch(out), x))
ttnn.close_device(device)
