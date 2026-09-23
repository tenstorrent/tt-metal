import sys, torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)


def hs(w):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (32, w), ttnn.ShardOrientation.ROW_MAJOR),
    )


for shape, mc, omc in [((1, 1, 2048, 512), hs(512), ttnn.DRAM_MEMORY_CONFIG)]:
    x = torch.randn(shape).to(torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    out = tilize(t, memory_config=omc)
ttnn.close_device(device)
