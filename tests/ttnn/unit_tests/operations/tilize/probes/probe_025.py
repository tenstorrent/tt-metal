import os, sys, torch, ttnn
from ttnn.operations.tilize import tilize

mode = sys.argv[1]
_L1 = ttnn.BufferType.L1
grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
shard = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.BLOCK_SHARDED, _L1, ttnn.ShardSpec(grid, (256, 64), ttnn.ShardOrientation.ROW_MAJOR)
)
shape = (1, 1, 2048, 512)
device = ttnn.open_device(device_id=0)
try:
    t = torch.randn(shape).bfloat16()
    if mode == "alias_in":
        in_cfg, out_cfg = shard, ttnn.DRAM_MEMORY_CONFIG
    else:
        in_cfg, out_cfg = ttnn.DRAM_MEMORY_CONFIG, shard
    tt_in = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_cfg)
    out = tilize(tt_in, out_cfg)
    ttnn.synchronize_device(device)
    print("path mode:", mode, "ok")
finally:
    ttnn.close_device(device)
