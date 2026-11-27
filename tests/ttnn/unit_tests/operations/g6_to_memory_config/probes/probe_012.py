# Verifier probe: SUPPORTED claims DRAM height/block-sharded targets (out_buffer=dram x out_scheme in
# {height_sharded, block_sharded}); no production case exercises them. Check exactness on production shapes.
import torch
import ttnn
from ttnn.operations.g6_to_memory_config import g6_to_memory_config, plan_of

device = ttnn.open_device(device_id=0)
try:
    dram = device.dram_grid_size()
    dram_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, 0))])
    nb = dram.x

    def dram_sharded(scheme, shard):
        return ttnn.MemoryConfig(
            scheme, ttnn.BufferType.DRAM, ttnn.ShardSpec(dram_grid, shard, ttnn.ShardOrientation.ROW_MAJOR)
        )

    l1_width = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))]),
            (32, 1664),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    cases = [
        (
            "1536x5120 DRAM il -> DRAM height",
            [1536, 5120],
            ttnn.DRAM_MEMORY_CONFIG,
            dram_sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (1536 // nb, 5120)),
        ),
        (
            "1536x5120 DRAM il -> DRAM block 1x8",
            [1536, 5120],
            ttnn.DRAM_MEMORY_CONFIG,
            dram_sharded(ttnn.TensorMemoryLayout.BLOCK_SHARDED, (1536, 5120 // nb)),
        ),
        (
            "1x1x16x12928 L1 width -> DRAM height",
            [1, 1, 16, 12928],
            l1_width,
            dram_sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (32, 12928)),
        ),
        (
            "16x128x1536 L1 il -> DRAM height",
            [16, 128, 1536],
            ttnn.L1_MEMORY_CONFIG,
            dram_sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (2048 // nb, 1536)),
        ),
    ]
    for name, shape, in_cfg, out_cfg in cases:
        try:
            x = ttnn.from_torch(
                torch.randn(shape).bfloat16(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=in_cfg,
            )
            regime = plan_of(x, out_cfg)["regime"]
            y = g6_to_memory_config(x, out_cfg)
            ok = torch.equal(ttnn.to_torch(y), ttnn.to_torch(x)) and y.memory_config() == out_cfg
            print("RES", name, regime, "exact+placement" if ok else "MISMATCH")
        except Exception as e:
            print("RES", name, "EXC", repr(e)[:200])
finally:
    ttnn.close_device(device)
