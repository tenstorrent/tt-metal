import torch, ttnn
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded_program_descriptor as d
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded as opmod_fn
import ttnn.operations.g6_interleaved_to_sharded.g6_interleaved_to_sharded as opmod

dev = ttnn.open_device(device_id=0)
try:
    x = ttnn.from_torch(
        torch.randn(1, 5120, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for (x0, y0, x1, y1), shard, orient, n in [
        ((0, 0, 7, 3), (32, 160), ttnn.ShardOrientation.ROW_MAJOR, 32),
        ((0, 0, 7, 3), (32, 160), ttnn.ShardOrientation.COL_MAJOR, 32),
        ((0, 0, 7, 7), (32, 160), ttnn.ShardOrientation.ROW_MAJOR, 32),
        ((0, 0, 6, 7), (32, 160), ttnn.ShardOrientation.ROW_MAJOR, 32),
        ((0, 0, 6, 7), (32, 160), ttnn.ShardOrientation.COL_MAJOR, 32),
    ]:
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shard, orient)
        )
        p = d.plan(x, mc)
        cr = d._core_ranges([b.core for b in p.blocks])
        print("RANGES", (x0, y0, x1, y1), orient, cr.ranges(), "num_cores", cr.num_cores())
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            (32, 160),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    opmod._DESCRIPTOR_CACHE.clear()
    y1 = opmod_fn(x, mc)
    desc = list(opmod._DESCRIPTOR_CACHE.values())[0]
    print("CRTA first", list(desc.kernels[0].common_runtime_args), x.buffer_address(), y1.buffer_address())
    x2 = ttnn.from_torch(
        torch.randn(1, 5120, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    y2 = opmod_fn(x2, mc)
    print("CRTA hit", list(desc.kernels[0].common_runtime_args), x2.buffer_address(), y2.buffer_address())
    print("HIT ok", torch.equal(ttnn.to_torch(y2).view(torch.int16), ttnn.to_torch(x2).view(torch.int16)))
finally:
    ttnn.close_device(dev)
