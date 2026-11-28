import torch, ttnn, math
import sys, ttnn.operations.g6_sharded_to_interleaved

m = sys.modules["ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved"]
from ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved_program_descriptor import Knobs

m.SUPPORTED.update(
    {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "in_scheme": ["width_sharded", "height_sharded"],
        "buffer": ["dram", "l1"],
    }
)
dev = ttnn.open_device(device_id=0)
cases = [
    (
        (1, 5120),
        ttnn.bfloat16,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        (32, 160),
        [((0, 0), (7, 3))],
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
    (
        (1, 5120),
        ttnn.bfloat16,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        (32, 96),
        [((0, 0), (8, 5))],
        ttnn.ShardOrientation.COL_MAJOR,
    ),
    (
        (1, 128, 256, 32),
        ttnn.bfloat8_b,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        (320, 32),
        [((0, 0), (10, 8)), ((0, 9), (3, 9))],
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
    (
        (1, 128, 256, 32),
        ttnn.bfloat8_b,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        (352, 32),
        [((0, 0), (8, 9)), ((9, 0), (9, 3))],
        ttnn.ShardOrientation.COL_MAJOR,
    ),
]
torch.manual_seed(0)
for shape, dt, lay, shard, grid, orient in cases:
    x = torch.randn(shape).to(torch.bfloat16) * 4
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b)) for a, b in grid])
    mc = ttnn.MemoryConfig(lay, ttnn.BufferType.L1, ttnn.ShardSpec(crs, shard, orient))
    t = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    ref = ttnn.to_torch(t)
    for omc in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
        for kn in (
            Knobs(),
            Knobs(writer_lanes=2),
            Knobs(blocks_per_shard=3),
            Knobs(blocks_per_shard=2, writer_lanes=2),
        ):
            o = m.g6_sharded_to_interleaved(t, omc, _knobs=kn)
            ok = torch.equal(ttnn.to_torch(o).view(torch.int16), ref.view(torch.int16))
            print("OK" if ok else "BAD", shape, dt, shard, orient, omc.buffer_type, kn)
            ttnn.deallocate(o)
    print("plan", m.last_plan())
ttnn.close_device(dev)
