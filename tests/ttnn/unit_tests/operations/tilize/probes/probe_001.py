import ttnn, torch
from eval.sharding import *  # noqa

grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
ss = ttnn.ShardSpec(grid, (128, 64), ttnn.ShardOrientation.ROW_MAJOR)
mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ss)
print(
    "legacy: layout",
    mc.memory_layout,
    "shard_spec",
    mc.shard_spec is not None,
    "nd",
    getattr(mc, "nd_shard_spec", None) is not None,
    "created_with_nd",
    getattr(mc, "created_with_nd_shard_spec", "n/a"),
)
nd = ttnn.NdShardSpec(ttnn.Shape([1, 1, 64, 64]), grid, ttnn.ShardOrientation.ROW_MAJOR)
mc2 = ttnn.MemoryConfig(ttnn.BufferType.L1, nd)
print(
    "nd: layout",
    mc2.memory_layout,
    "shard_spec",
    mc2.shard_spec is not None,
    "nd",
    mc2.nd_shard_spec is not None,
    "created_with_nd",
    getattr(mc2, "created_with_nd_shard_spec", "n/a"),
)
print([a for a in dir(mc) if not a.startswith("_")])
