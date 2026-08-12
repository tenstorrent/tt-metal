import torch, ttnn
from ttnn.operations.tilize.tilize_program_descriptor import shard_identity, _nd_identity


def crs(*r):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*s), ttnn.CoreCoord(*e)) for (s, e) in r})


dev = ttnn.open_device(device_id=0)
try:
    mem = ttnn.MemoryConfig(
        ttnn.BufferType.L1,
        ttnn.NdShardSpec(ttnn.Shape((1, 1, 64, 64)), crs(((0, 0), (1, 1))), ttnn.ShardOrientation.ROW_MAJOR),
    )
    t = ttnn.from_torch(
        torch.randn(1, 1, 128, 128).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=mem,
    )
    print("constructed:", shard_identity(mem), _nd_identity(mem))
    print("live       :", shard_identity(t.memory_config()), _nd_identity(t.memory_config()))
    tt = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 128, 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, mem)
    print("live out   :", shard_identity(tt.memory_config()), _nd_identity(tt.memory_config()))
finally:
    ttnn.close_device(dev)
