import torch, ttnn
from ttnn.operations.tilize.tilize_program_descriptor import shard_identity


def crs(*r):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*s), ttnn.CoreCoord(*e)) for (s, e) in r})


dev = ttnn.open_device(device_id=0)
try:
    for name, layout, shape, shard, grid in [
        ("H", ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (1, 1, 512, 64), (128, 64), (((0, 0), (3, 0)),)),
        ("W", ttnn.TensorMemoryLayout.WIDTH_SHARDED, (1, 1, 64, 512), (64, 128), (((0, 0), (3, 0)),)),
        ("B", ttnn.TensorMemoryLayout.BLOCK_SHARDED, (1, 1, 128, 128), (64, 64), (((0, 0), (1, 1)),)),
    ]:
        mem = ttnn.MemoryConfig(
            layout, ttnn.BufferType.L1, ttnn.ShardSpec(crs(*grid), shard, ttnn.ShardOrientation.ROW_MAJOR)
        )
        t = ttnn.from_torch(
            torch.randn(*shape).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=mem,
        )
        a = shard_identity(mem)
        b = shard_identity(t.memory_config())
        print(name, "equal:", a == b)
        if a != b:
            for x, y in zip(a, b):
                if x != y:
                    print("   DIFF:", x, "||", y)
finally:
    ttnn.close_device(dev)
