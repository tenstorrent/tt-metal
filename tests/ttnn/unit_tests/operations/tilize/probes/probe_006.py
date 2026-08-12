import torch, ttnn
from ttnn.operations.tilize.tilize_program_descriptor import shard_identity


def crs(*r):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*s), ttnn.CoreCoord(*e)) for (s, e) in r})


dev = ttnn.open_device(device_id=0)
try:
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs(((0, 0), (3, 0))), (128, 64), ttnn.ShardOrientation.ROW_MAJOR),
    )
    t = ttnn.from_torch(
        torch.randn(1, 1, 512, 64).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=mem,
    )
    print("A:", shard_identity(mem))
    print("B:", shard_identity(t.memory_config()))
    for mc, tag in ((mem, "constructed"), (t.memory_config(), "live")):
        try:
            print(tag, "nd:", mc.nd_shard_spec)
        except Exception as e:
            print(tag, "nd raised", type(e).__name__, e)
finally:
    ttnn.close_device(dev)
