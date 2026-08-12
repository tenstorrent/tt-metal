import torch, ttnn
from ttnn.operations.tilize.tilize_program_descriptor import shard_view, shard_identity, _nd_identity

grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
dev = ttnn.open_device(device_id=0)
try:
    for shape, sh in ([4, 128, 128], [2, 64, 64]), ([3, 160, 160], [2, 64, 64]), ([23, 96, 160], [4, 64, 96]):
        mem = ttnn.MemoryConfig(
            buffer_type=ttnn.BufferType.L1,
            nd_shard_spec=ttnn.NdShardSpec(shard_shape=sh, grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR),
        )
        rm = ttnn.from_torch(
            torch.rand(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=mem,
        )
        tl = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, mem)
        print(shape, sh)
        print(
            "   rm  view",
            shard_view(rm.memory_config()),
            "nd",
            _nd_identity(rm.memory_config()),
            "bytes/bank",
            rm.buffer_page_size(),
            rm.buffer_num_pages(),
        )
        print(
            "   til view",
            shard_view(tl.memory_config()),
            "nd",
            _nd_identity(tl.memory_config()),
            "pages",
            tl.buffer_num_pages(),
        )
        print("   same:", shard_identity(rm.memory_config()) == shard_identity(tl.memory_config()))
finally:
    ttnn.close_device(dev)
