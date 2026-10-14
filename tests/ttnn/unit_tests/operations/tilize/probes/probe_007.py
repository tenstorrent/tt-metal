import torch, ttnn

device = ttnn.open_device(device_id=0)


def crs(*r):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*s), ttnn.CoreCoord(*e)) for s, e in r})


L1 = ttnn.BufferType.L1
cfgs = {
    "height": (
        [1, 1, 512, 64],
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            L1,
            ttnn.ShardSpec(crs(((0, 0), (3, 0))), (128, 64), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    ),
    "width": (
        [1, 1, 64, 512],
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            L1,
            ttnn.ShardSpec(crs(((0, 0), (3, 0))), (64, 128), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    ),
    "blockcol": (
        [1, 1, 128, 128],
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            L1,
            ttnn.ShardSpec(crs(((0, 0), (1, 1))), (64, 64), ttnn.ShardOrientation.COL_MAJOR),
        ),
    ),
    "nd": (
        [1, 1, 128, 64],
        ttnn.MemoryConfig(
            L1, ttnn.NdShardSpec(ttnn.Shape([1, 1, 64, 64]), crs(((0, 0), (1, 0))), ttnn.ShardOrientation.ROW_MAJOR)
        ),
    ),
    "dram": (
        [1, 1, 128, 64],
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(crs(((0, 0), (1, 0))), (64, 64), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    ),
}
for k, (shape, mc) in cfgs.items():
    t = ttnn.from_torch(
        torch.randn(shape).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mc,
    )
    m = t.memory_config()
    print(k, m.memory_layout, m.shard_spec, m.nd_shard_spec, "page", t.buffer_page_size(), t.buffer_aligned_page_size())
    print("  ct args", list(ttnn.TensorAccessorArgs(t).get_compile_time_args()))
    print("  dir", [a for a in dir(m.nd_shard_spec) if not a.startswith("_")])
    tt = ttnn.from_torch(
        torch.randn(shape).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )
    print("  tile page", tt.buffer_page_size(), tt.memory_config().nd_shard_spec)
ttnn.close_device(device)
