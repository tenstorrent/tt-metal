import torch, ttnn

device = ttnn.open_device(device_id=0)


def crs(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


cases = {
    "nd_2deq": (
        ttnn.MemoryConfig(
            ttnn.BufferType.L1,
            ttnn.NdShardSpec(ttnn.Shape([1, 1, 64, 64]), crs(0, 0, 1, 0), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        [1, 1, 128, 64],
    ),
    "nd_3d": (
        ttnn.MemoryConfig(
            ttnn.BufferType.L1,
            ttnn.NdShardSpec(ttnn.Shape([2, 64, 64]), crs(0, 0, 1, 1), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        [4, 128, 128],
    ),
    "nd_w": (
        ttnn.MemoryConfig(
            ttnn.BufferType.L1,
            ttnn.NdShardSpec(ttnn.Shape([1, 64, 128]), crs(0, 0, 1, 1), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        [4, 128, 128],
    ),
    "legacy_w": (
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs(0, 0, 3, 0), (64, 128), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        [1, 1, 64, 512],
    ),
    "legacy_b_col": (
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs(0, 0, 1, 1), (64, 64), ttnn.ShardOrientation.COL_MAJOR),
        ),
        [1, 1, 128, 128],
    ),
}
for name, (mc, shape) in cases.items():
    t = ttnn.from_torch(
        torch.randn(shape).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mc,
    )
    m = t.memory_config()
    print(
        name,
        "layout",
        m.memory_layout,
        "shard_spec",
        m.shard_spec,
        "nd",
        m.nd_shard_spec,
        "page",
        t.buffer_page_size(),
        t.buffer_aligned_page_size(),
        "json_nd",
        '"created_with_nd_shard_spec":true' in m.to_json().replace(" ", ""),
    )
    print("  attrs", [a for a in dir(t) if "buf" in a or "shard" in a])
tt = ttnn.from_torch(
    torch.randn([1, 1, 128, 128]).bfloat16(),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=cases["legacy_b_col"][0],
)
print("tile page", tt.buffer_page_size())
ttnn.close_device(device)
