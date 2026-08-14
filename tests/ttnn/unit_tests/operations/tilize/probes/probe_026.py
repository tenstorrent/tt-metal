import torch, ttnn


def crs(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


def alloc_spec(shape, dtype, mc, tile, device):
    if not mc.is_sharded():
        spec = ttnn.TensorSpec(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, buffer_type=mc.buffer_type, tile=tile)
    elif mc.shard_spec is not None:
        spec = ttnn.TensorSpec(
            ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, mc.memory_layout, mc.shard_spec, mc.buffer_type, tile
        )
    else:
        spec = ttnn.TensorSpec(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, mc.nd_shard_spec, mc.buffer_type, tile)
    return ttnn.allocate_tensor_on_device(spec, device)


device = ttnn.open_device(device_id=0)
try:
    shape = [1, 1, 128, 256]
    for name, mc in [
        ("dram", ttnn.DRAM_MEMORY_CONFIG),
        ("l1", ttnn.L1_MEMORY_CONFIG),
        (
            "legacy_h",
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs(0, 0, 3, 0), (32, 256), ttnn.ShardOrientation.ROW_MAJOR),
            ),
        ),
        (
            "nd",
            ttnn.MemoryConfig(
                ttnn.BufferType.L1,
                ttnn.NdShardSpec(ttnn.Shape([1, 1, 32, 256]), crs(0, 0, 3, 0), ttnn.ShardOrientation.ROW_MAJOR),
            ),
        ),
    ]:
        a = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
        b = alloc_spec(shape, ttnn.bfloat16, mc, ttnn.Tile([32, 32]), device)
        c = alloc_spec(shape, ttnn.bfloat16, mc, ttnn.Tile([16, 32]), device)
        print(
            name,
            "same spec:",
            a.spec == b.spec,
            "| pagesz",
            a.buffer_page_size(),
            b.buffer_page_size(),
            c.buffer_page_size(),
            "| tile16",
            c.tile.tile_shape,
            c.padded_shape,
        )
finally:
    ttnn.close_device(device)
