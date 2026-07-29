import torch, ttnn

for n in [
    "tile_size",
    "grid_to_cores",
    "num_cores_to_corerangeset",
    "find_max_divisor",
    "div_up",
    "round_up",
    "UnpackToDestMode",
    "ComputeConfigDescriptor",
    "cb_descriptor_from_sharded_tensor",
    "get_l1_alignment",
    "get_dram_alignment",
    "CBDescriptor",
    "CBFormatDescriptor",
    "RuntimeArgs",
    "TensorAccessorArgs",
]:
    print(f"ttnn.{n}:", hasattr(ttnn, n))

dev = ttnn.open_device(device_id=0)
print("grid:", dev.compute_with_storage_grid_size())
print("l1_size_per_core:", hasattr(dev, "l1_size_per_core"))
print("dev attrs:", [a for a in dir(dev) if "l1" in a.lower() or "grid" in a.lower()])

cfg = ttnn.ComputeConfigDescriptor()
print("cfg unpack_to_dest_mode default:", cfg.unpack_to_dest_mode)
print("cfg attrs:", [a for a in dir(cfg) if not a.startswith("_")])
print("UnpackToDestMode members:", [a for a in dir(ttnn.UnpackToDestMode) if not a.startswith("_")])

# tile sizes
for dt in [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.uint32, ttnn.uint16, ttnn.int32]:
    print("tile_size", dt, ttnn.tile_size(dt))

# RM interleaved page semantics
t = ttnn.from_torch(
    torch.randn(1, 1, 64, 128).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
)
print(
    "RM interleaved page_size",
    t.buffer_page_size(),
    "aligned",
    t.buffer_aligned_page_size(),
    "num_pages",
    t.buffer_num_pages(),
)

# RM height-sharded
mc = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        (128, 64),
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)
ts = ttnn.from_torch(
    torch.randn(1, 1, 512, 64).bfloat16(),
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=dev,
    memory_config=mc,
)
print(
    "RM hs page_size",
    ts.buffer_page_size(),
    "aligned",
    ts.buffer_aligned_page_size(),
    "num_pages",
    ts.buffer_num_pages(),
)
print("shard_spec", ts.memory_config().shard_spec)

# RM width-sharded
mcw = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        (64, 128),
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)
tw = ttnn.from_torch(
    torch.randn(1, 1, 64, 512).bfloat16(),
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=dev,
    memory_config=mcw,
)
print(
    "RM ws page_size",
    tw.buffer_page_size(),
    "aligned",
    tw.buffer_aligned_page_size(),
    "num_pages",
    tw.buffer_num_pages(),
)

# TILE sharded out (allocate)
mct = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        (128, 64),
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)
ot = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 512, 64]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, mct)
print("TILE hs page_size", ot.buffer_page_size(), "num_pages", ot.buffer_num_pages())

ttnn.close_device(dev)
