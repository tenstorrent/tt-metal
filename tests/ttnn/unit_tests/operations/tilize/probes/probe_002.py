import torch, ttnn

dev = ttnn.open_device(device_id=0)


def show(name, mc):
    print(f"--- {name}")
    print("  memory_layout:", mc.memory_layout, "buffer_type:", mc.buffer_type, "is_sharded:", mc.is_sharded())
    print("  shard_spec:", mc.shard_spec)
    print("  nd_shard_spec:", getattr(mc, "nd_shard_spec", "N/A"))
    print("  attrs:", [a for a in dir(mc) if not a.startswith("_")])


show("dram", ttnn.DRAM_MEMORY_CONFIG)
show("l1", ttnn.L1_MEMORY_CONFIG)
grid4 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
hs = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(grid4, (128, 64), ttnn.ShardOrientation.ROW_MAJOR),
)
show("hs", hs)
grid22 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
nd = ttnn.MemoryConfig(
    ttnn.BufferType.L1, ttnn.NdShardSpec(ttnn.Shape((1, 1, 64, 64)), grid22, ttnn.ShardOrientation.ROW_MAJOR)
)
show("nd", nd)
print(
    "nd shard_shape:",
    nd.nd_shard_spec.shard_shape,
    "grid:",
    nd.nd_shard_spec.grid,
    "num_cores",
    nd.nd_shard_spec.grid.num_cores(),
)
print(
    "hs grid num_cores:",
    hs.shard_spec.grid.num_cores(),
    "shape",
    hs.shard_spec.shape,
    "orient",
    hs.shard_spec.orientation,
)

# ND RM tensor page semantics
t = ttnn.from_torch(
    torch.arange(128 * 128).reshape(1, 1, 128, 128).to(torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=dev,
    memory_config=nd,
)
print(
    "nd RM page_size", t.buffer_page_size(), "aligned", t.buffer_aligned_page_size(), "num_pages", t.buffer_num_pages()
)
ot = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 128, 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, nd)
print("nd TILE page_size", ot.buffer_page_size(), "num_pages", ot.buffer_num_pages())
print("out mc:", ot.memory_config())

# block sharded col major
bs = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(grid22, (64, 64), ttnn.ShardOrientation.COL_MAJOR),
)
tb = ttnn.from_torch(
    torch.randn(1, 1, 128, 128).bfloat16(),
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=dev,
    memory_config=bs,
)
print("bs RM page_size", tb.buffer_page_size(), "num_pages", tb.buffer_num_pages())

# CBFormatDescriptor for int types
for dt in [ttnn.uint32, ttnn.uint16, ttnn.int32, ttnn.bfloat8_b]:
    try:
        f = ttnn.CBFormatDescriptor(buffer_index=0, data_format=dt, page_size=ttnn.tile_size(dt))
        print("CBFormat ok", dt, f.data_format_as_uint8)
    except Exception as e:
        print("CBFormat FAIL", dt, e)

# tile_size vs shard alignment check
print("aligned_size_per_bank probes:")
print("  hs rm tensor:", t.buffer_page_size() * t.buffer_num_pages())
ttnn.close_device(dev)
