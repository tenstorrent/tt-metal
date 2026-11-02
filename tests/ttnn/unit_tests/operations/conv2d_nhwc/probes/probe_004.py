import torch, ttnn

device = ttnn.open_device(device_id=0)
print(device.compute_with_storage_grid_size())
# C_out=320 over 8 slices of 64 -> 512 wide total; shards 5..7 empty
for ml, grid, shard, shape in [
    (
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        [256, 64],
        [1, 32, 32, 320],
    ),
    (
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
        [64, 32],
        [1, 7, 7, 4],
    ),
]:
    spec = ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ml, ttnn.BufferType.L1, spec)
    t = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, device, mc)
    print("alloc ok", t.memory_config(), t.buffer_address())
    x = torch.randn(shape)
    t2 = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    y = ttnn.to_torch(t2)
    print("roundtrip", torch.equal(x, y))
print(
    device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0)),
    device.worker_core_from_logical_core(ttnn.CoreCoord(3, 2)),
)
ttnn.close_device(device)
