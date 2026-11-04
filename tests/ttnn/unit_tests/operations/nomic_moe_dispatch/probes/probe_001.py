import ttnn

device = ttnn.open_device(device_id=0)
print("grid", device.compute_with_storage_grid_size())
R, w = 8416, 12
grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9))])
nd = ttnn.NdShardSpec(
    ttnn.Shape([32, 32 * w]), grid, ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D
)
mc = ttnn.MemoryConfig(ttnn.BufferType.L1, nd)
print(mc)
spec = (
    ttnn.TensorSpec(
        ttnn.Shape([1, 1, R, 768]), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1, memory_config=mc
    )
    if False
    else None
)
import inspect

print(ttnn.TensorSpec.__init__.__doc__)
t = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, R, 768]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
print("out mc", t.memory_config(), t.memory_config() == mc, t.is_sharded())
cb = ttnn.cb_descriptor_from_sharded_tensor(2, t, core_ranges=grid)
print("cb total", cb.total_size, cb.format_descriptors[0].page_size)
print("l1 banks", device.num_banks(ttnn.BufferType.L1) if hasattr(device, "num_banks") else "?")
ttnn.close_device(device)
