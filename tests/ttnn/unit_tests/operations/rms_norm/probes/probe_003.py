import torch, ttnn, sys
sys.path.insert(0, ".")
from eval.sharding import auto_shard_config, shard_config
device = ttnn.open_device(device_id=0)
print("L1 base", ttnn._ttnn.device.GetAllocatorBaseAddress(device, ttnn.BufferType.L1), "dram align", ttnn._ttnn.device.get_dram_alignment(), "l1 unreserved", ttnn._ttnn.device.get_max_worker_l1_unreserved_size())
g = device.compute_with_storage_grid_size(); print("grid", g.x, g.y)
for shape, ml, layout in [((1,1,256,512), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TILE_LAYOUT),
                          ((1,1,32,4096), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.ROW_MAJOR_LAYOUT),
                          ((1,1,17,50), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.ROW_MAJOR_LAYOUT),
                          ((1,1,2048,256), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.ROW_MAJOR_LAYOUT),
                          ((1,1,256,512), ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TILE_LAYOUT),
                          ((1,1,416,1184), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TILE_LAYOUT)]:
    mc = auto_shard_config(list(shape), ml, layout=layout, dtype=ttnn.bfloat16, device=device)
    t = ttnn.from_torch(torch.randn(shape).bfloat16(), dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mc)
    ss = t.memory_config().shard_spec
    print(shape, ml, layout, "shard", ss.shape, ss.orientation, ss.grid, "addr", t.buffer_address(), "page", t.buffer_page_size(), "padded", t.padded_shape)
    print("  cores", [ (c.x,c.y) for c in ttnn.corerange_to_cores(ss.grid, row_wise=True)][:12])
    y = ttnn.allocate_tensor_on_device(t.shape, t.dtype, t.layout, device, t.memory_config())
    print("  out addr", y.buffer_address())
ttnn.close_device(device)
