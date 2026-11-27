import ttnn

d = ttnn.open_device(device_id=0)
print("isMesh", isinstance(d, ttnn.MeshDevice), ttnn.Device is ttnn.MeshDevice)
print("grid", d.compute_with_storage_grid_size(), "dram", d.dram_grid_size(), "arch", d.arch(), repr(d.arch()))
print("shape", d.shape, "num_devices", d.get_num_devices() if hasattr(d, "get_num_devices") else None)
print("mesh id", d.id() if hasattr(d, "id") else None)
print("has optimal", hasattr(d, "get_optimal_dram_bank_to_logical_worker_assignment"))
for dt in (ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b):
    u = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 4, 256, 128]), dt, ttnn.TILE_LAYOUT, d, ttnn.DRAM_MEMORY_CONFIG)
    print(dt, u.buffer_num_pages(), u.buffer_page_size(), u.buffer_aligned_page_size(), hex(u.buffer_address()))
    ttnn.deallocate(u)
print("dram align", ttnn.get_dram_alignment(), "l1 align", ttnn.get_l1_alignment())
print([a for a in dir(ttnn.Arch)])
mc = ttnn.DRAM_MEMORY_CONFIG
print(mc.memory_layout, mc.buffer_type, mc.is_sharded() if hasattr(mc, "is_sharded") else None)
print([a for a in dir(ttnn.KernelDescriptor) if not a.startswith("_")])
print([a for a in dir(ttnn.ProgramDescriptor) if not a.startswith("_")])
ttnn.close_device(d)
