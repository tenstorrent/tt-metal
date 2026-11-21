import ttnn, torch
dev = ttnn.open_device(device_id=0)
print("l1", ttnn.get_l1_alignment(), "dram", ttnn.get_dram_alignment(), "unres", ttnn.get_max_worker_l1_unreserved_size())
print("grid", dev.compute_with_storage_grid_size())
try:
    print("l1 banks", ttnn.device.GetMemoryView(dev, ttnn.BufferType.L1).num_banks)
except Exception as e: print("GMV err", e)
t = ttnn.from_torch(torch.randn(1,4,16,20, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
print("pagesz", t.buffer_page_size(), [a for a in dir(t) if 'page' in a or 'aligned' in a])
print(t.memory_config(), t.memory_config().buffer_type)
E = ttnn.experimental
c = E.ComputeHardwareConfig(fpu_math_fidelity=ttnn.MathFidelity.LoFi, enable_32_bit_dest=False)
print("cfg ok", c)
print([a for a in dir(E) if 'config' in a.lower()])
ttnn.close_device(dev)
