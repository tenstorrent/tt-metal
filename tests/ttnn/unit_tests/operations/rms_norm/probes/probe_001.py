import ttnn

d = ttnn.open_device(device_id=0)
print("grid", d.compute_with_storage_grid_size())
print([a for a in dir(d) if "l1" in a.lower() or "size" in a.lower() or "alloc" in a.lower()])
print("unreserved", ttnn.get_max_worker_l1_unreserved_size())
try:
    print("l1_size_per_core", d.l1_size_per_core())
except Exception as e:
    print(e)
print("dram align", ttnn.get_dram_alignment(), "l1 align", ttnn.get_l1_alignment())
print(d.arch())
ttnn.close_device(d)
