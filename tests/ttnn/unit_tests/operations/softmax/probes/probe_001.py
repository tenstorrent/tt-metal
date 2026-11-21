import ttnn
d = ttnn.open_device(device_id=0)
print("UNRESERVED", ttnn.get_max_worker_l1_unreserved_size())
print("GRID", d.compute_with_storage_grid_size())
print("L1ALIGN", ttnn.get_l1_alignment())
ttnn.close_device(d)
