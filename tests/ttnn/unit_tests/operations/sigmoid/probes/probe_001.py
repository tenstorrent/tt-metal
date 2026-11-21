import ttnn
d = ttnn.open_device(device_id=0)
print("L1_UNRESERVED", ttnn.get_max_worker_l1_unreserved_size())
print("GRID", d.compute_with_storage_grid_size())
print("L1_ALIGN", ttnn.get_l1_alignment(), "DRAM_ALIGN", ttnn.get_dram_alignment())
print("HAS_CCD", hasattr(ttnn, "ComputeConfigDescriptor"))
ttnn.close_device(d)
