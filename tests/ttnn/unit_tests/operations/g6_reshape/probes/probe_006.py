import torch, ttnn

d = ttnn.open_device(device_id=0)
g = d.compute_with_storage_grid_size()
print("arch", d.arch(), "grid", g.x, g.y, "dram_align", ttnn.get_dram_alignment(), "l1_align", ttnn.get_l1_alignment())
print("generic_op", hasattr(ttnn, "generic_op"), "ProgramDescriptor", hasattr(ttnn, "ProgramDescriptor"))
ttnn.close_device(d)
