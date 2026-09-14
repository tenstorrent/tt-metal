import ttnn

dev = ttnn.open_device(device_id=0)
g = dev.compute_with_storage_grid_size()
print("GRID", g.x, g.y)
print("L1", ttnn.get_max_worker_l1_unreserved_size())
for x in range(g.x):
    c = dev.worker_core_from_logical_core(ttnn.CoreCoord(x, 0))
    print("virt", x, "->", c.x, c.y)
ttnn.close_device(dev)
