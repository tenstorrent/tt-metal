import ttnn

d = ttnn.open_device(device_id=0)
print("dram grid", d.dram_grid_size())
for c in range(d.dram_grid_size().x):
    print(c, d.dram_core_from_logical_core(ttnn.CoreCoord(c, 0)))
g = d.compute_with_storage_grid_size()
print("grid", g)
print([(x, d.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x) for x in range(g.x)])
print([(y, d.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y) for y in range(g.y)])
ttnn.close_device(d)
