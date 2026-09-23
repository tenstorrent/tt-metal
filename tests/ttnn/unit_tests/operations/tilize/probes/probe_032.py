import ttnn

d = ttnn.open_device(device_id=0)
g = d.compute_with_storage_grid_size()
print("grid", g)
for y in range(g.y):
    print(y, [(c.x, c.y) for c in [d.worker_core_from_logical_core(ttnn.CoreCoord(x, y)) for x in range(g.x)]])
try:
    print("dram grid", d.dram_grid_size())
    for c in range(d.dram_grid_size().x):
        print("dram", c, d.dram_core_from_logical_core(ttnn.CoreCoord(c, 0)))
except Exception as e:
    print("err", e)
ttnn.close_device(d)
