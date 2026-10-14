import ttnn

d = ttnn.open_device(device_id=0)
g = d.compute_with_storage_grid_size()
print("MAP x", [(lambda v: (v.x))(d.worker_core_from_logical_core(ttnn.CoreCoord(x, 0))) for x in range(g.x)])
print("MAP y", [(lambda v: (v.y))(d.worker_core_from_logical_core(ttnn.CoreCoord(0, y))) for y in range(g.y)])
ttnn.close_device(d)
