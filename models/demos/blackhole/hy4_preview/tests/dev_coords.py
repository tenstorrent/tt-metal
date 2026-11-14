import ttnn

for i in range(32):
    d = ttnn.open_device(device_id=i)
    try:
        xs = [d.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x for x in range(8)]
        ys = [d.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y for y in range(10)]
        g = d.compute_with_storage_grid_size()
        print("COORD", i, "grid", g.x, g.y, "physx", xs, "physy", ys, flush=True)
    finally:
        ttnn.close_device(d)
