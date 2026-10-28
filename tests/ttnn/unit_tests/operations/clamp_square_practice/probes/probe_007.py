import ttnn

d = ttnn.open_device(device_id=0)
g = d.compute_with_storage_grid_size()
print("COORD grid", g.x, g.y)
print(
    "COORD row0",
    [
        (
            d.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x,
            d.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).y,
        )
        for x in range(g.x)
    ],
)
print("COORD col0", [d.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y for y in range(g.y)])
ttnn.close_device(d)
