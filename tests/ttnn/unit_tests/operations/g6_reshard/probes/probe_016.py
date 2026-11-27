import ttnn

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("GRID", g.x, g.y)
    print("ARCH", device.arch())
    print("API", [a for a in dir(device) if "core" in a.lower() or "noc" in a.lower()])
    for y in range(g.y):
        print(
            "ROW",
            y,
            [
                tuple((lambda v: (v.x, v.y))(device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))))
                for x in range(g.x)
            ],
        )
finally:
    ttnn.close_device(device)
