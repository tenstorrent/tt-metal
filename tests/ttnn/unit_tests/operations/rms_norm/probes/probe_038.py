import ttnn

device = ttnn.open_device(device_id=0)
try:
    arch = device.arch()
    g = device.compute_with_storage_grid_size()
    print("ARCH", arch, "COMPUTE_GRID", g.x, g.y)

    def v(x, y):
        c = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        return (int(c.x), int(c.y))

    # Dump virtual x for logical row y=0, all logical x
    row0 = [v(x, 0) for x in range(g.x)]
    print("logical row y=0 -> virtual:", row0)
    print("virtual X values across logical x=0..%d:" % (g.x - 1), [a for a, b in row0])

    # Dump virtual y for logical column x=0, all logical y
    col0 = [v(0, y) for y in range(g.y)]
    print("virtual Y values across logical y=0..%d:" % (g.y - 1), [b for a, b in col0])

    # Rectangle check for the geometries in probe_037
    print("=== rectangle (area==nmembers) check ===")
    for gx, gy, label in [(8, 1, "8x1"), (9, 1, "9x1"), (8, 4, "8x4"), (7, 4, "7x4"), (8, 8, "8x8")]:
        members = [v(x, y) for y in range(gy) for x in range(gx)]
        xs = [a for a, b in members]
        ys = [b for a, b in members]
        area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
        print(
            f"{label}: n={len(members)} vx[{min(xs)},{max(xs)}] vy[{min(ys)},{max(ys)}] area={area} contiguous_rect={area==len(members)}"
        )
finally:
    ttnn.close_device(device)
