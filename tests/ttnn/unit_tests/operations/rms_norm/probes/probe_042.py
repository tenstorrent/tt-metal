import ttnn

dev = ttnn.open_device(device_id=0)
try:
    arch = dev.arch()
    g = dev.compute_with_storage_grid_size()
    print("ARCH:", arch)
    print("compute_with_storage_grid_size:", g.x, "x", g.y)
    xs = set()
    ys = set()
    rowmap = {}
    for ly in range(g.y):
        row = []
        for lx in range(g.x):
            v = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx, ly))
            row.append((v.x, v.y))
            xs.add(v.x)
            ys.add(v.y)
        rowmap[ly] = row
    print("virtual x values (sorted):", sorted(xs))
    print("virtual y values (sorted):", sorted(ys))
    print("contiguous x?", sorted(xs) == list(range(min(xs), max(xs) + 1)))
    print("logical row y=0 -> virtual:", rowmap[0])
finally:
    ttnn.close_device(dev)
