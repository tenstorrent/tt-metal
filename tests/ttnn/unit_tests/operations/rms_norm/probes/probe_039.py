import ttnn

device = ttnn.open_device(device_id=0)
try:
    print("ARCH", device.arch())
    g = device.compute_with_storage_grid_size()
    print("COMPUTE_GRID", g.x, g.y)
    # DRAM grid
    try:
        dg = device.dram_grid_size()
        print("DRAM_GRID", dg.x, dg.y)
    except Exception as e:
        print("dram_grid_size err", e)

    # Where do DRAM cores land in virtual space?
    print("=== DRAM cores: logical -> virtual ===")
    dg = device.dram_grid_size()
    for y in range(dg.y):
        row = []
        for x in range(dg.x):
            try:
                c = device.virtual_core_from_logical_core(ttnn.CoreCoord(x, y), ttnn.CoreType.DRAM)
                row.append((int(c.x), int(c.y)))
            except Exception as e:
                row.append(("ERR", str(e)[:20]))
        print(f"dram logical y={y}: {row}")

    # Worker virtual coords again for reference
    def wv(x, y):
        c = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        return (int(c.x), int(c.y))

    print("worker virtual X (logical x=0..%d):" % (g.x - 1), [wv(x, 0)[0] for x in range(g.x)])
finally:
    ttnn.close_device(device)
