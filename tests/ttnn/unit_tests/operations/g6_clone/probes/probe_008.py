import ttnn

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("grid", g.x, g.y, "dram_grid", device.dram_grid_size(), "arch", device.arch())
    for noc in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1):
        cores = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, noc)
        print("optimal", noc, [(c.x, c.y) for c in cores])
    print(
        "worker virtual row y=0:",
        [(x, device.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x) for x in range(g.x)],
    )
    print(
        "worker virtual col x=0:",
        [(y, device.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y) for y in range(g.y)],
    )
    d = device.dram_grid_size()
    for x in range(d.x):
        for y in range(d.y):
            v = device.dram_core_from_logical_core(ttnn.CoreCoord(x, y))
            print("dram logical", (x, y), "virtual", (v.x, v.y))
finally:
    ttnn.close_device(device)
