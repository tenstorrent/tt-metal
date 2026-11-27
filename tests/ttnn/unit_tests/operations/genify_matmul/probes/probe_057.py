import ttnn

dev = ttnn.open_device(device_id=0)
for noc in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1):
    opt = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(dev, noc)
    print(
        "OPT",
        noc,
        [(c.x, c.y) for c in opt],
        [(dev.worker_core_from_logical_core(c).x, dev.worker_core_from_logical_core(c).y) for c in opt],
    )
print("OPT grid", dev.compute_with_storage_grid_size())
ttnn.close_device(dev)
