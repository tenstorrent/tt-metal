import ttnn

dev = ttnn.open_device(device_id=0)
print("OUT dram grid", dev.dram_grid_size(), "grid", dev.compute_with_storage_grid_size())
for noc in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1):
    opt = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(dev, noc)
    print("OUT", noc, [(c.x, c.y) for c in opt])
v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
print("OUT dram banks", v.num_banks)
for x in range(11):
    print(
        "OUT logical",
        (x, 0),
        "->",
        dev.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)),
        (x, 9),
        dev.worker_core_from_logical_core(ttnn.CoreCoord(x, 9)),
    )
ttnn.close_device(dev)
