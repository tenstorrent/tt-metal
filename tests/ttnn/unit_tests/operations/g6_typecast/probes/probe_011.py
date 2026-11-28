import ttnn, torch

dev = ttnn.open_device(device_id=0)
g = dev.compute_with_storage_grid_size()
print("P grid", g.x, g.y, "dram_grid", dev.dram_grid_size())
for noc in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1):
    a = dev.get_optimal_dram_bank_to_logical_worker_assignment(noc)
    print("P opt", noc, [(c.x, c.y) for c in a])
for b in range(8):
    try:
        print("P dram", b, dev.dram_core_from_logical_core(ttnn.CoreCoord(b, 0)))
    except Exception as e:
        print("P dram err", b, e)
print("P row0 phys", [str(dev.worker_core_from_logical_core(ttnn.CoreCoord(x, 0))) for x in range(g.x)])
print("P col0 phys", [str(dev.worker_core_from_logical_core(ttnn.CoreCoord(0, y))) for y in range(g.y)])
print("P l1 banks", dev.num_banks(ttnn.BufferType.L1), "dram banks", dev.num_banks(ttnn.BufferType.DRAM))
mv = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
print("P l1 view", [m for m in dir(mv) if not m.startswith("_")])
print("P", [m for m in dir(ttnn) if "bank" in m.lower()])
ttnn.close_device(dev)
