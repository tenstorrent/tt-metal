import ttnn

dev = ttnn.open_device(device_id=0)
g = dev.compute_with_storage_grid_size()
print("grid", g.x, g.y)
print("virt x", [dev.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x for x in range(g.x)])
print("virt y", [dev.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y for y in range(g.y)])
for e in ttnn._ttnn.cluster.get_dram_bank_table(0) if hasattr(ttnn._ttnn, "cluster") else []:
    print(e)
ttnn.close_device(dev)
