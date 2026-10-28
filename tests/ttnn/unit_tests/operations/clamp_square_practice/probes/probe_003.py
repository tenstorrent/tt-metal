import ttnn

d = ttnn.open_device(device_id=0)
g = d.compute_with_storage_grid_size()
print("grid", g.x, g.y, "dram", d.dram_grid_size().x, d.dram_grid_size().y)
print("cols", [(x, d.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x) for x in range(g.x)])
print("rows", [(y, d.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y) for y in range(g.y)])
for b in range(d.dram_grid_size().x):
    c = d.dram_core_from_logical_core(ttnn.CoreCoord(b, 0))
    print("bank", b, c.x, c.y)
for n in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1):
    a = d.get_optimal_dram_bank_to_logical_worker_assignment(n)
    print(
        n,
        [(k, v.x, v.y) for k, v in sorted(a.items())]
        if isinstance(a, dict)
        else [(i, v.x, v.y) for i, v in enumerate(a)],
    )
print([m for m in dir(d) if "core" in m.lower() or "dram" in m.lower() or "noc" in m.lower()])
ttnn.close_device(d)
