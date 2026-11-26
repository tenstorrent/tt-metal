import ttnn

d = ttnn.open_device(device_id=0)
g = d.compute_with_storage_grid_size()
print("grid", g.x, g.y)
print("workers x:", [(d.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x) for x in range(g.x)])
print("workers y:", [(d.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y) for y in range(g.y)])
for b in range(8):
    try:
        c = d.dram_core_from_logical_core(ttnn.CoreCoord(b, 0))
        print("dram", b, c.x, c.y)
    except Exception as e:
        print("dram", b, "err", str(e)[:80])
try:
    a0 = d.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    a1 = d.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_1)
    print("opt0", [(k, v.x, v.y) for k, v in sorted(a0.items())] if isinstance(a0, dict) else [(v.x, v.y) for v in a0])
    print("opt1", [(k, v.x, v.y) for k, v in sorted(a1.items())] if isinstance(a1, dict) else [(v.x, v.y) for v in a1])
except Exception as e:
    print("opt err", e)
try:
    h = ttnn._ttnn.experimental.get_worker_noc_hop_distance if hasattr(ttnn._ttnn, "experimental") else None
    print("hop fn", h)
except Exception as e:
    print("hop err", e)
ttnn.close_device(d)
