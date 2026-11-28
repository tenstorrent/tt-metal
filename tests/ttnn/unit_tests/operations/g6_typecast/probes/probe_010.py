import ttnn

d = ttnn.open_device(device_id=0)
print("B grid", d.dram_grid_size())
for b in range(8):
    try:
        print("B", b, d.dram_core_from_logical_core(ttnn.CoreCoord(b, 0)))
    except Exception as e:
        print("E", e)
for noc in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1):
    try:
        print("O", noc, d.get_optimal_dram_bank_to_logical_worker_assignment(noc))
    except Exception as e:
        try:
            print("O", d.get_optimal_dram_bank_to_logical_worker_assignment())
        except Exception as e2:
            print("E", e2)
ttnn.close_device(d)
