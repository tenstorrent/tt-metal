import ttnn

d = ttnn.open_device(device_id=0)
try:
    for noc in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1):
        a = d.get_optimal_dram_bank_to_logical_worker_assignment(noc)
        print("NOC", noc, [(c.x, c.y) for c in a])
except Exception as e:
    print("ERR", repr(e))
ttnn.close_device(d)
