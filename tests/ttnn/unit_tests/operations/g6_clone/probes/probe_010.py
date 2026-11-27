# DRAM endpoint placement per NoC: optimal worker anchors for NOC_0 and NOC_1.
import ttnn

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("grid", g.x, g.y)
    for noc in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1):
        cores = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, noc)
        print("optimal", noc, [(c.x, c.y) for c in cores])
finally:
    ttnn.close_device(device)
