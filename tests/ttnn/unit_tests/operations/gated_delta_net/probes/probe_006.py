import ttnn
from ttnn.operations.gated_delta_net import gated_delta_net_program_descriptor as d

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("GRID", g.x, g.y)
    cores = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]
    v = [device.worker_core_from_logical_core(c) for c in cores]
    print("MAP", [(c.x, c.y, w.x, w.y) for c, w in zip(cores, v)][:40])
finally:
    ttnn.close_device(device)
