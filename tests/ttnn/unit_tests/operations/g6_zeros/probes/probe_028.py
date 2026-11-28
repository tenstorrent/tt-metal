import ttnn
from ttnn.operations.g6_zeros import g6_zeros as m

device = ttnn.open_device(device_id=0)
try:
    seeds = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_1)
    print("SEEDS", [(c.x, c.y) for c in seeds])
    d = m.describe_plan((1, 4, 256, 128), dtype=ttnn.bfloat16, device=device)
    print("PLAN", {k: v for k, v in d.items() if k != "roles"})
finally:
    ttnn.close_device(device)
