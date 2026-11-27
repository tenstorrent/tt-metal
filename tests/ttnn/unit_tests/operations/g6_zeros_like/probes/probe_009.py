import ttnn, json

device = ttnn.open_device(device_id=0)
from ttnn.operations.g6_zeros_like import g6_zeros_like_program_descriptor as pd

for noc in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1):
    a = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, noc)
    print("anchors", noc, [(c.x, c.y) for c in a])
plan = json.load(open("/tmp/g6zl/plan_fp32.json"))
from collections import defaultdict

bb = defaultdict(list)
for p in plan:
    bb[p["bank"]].append(tuple(p["core"]))
for b in sorted(bb):
    print("bank", b, sorted(bb[b]))
ttnn.close_device(device)
