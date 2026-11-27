import json, torch, ttnn
from ttnn.operations.g6_zeros_like import g6_zeros_like
from ttnn.operations.g6_zeros_like.g6_zeros_like import describe_plan
from ttnn.operations.g6_zeros_like import g6_zeros_like_program_descriptor as pd

device = ttnn.open_device(device_id=0)
x = ttnn.from_torch(
    torch.zeros((1, 12, 128, 128)),
    dtype=ttnn.float32,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
plan = describe_plan(x)
out = []
for p in plan:
    lc = ttnn.CoreCoord(*p["core"])
    vc = device.worker_core_from_logical_core(lc)
    out.append(dict(p, virt=(vc.x, vc.y)))
json.dump(out, open("/tmp/g6zl/plan_fp32.json", "w"))
anch = pd._bank_anchors(device)
print("anchors", [(a.x, a.y) for a in anch])
print("grid", device.compute_with_storage_grid_size())
from collections import Counter

print(Counter((p["bank"], p["num_pages"]) for p in plan))
ttnn.close_device(device)
