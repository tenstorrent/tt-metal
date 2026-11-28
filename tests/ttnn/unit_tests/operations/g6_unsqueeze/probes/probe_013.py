import torch, ttnn
import ttnn.operations.g6_unsqueeze.g6_unsqueeze_program_descriptor as pd_mod

dev = ttnn.open_device(device_id=0)
print("PROBE grid", dev.compute_with_storage_grid_size())
for w in (512, 256):
    x = ttnn.from_torch(torch.randn(1, w), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    p = pd_mod.select_plan(x)
    print(pd_mod.describe_plan(p))
    print("PROBE ranges", p.core_ranges)
ttnn.close_device(dev)
