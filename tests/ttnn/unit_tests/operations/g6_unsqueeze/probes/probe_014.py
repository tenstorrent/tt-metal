import torch, ttnn
from ttnn.operations.g6_unsqueeze import g6_unsqueeze_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
for W, mc in [(512, None), (256, None), (512, 8), (512, 3), (512, 5), (4100, None), (13 * 32, None), (500, None)]:
    pd.MAX_CORES = mc
    x = ttnn.from_torch(torch.randn(1, W), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    p = pd.select_plan(x)
    print(W, mc, pd.describe_plan(p).splitlines()[:2], p.core_ranges)
ttnn.close_device(dev)
