import torch, ttnn
from ttnn.operations.g6_reshape import g6_reshape

device = ttnn.open_device(device_id=0)
for src, dst in [((16, 1, 5120), (1, 1, 16, 5120)), ((1, 1, 16, 4608), (16, 1, 4608))]:
    x = torch.randn(src).to(torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for _ in range(3):
        y = g6_reshape(t, list(dst))
    ttnn.synchronize_device(device)
ttnn.ReadDeviceProfiler(device)
print("DONE")
ttnn.close_device(device)
