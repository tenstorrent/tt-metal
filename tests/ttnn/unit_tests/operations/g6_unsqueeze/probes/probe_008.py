import torch, ttnn
from ttnn.operations.g6_unsqueeze import g6_unsqueeze, g6_unsqueeze_plan

device = ttnn.open_device(device_id=0)
try:
    print("device id:", device.id())
    for shape in [(1, 512), (1, 256), (40,), (1, 1, 100)]:
        x = torch.randn(shape, dtype=torch.float32)
        t = ttnn.from_torch(
            x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        print(g6_unsqueeze_plan(t).report())
        out = ttnn.to_torch(g6_unsqueeze(t, -1))
        exp = torch.unsqueeze(x, -1)
        print(shape, list(out.shape), "bit-exact:", torch.equal(out.view(torch.int32), exp.view(torch.int32)))
finally:
    ttnn.close_device(device)
