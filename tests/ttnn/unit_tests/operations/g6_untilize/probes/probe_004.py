import torch, ttnn
from ttnn.operations.g6_untilize import describe_plan, g6_untilize

device = ttnn.open_device(device_id=0)
try:
    for shape in [(2, 3, 40, 100), (1, 1, 64, 64), (1, 2, 17, 33), (1, 1, 1, 16), (3, 1, 96, 4096)]:
        torch.manual_seed(1)
        x = torch.randint(-(2**31), 2**31, shape, dtype=torch.int64).to(torch.int32).view(torch.uint32)
        xd = ttnn.from_torch(
            x, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        d = describe_plan(xd)
        r = ttnn.to_torch(g6_untilize(xd))
        ok = list(r.shape) == list(shape) and torch.equal(r.view(torch.int32), x.view(torch.int32))
        print(f"SHAPE {shape} {d['rectangle']} {d['knobs']} -> {'OK' if ok else 'MISMATCH'}")
finally:
    ttnn.close_device(device)
