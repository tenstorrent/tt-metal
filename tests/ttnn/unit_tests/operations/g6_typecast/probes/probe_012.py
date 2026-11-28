import torch, ttnn
from ttnn.operations.g6_typecast import g6_typecast

dev = ttnn.open_device(device_id=0)
try:
    for seed, shape in ((1, (1, 1, 1024, 1024)), (2, (3, 5, 67, 45)), (3, (1, 1, 32, 128))):
        g = torch.Generator().manual_seed(seed)
        x = torch.randint(-(2**31), 2**31, shape, generator=g, dtype=torch.int64).to(torch.int32)
        flat = x.view(-1)
        w = [
            33685505,
            2**24 + 2**16 + 1,
            257,
            -(2**31),
            2**31 - 1,
            2**24 + 1,
            2**24 + 3,
            -(2**24 + 1),
            0,
            -1,
            16777217,
            2**31 - 128,
            2**31 - 129,
        ]
        flat[: len(w)] = torch.tensor(w, dtype=torch.int32)
        t = ttnn.from_torch(x, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=dev)
        y = ttnn.to_torch(g6_typecast(t, ttnn.bfloat16))
        e = x.to(torch.float32).to(torch.bfloat16)
        bad = (y.view(torch.int16) != e.view(torch.int16)).sum().item()
        print("RESULT", shape, "bad", bad, "of", x.numel())
finally:
    ttnn.close_device(dev)
