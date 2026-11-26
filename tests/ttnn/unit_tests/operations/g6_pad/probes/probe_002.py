import torch, ttnn
from ttnn.operations.g6_pad import g6_pad

device = ttnn.open_device(device_id=0)
try:
    shape = (1, 1, 8, 62080)
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    tiny = torch.finfo(torch.bfloat16).tiny
    specials = torch.tensor([-0.0, float("inf"), float("-inf"), tiny, tiny / 4, -tiny / 8]).to(torch.bfloat16)
    flat = x.reshape(-1)
    flat[:6] = specials
    x = flat.reshape(shape)
    print("PROBE specials bits", specials.view(torch.int16).tolist())
    for poison in (None, 1000.0):
        t = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if poison is not None:
            t = ttnn.fill_implicit_tile_padding(t, poison)
        back = ttnn.to_torch(t)
        d = (back.view(torch.int16) != x.view(torch.int16)).nonzero()
        print(
            "PROBE poison",
            poison,
            "input roundtrip mismatches",
            d.tolist()[:6],
            back.reshape(-1)[:6].view(torch.int16).tolist(),
        )
        out = g6_pad(t, ((0, 0), (0, 0), (0, 8), (0, 0)), value=0)
        o = ttnn.to_torch(out)
        exp = torch.nn.functional.pad(x, [0, 0, 0, 8], value=0)
        d = (o.view(torch.int16) != exp.view(torch.int16)).nonzero()
        print("PROBE poison", poison, "output mismatches", d.tolist()[:6], o.reshape(-1)[:6].view(torch.int16).tolist())
finally:
    ttnn.close_device(device)
