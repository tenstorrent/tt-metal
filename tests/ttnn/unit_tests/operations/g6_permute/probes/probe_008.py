import torch, ttnn
from ttnn.operations.g6_permute import g6_permute

device = ttnn.open_device(device_id=0)
try:
    device.enable_program_cache()
    torch.manual_seed(0)
    shape = (1, 6, 16, 64)
    xs = [torch.randn(shape).to(torch.bfloat16) for _ in range(3)]
    dev = [
        ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for x in xs
    ]
    outs = []
    for i, d in enumerate(dev):
        o = g6_permute(d, (0, 2, 1, 3))
        outs.append(o)
        print("addr", i, d.buffer_address(), o.buffer_address(), "entries", device.num_program_cache_entries())
    for i, (x, o) in enumerate(zip(xs, outs)):
        exp = torch.permute(x, (0, 2, 1, 3)).contiguous()
        got = ttnn.to_torch(o)
        same = torch.equal(got.view(torch.int16), exp.view(torch.int16))
        print("ok" if same else "MISMATCH", i)
finally:
    ttnn.close_device(device)
