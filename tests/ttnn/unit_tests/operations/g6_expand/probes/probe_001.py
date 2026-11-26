import torch, ttnn
from ttnn.operations.g6_expand import g6_expand

device = ttnn.open_device(device_id=0)
try:
    print("grid", device.compute_with_storage_grid_size())
    for W in [512, 196]:
        torch.manual_seed(0)
        x = torch.randn((1, 1, 1, W)).to(torch.bfloat16)
        xt = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = g6_expand(xt, [1, 1, W, W])
        got = ttnn.to_torch(out)
        exp = x.expand(1, 1, W, W).contiguous()
        diff = got.view(torch.int16) != exp.view(torch.int16)
        print("W", W, "plan", g6_expand.last_plan.num_cores, "mismatches", int(diff.sum()), "of", diff.numel())
        if diff.any():
            idx = diff.nonzero()[:5].tolist()
            print("first", idx, [(got[tuple(i)].item(), exp[tuple(i)].item()) for i in idx])
finally:
    ttnn.close_device(device)
