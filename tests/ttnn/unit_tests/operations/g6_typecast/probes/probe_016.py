import sys, torch, ttnn

sys.path.insert(0, "ttnn/ttnn/operations/g6_typecast/perf_experiments/subtile_split")
import subsplit
from ttnn.operations.g6_typecast import g6_typecast

dev = ttnn.open_device(device_id=0)
try:
    r = torch.arange(32).view(32, 1).float()
    c = torch.arange(32).view(1, 32).float()
    x = (2.0 ** (r % 8)) * (1 + c / 64) * torch.where((r // 16 + c // 16) % 2 == 0, 1.0, -1.0)
    x = x.view(1, 1, 32, 32).to(torch.bfloat16)
    tin = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    subsplit.install()
    for ht in [(32, 16), (16, 32)]:
        subsplit.HALF_TILE = ht
        subsplit.set_mode("half")
        out = ttnn.to_torch(g6_typecast(tin, ttnn.bfloat8_b)).float().view(32, 32)
        ref = x.float().view(32, 32)
        bad = (out - ref).abs() > ref.abs() * 0.05 + 1e-3
        rows = bad.nonzero()
        print("half", ht, "bad", bad.sum().item(), sorted(set(rows[:, 0].tolist())))
finally:
    ttnn.close_device(dev)
