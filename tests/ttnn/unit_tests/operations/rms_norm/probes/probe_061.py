import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
torch.manual_seed(42)
for shape, dt in [
    ((1, 1, 32, 64), ttnn.float32),
    ((1, 1, 32, 64), ttnn.bfloat16),
    ((1, 1, 128, 256), ttnn.float32),
    ((1, 1, 32, 1024), ttnn.bfloat16),
]:
    x = torch.rand(shape, dtype=torch.float32)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    t = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    o = ttnn.to_torch(rms_norm(t)).to(torch.float32)
    k = o.flatten()[:64] / ref.flatten()[:64]
    print(
        shape,
        dt,
        "ratio mean",
        round(k.mean().item(), 5),
        "min",
        round(k.min().item(), 5),
        "max",
        round(k.max().item(), 5),
    )
ttnn.close_device(device)
