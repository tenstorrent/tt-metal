import torch, ttnn
from ttnn.operations.rms_norm import rms_norm, default_compute_kernel_config
from ttnn.operations.rms_norm.rms_norm_program_descriptor import blocking_plan

torch.manual_seed(42)
for shape, dt in [((1, 1, 32, 64), ttnn.float32), ((1, 1, 32, 64), ttnn.bfloat16), ((1, 1, 128, 256), ttnn.float32)]:
    x = torch.rand(shape, dtype=torch.float32)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    t = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    o = ttnn.to_torch(rms_norm(t))
    k = o.flatten()[:100] / ref.flatten()[:100]
    print(shape, dt, "ratio mean", k.mean().item(), "min", k.min().item(), "max", k.max().item())
