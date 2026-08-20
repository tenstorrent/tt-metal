import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
torch.manual_seed(42)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi2
cfg.fp32_dest_acc_en = False
cfg.math_approx_mode = False
for shape in [(1, 1, 32, 7168), (1, 1, 32, 1024), (1, 1, 32, 64)]:
    x = torch.rand(shape, dtype=torch.float32)
    g = torch.rand((1, 1, 1, shape[-1]), dtype=torch.float32)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gt = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    o = ttnn.to_torch(rms_norm(t, gamma=gt, compute_kernel_config=cfg)).to(torch.float32)
    print(
        "RATIO",
        shape,
        round((o / ref).mean().item(), 5),
        "pcc",
        round(torch.corrcoef(torch.stack([o.flatten(), ref.flatten()]))[0, 1].item(), 6),
    )
ttnn.close_device(device)
