import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi2
cfg.fp32_dest_acc_en = False
cfg.math_approx_mode = False
for shape, lay, glay, dt in [
    ((32, 17), ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ((1, 1, 32, 7168), ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ((5, 3, 928, 544), ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ((1, 1, 8192, 4095), ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ((1, 1, 1024, 1024), ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
]:
    x = torch.rand(shape, dtype=torch.float32)
    g = torch.rand(tuple(list(shape[:-1] and [1, 1, 1]) + [shape[-1]]), dtype=torch.float32)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g
    t = ttnn.from_torch(x, dtype=dt, layout=lay, device=device)
    gt = ttnn.from_torch(g, dtype=dt, layout=glay, device=device)
    a = ttnn.to_torch(rms_norm(t, gamma=gt, compute_kernel_config=cfg)).to(torch.float32)
    b = ttnn.to_torch(rms_norm(t, gamma=gt, compute_kernel_config=cfg, _levers=dict(no_reconfig=0))).to(torch.float32)
    print(
        "RES",
        shape,
        dt,
        "identical",
        bool(torch.equal(a, b)),
        "pcc",
        round(torch.corrcoef(torch.stack([a.flatten(), ref.flatten()]))[0, 1].item(), 6),
    )
ttnn.close_device(device)
