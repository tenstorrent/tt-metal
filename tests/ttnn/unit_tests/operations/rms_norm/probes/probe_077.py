import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi4
cfg.fp32_dest_acc_en = False
cfg.math_approx_mode = False
for shape, lay, glay in [
    ((5, 3, 928, 544), ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    ((5, 3, 928, 544), ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    ((7, 1, 352, 1184), ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    ((1, 1, 1023, 416), ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
]:
    x = torch.rand(shape, dtype=torch.float32)
    g = torch.rand((1, 1, 1, shape[-1]), dtype=torch.float32)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=lay, device=device)
    gt = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=glay, device=device)
    out = ttnn.to_torch(rms_norm(t, gamma=gt, compute_kernel_config=cfg)).to(torch.float32)
    print(
        "RES",
        shape,
        lay,
        glay,
        "pcc",
        round(torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item(), 6),
    )
ttnn.close_device(device)
