import ttnn, torch
from ttnn.operations.rms_norm.rms_norm import rms_norm

d = ttnn.open_device(device_id=0)
x = ttnn.from_torch(torch.randn(1, 1, 8192, 1024), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)
g = ttnn.from_torch(torch.randn(1, 1, 1, 1024), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi2
cfg.fp32_dest_acc_en = False
ref = ttnn.to_torch(x).float()
ref = ref * torch.rsqrt(ref.pow(2).mean(-1, keepdim=True) + 1e-6) * ttnn.to_torch(g).float()
for arm, lv in (("default", None), ("D20 force_regime", dict(force_regime=1))):
    out = ttnn.to_torch(rms_norm(x, gamma=g, compute_kernel_config=cfg, _levers=lv)).float()
    pcc = torch.corrcoef(torch.stack([ref.flatten().double(), out.flatten().double()]))[0, 1].item()
    print(f"D20 arm {arm}: pcc={pcc:.6f}")
ttnn.close_device(d)
