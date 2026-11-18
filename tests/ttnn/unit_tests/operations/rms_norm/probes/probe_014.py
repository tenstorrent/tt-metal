import torch, ttnn
from ttnn.operations.rms_norm.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
)
W = 5120
x = torch.ones(1, 1, 32, W).bfloat16()
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()
print("RES ones", o.mean().item(), o.min().item(), o.max().item())
ttnn.close_device(device)
