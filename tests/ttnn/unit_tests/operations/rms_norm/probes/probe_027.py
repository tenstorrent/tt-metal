import torch, ttnn
from ttnn.operations.rms_norm.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
W = 8192
x = torch.zeros(1, 1, 32, W)
for j in range(8):
    x[..., j * 1024 : (j + 1) * 1024] = j + 1
tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()
print("o", o[0, 0, 0, ::1024])
ttnn.close_device(device)
