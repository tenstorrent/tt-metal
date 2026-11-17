import torch, ttnn
from ttnn.operations.rms_norm.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
W = 8192
x = torch.zeros(1, 1, 32, W)
# column block j (of 8) gets constant (j+1): partial_j proportional to (j+1)^2
for j in range(8):
    x[..., j * 1024 : (j + 1) * 1024] = j + 1
tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()
ref = x / torch.sqrt((x**2).mean(-1, keepdim=True))
print("ratio per block", [(o[0, 0, 0, j * 1024] / ref[0, 0, 0, j * 1024]).item() for j in range(8)])
print("ratio rows", o[0, 0, :6, 0] / ref[0, 0, :6, 0])
ttnn.close_device(device)
