import torch, ttnn
from ttnn.operations.rms_norm.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
)
W = 5120
for nz in (512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120):
    x = torch.zeros(1, 1, 32, W)
    x[..., :nz] = 1
    x = x.bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()
    v = o[..., 0].mean().item()
    print("RES", nz, "mean implied", (1 / v) ** 2, "true", nz / W)
ttnn.close_device(device)
