import torch, ttnn
from ttnn.operations.rms_norm.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
)
for W in (5120,):
    for v in (1.0, 0.5, 0.25, 2.0):
        x = torch.full((1, 1, 32, W), v).bfloat16()
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()
        print("RES", W, v, o.mean().item())
# partial: only first 128 cols ones, rest zeros
for nz in (4096, 4128, 4608, 5120):
    x = torch.zeros(1, 1, 32, 5120)
    x[..., :nz] = 1
    x = x.bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()
    print("RES nz", nz, o[..., 0].mean().item(), "expected", (5120 / nz) ** 0.5)
ttnn.close_device(device)
