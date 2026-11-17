import torch, ttnn
from ttnn.operations.rms_norm.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
for fp32 in (True, False):
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32, math_approx_mode=False
    )
    for W in (4096, 4352, 5120, 8192):
        x = torch.ones(1, 1, 32, W).bfloat16()
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()
        print("RES", fp32, W, o.mean().item())
ttnn.close_device(device)
