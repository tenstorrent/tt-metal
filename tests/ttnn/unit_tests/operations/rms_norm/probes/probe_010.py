import torch, ttnn
from ttnn.operations.rms_norm.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
)
for W in (1024, 2048, 5120):
    x = torch.ones(1, 1, 32, W).bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()
    print("RES ones", W, o.mean().item())
    x = torch.randn(1, 1, 32, W).bfloat16()
    ref = x.float() / torch.sqrt((x.float() ** 2).mean(-1, keepdim=True) + 1e-6)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()
    print("RES randn", W, (o / ref).mean().item())
    x = (torch.rand(1, 1, 32, W) + 1).bfloat16()
    ref = x.float() / torch.sqrt((x.float() ** 2).mean(-1, keepdim=True) + 1e-6)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()
    print("RES pos", W, (o / ref).mean().item())
ttnn.close_device(device)
