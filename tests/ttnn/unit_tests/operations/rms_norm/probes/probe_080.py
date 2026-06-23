import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
gx = grid.x


def torch_rms(x, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def pcc(a, b):
    a = a.flatten().float() - a.flatten().float().mean()
    b = b.flatten().float() - b.flatten().float().mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()


eps = 1e-5
torch.manual_seed(0)
# Clean shape Wt=256 (W=8192): Wt_pad==Wt; confirm correct (byte-identity by construction).
W = 8192
x = torch.randn(1, 1, 32, W)
for layout, ttl in [("TILE", ttnn.TILE_LAYOUT), ("RM", ttnn.ROW_MAJOR_LAYOUT)]:
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttl, device=device)
    r = ttnn.to_torch(rms_norm(tx, epsilon=eps)).float()
    me = (r - torch_rms(x, eps)).abs().max().item()
    print(f"CLEAN Wt=256 {layout} maxerr={me:.5f} {'PASS' if me<=0.05 else 'FAIL'}")
# Wt=264 isolation (the latent RM-B bug we found): now must pass.
pd._FORCE_REGIME = "B"
W = 32 * 264
x = torch.randn(1, 1, 32, W)
for layout, ttl in [("RM", ttnn.ROW_MAJOR_LAYOUT), ("TILE", ttnn.TILE_LAYOUT)]:
    tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttl, device=device)
    r = ttnn.to_torch(rms_norm(tx, epsilon=eps)).float()
    me = (r - torch_rms(x, eps)).abs().max().item()
    print(f"Wt264 {layout} FORCED_B maxerr={me:.5f} pcc={pcc(r,torch_rms(x,eps)):.5f} {'PASS' if me<=0.05 else 'FAIL'}")
pd._FORCE_REGIME = None
ttnn.close_device(device)
