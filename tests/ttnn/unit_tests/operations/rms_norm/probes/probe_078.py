import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
gx = grid.x


def torch_rms(x, g, eps):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y * g if g is not None else y


# Force RM Regime B on clean mult-of-gx widths to isolate: is RM-B scaling correct
# when Wt_pad==Wt (no padding)? Wt=256 (W=8192) and Wt=128 (W=4096), gx=8.
eps = 1e-5
for Wt in [128, 256, 264]:  # 264 = 8*33 mult of gx but not power-of-2; 128/256 clean
    W = 32 * Wt
    Wt_pad = ((Wt + gx - 1) // gx) * gx
    pd._FORCE_REGIME = "B"
    x = torch.randn(1, 1, 32, W)
    for layout, ttl in [("RM", ttnn.ROW_MAJOR_LAYOUT), ("TILE", ttnn.TILE_LAYOUT)]:
        tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttl, device=device)
        r = ttnn.to_torch(rms_norm(tx, epsilon=eps)).float()
        exp = torch_rms(x, None, eps)
        maxerr = (r - exp).abs().max().item()
        print(f"Wt={Wt} W={W} Wt_pad={Wt_pad} pad={Wt_pad-Wt} {layout} FORCED_B maxerr={maxerr:.5f}")
    pd._FORCE_REGIME = None

# all-ones RM-B at a clean width to see exact scale
pd._FORCE_REGIME = "B"
W = 32 * 256
x1 = torch.ones(1, 1, 32, W)
tx = ttnn.from_torch(x1, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
r = ttnn.to_torch(rms_norm(tx, epsilon=eps)).float()
print(f"clean Wt=256 RM-B all-ones out sample={r.flatten()[:4].tolist()} maxerr={(r-1.0).abs().max().item():.5f}")
pd._FORCE_REGIME = None
ttnn.close_device(device)
