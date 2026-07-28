import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
for W in (64, 128, 1024):
    shape = (1, 1, 32, W)
    x = torch.randn(shape, dtype=torch.float32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(tt_x, epsilon=1e-6)).to(torch.float64)
    xd = x.to(torch.float64)
    true_rms = torch.sqrt((xd**2).mean(dim=-1, keepdim=True) + 1e-6)
    # implied rms from the device output
    implied = xd / out
    print(f"W={W}")
    print(
        "   per-row implied/true rms  min/med/max:",
        (implied / true_rms).amin().item(),
        (implied / true_rms).median().item(),
        (implied / true_rms).amax().item(),
    )
    # per-row spread of implied rms (is it a clean per-row scalar?)
    rowspread = (implied.amax(dim=-1) - implied.amin(dim=-1)) / implied.median(dim=-1).values
    print("   within-row implied-rms rel spread max:", rowspread.amax().item())
    # what mean(x^2) would produce that rms
    implied_ms = implied.median(dim=-1).values ** 2 - 1e-6
    true_ms = (xd**2).mean(dim=-1)
    print(
        "   implied mean(x^2)/true:",
        (implied_ms / true_ms).amin().item(),
        (implied_ms / true_ms).median().item(),
        (implied_ms / true_ms).amax().item(),
    )
ttnn.close_device(device)
