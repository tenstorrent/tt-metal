import torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn

device = ttnn.open_device(device_id=0)
try:
    for dt in (ttnn.float32, ttnn.bfloat16):
        for W in (32, 64, 128, 256, 1024):
            x = torch.zeros(1, 1, 32, W, dtype=torch.float32)
            x[..., 0] = 1.0
            tx = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
            out = ttnn.to_torch(rms_norm_ttnn(tx, epsilon=1e-12)).float()
            hot = out[..., 0].flatten()[0].item()
            eff_W = (1.0 / (hot * hot)) ** -1 if hot else float("nan")
            print(
                f"SPARSE dt={str(dt).split('.')[-1]:9s} W={W:5d} hot={hot:9.4f} expect={W**0.5:9.4f} "
                f"implied_W={hot*hot:9.2f} ratio={hot/(W**0.5):.5f}"
            )
        # dense control
        for W in (64, 256):
            torch.manual_seed(0)
            x = torch.randn(1, 1, 32, W, dtype=torch.float32)
            exp = torch_rms_norm_ttnn(x, epsilon=1e-12)
            tx = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
            got = ttnn.to_torch(rms_norm_ttnn(tx, epsilon=1e-12)).float()
            r = got.flatten() / exp.float().flatten()
            print(
                f"RANDN  dt={str(dt).split('.')[-1]:9s} W={W:5d} ratio_med={r.median().item():.6f} "
                f"ratio_std={r.std().item():.2e}"
            )
    # sparsity sweep at W=64 fp32: how many hot lanes?
    for k in (1, 2, 4, 8, 32, 64):
        x = torch.zeros(1, 1, 32, 64, dtype=torch.float32)
        x[..., :k] = 1.0
        exp = torch_rms_norm_ttnn(x, epsilon=1e-12)
        tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        got = ttnn.to_torch(rms_norm_ttnn(tx, epsilon=1e-12)).float()
        print(
            f"W=64 fp32 hot_lanes={k:3d} got={got[...,0].flatten()[0].item():9.4f} "
            f"expect={exp[...,0].flatten()[0].item():9.4f}"
        )
finally:
    ttnn.close_device(device)
