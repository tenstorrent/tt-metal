import torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn

device = ttnn.open_device(device_id=0)


def cfg(approx):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = False
    c.math_approx_mode = approx
    return c


def run(x, dt, eps, approx):
    exp = torch_rms_norm_ttnn(x, epsilon=eps).float()
    tx = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(rms_norm_ttnn(tx, epsilon=eps, compute_kernel_config=cfg(approx))).float()
    return got, exp


try:
    # 1. width sweep, fp32/approx/eps=1e-12, fixed seed
    for W in (32, 64, 96, 128, 160, 192, 224, 256, 512, 1024, 50, 17):
        torch.manual_seed(0)
        x = torch.randn(1, 1, 32, W, dtype=torch.float32)
        got, exp = run(x, ttnn.float32, 1e-12, True)
        bad = (~torch.isfinite(got)).sum().item()
        print(f"RES W={W:5d} nonfinite={bad:5d}/{got.numel()}")
    # 2. epsilon sweep at the broken width
    for eps in (0.0, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5):
        torch.manual_seed(0)
        x = torch.randn(1, 1, 32, 64, dtype=torch.float32)
        got, exp = run(x, ttnn.float32, eps, True)
        bad = (~torch.isfinite(got)).sum().item()
        print(f"EPS eps={eps:g} nonfinite={bad:5d}")
    # 3. seed sweep -- is it data-dependent?
    for s in range(6):
        torch.manual_seed(s)
        x = torch.randn(1, 1, 32, 64, dtype=torch.float32)
        got, exp = run(x, ttnn.float32, 1e-12, True)
        bad = (~torch.isfinite(got)).sum().item()
        print(f"SEED {s} nonfinite={bad:5d}")
finally:
    ttnn.close_device(device)
