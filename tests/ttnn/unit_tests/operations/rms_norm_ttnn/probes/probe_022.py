import os, torch, ttnn

print("cache env:", os.environ.get("TT_METAL_CACHE"))
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn

device = ttnn.open_device(device_id=0)


def cfg(approx):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = False
    c.math_approx_mode = approx
    return c


try:
    for approx in (True, False):
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            x = torch.randn(1, 1, 32, 64, dtype=torch.float32)
            exp = torch_rms_norm_ttnn(x, epsilon=1e-12).float()
            tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            got = ttnn.to_torch(rms_norm_ttnn(tx, epsilon=1e-12, compute_kernel_config=cfg(approx))).float()
            bad = (~torch.isfinite(got)).sum().item()
            print(f"RES approx={approx} seed={seed} nonfinite={bad}")
finally:
    ttnn.close_device(device)
