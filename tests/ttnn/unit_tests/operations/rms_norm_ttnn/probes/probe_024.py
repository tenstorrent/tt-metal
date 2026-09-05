import os, torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn

MODE = os.environ.get("VMODE", "?")
device = ttnn.open_device(device_id=0)


def cfg(fp32dest):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = fp32dest
    c.math_approx_mode = True
    return c


try:
    for name, mk in [("positive_only", lambda s: torch.rand(s) + 0.5), ("randn        ", lambda s: torch.randn(s))]:
        for shape in [(1, 1, 32, 64), (1, 1, 128, 256)]:
            for lbl, d32 in [("dest16", False), ("dest32", True)]:
                torch.manual_seed(42)
                x = mk(shape).to(torch.float32)
                exp = torch_rms_norm_ttnn(x, epsilon=1e-12).float()
                tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
                got = ttnn.to_torch(rms_norm_ttnn(tx, epsilon=1e-12, compute_kernel_config=cfg(d32))).float()
                rel = ((got - exp).pow(2).mean().sqrt() / exp.pow(2).mean().sqrt()).item()
                bad = (~torch.isfinite(got)).sum().item()
                print(f"ACC[{MODE}] {name} {str(shape):16s} {lbl} rel_rms={rel:.5f} nonfinite={bad}")
    x = torch.randn(1, 1, 128, 512).to(torch.bfloat16)
    exp = torch_rms_norm_ttnn(x, epsilon=1e-12).float()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(rms_norm_ttnn(tx, epsilon=1e-12)).float()
    print(f"ACC[{MODE}] bf16 control rel_rms={((got-exp).pow(2).mean().sqrt()/exp.pow(2).mean().sqrt()).item():.5f}")
finally:
    ttnn.close_device(device)
