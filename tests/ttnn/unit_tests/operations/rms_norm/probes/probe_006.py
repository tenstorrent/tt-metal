import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def ref(x, g, eps=1e-6):
    xf = x.to(torch.float64)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    return (xf / rms) * g.to(torch.float64).reshape(-1)


dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    for shape in ((1, 1, 128, 512), (1, 1, 32, 64)):
        W = shape[-1]
        x = torch.randn(shape, dtype=torch.float32)
        g = torch.randn(W, dtype=torch.float32)
        exp = ref(x, g)
        tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        tg = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        for approx in (False, True):
            c = ttnn.ComputeConfigDescriptor()
            c.math_fidelity = ttnn.MathFidelity.HiFi4
            c.fp32_dest_acc_en = True
            c.math_approx_mode = approx
            got = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=c)).to(torch.float64)
            err = (got - exp).abs()
            rel_rms = (torch.sqrt((err**2).mean()) / exp.std()).item()
            r = got[exp.abs() > 1e-6] / exp[exp.abs() > 1e-6]
            print(
                f"RES shape={shape} approx={approx}: rel_rms={rel_rms:.3e} max_abs={err.max().item():.3e} ratio_med={r.median().item():.8f} ratio_p5={torch.quantile(r,0.05).item():.8f} ratio_p95={torch.quantile(r,0.95).item():.8f}"
            )
finally:
    ttnn.close_device(dev)
