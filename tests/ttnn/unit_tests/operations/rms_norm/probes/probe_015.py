import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)


def cfg(acc, fid=ttnn.MathFidelity.HiFi4):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = fid
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


for shape in [(1, 1, 32, 1024), (1, 1, 32, 7168), (1, 1, 160, 11008)]:
    W = shape[-1]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    xf = x.float()
    true_s = (xf**2).sum(-1).reshape(-1)  # sum of squares per row
    true_scale = 1.0 / torch.sqrt(true_s / W + 1e-6)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(f"[{shape}] true_scale[:4]={[round(v,6) for v in true_scale[:4].tolist()]}")
    for acc in (True, False):
        a = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg(acc))).float()
        af = a.reshape(-1, W)
        xr = xf.reshape(-1, W)
        big = xr.abs() > 1.0
        r = torch.where(big, af / torch.where(big, xr, torch.ones_like(xr)), torch.full_like(xr, float("nan")))
        got_scale = r.nanmedian(dim=-1).values
        rel = (got_scale - true_scale) / true_scale
        # implied sum-of-squares the kernel must have used
        implied_s = (1.0 / got_scale**2 - 1e-6) * W
        print(f"   acc={int(acc)}: got_scale[:4]={[round(v,6) for v in got_scale[:4].tolist()]}")
        print(
            f"            scale_rel: median={rel.median().item():+.5f} p5={rel.quantile(0.05).item():+.5f} p95={rel.quantile(0.95).item():+.5f}"
        )
        print(
            f"            implied_sum[:4]={[round(v,1) for v in implied_s[:4].tolist()]} vs true {[round(v,1) for v in true_s[:4].tolist()]}"
        )
ttnn.close_device(device)
