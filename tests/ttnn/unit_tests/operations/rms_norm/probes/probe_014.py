import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)


def cfg(acc):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


for shape in [(1, 1, 160, 11008), (1, 1, 32, 7168), (1, 1, 32, 1024)]:
    W = shape[-1]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    # no gamma -> isolate the stat
    xf = x.float()
    true_scale = 1.0 / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(
        f"[{shape}] true sum(x^2) = {(xf**2).sum(-1).flatten()[:4].tolist()}  true_scale={true_scale.flatten()[:4].tolist()}"
    )
    for acc in (True, False):
        out = rms_norm(tx, compute_kernel_config=cfg(acc))
        a = ttnn.to_torch(out).float()
        # implied per-row scale
        m = xf.abs() > 0.3
        got_scale = torch.where(m, a / xf.clamp(min=1e-9).where(xf.abs() > 1e-9, torch.ones_like(xf)), torch.nan)
        gs = torch.nanmedian(got_scale.reshape(-1, W), dim=-1).values
        ts = true_scale.reshape(-1, 1).flatten()
        relerr = (gs - ts) / ts
        print(f"   acc={int(acc)}: implied_scale[:4]={gs[:4].tolist()}")
        print(f"            scale_relerr: median={relerr.median().item():.5f} max={relerr.abs().max().item():.5f}")
ttnn.close_device(device)
