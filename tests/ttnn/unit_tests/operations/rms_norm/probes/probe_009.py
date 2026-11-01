import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)


def cfg(f, fid):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = fid
    c.fp32_dest_acc_en = f
    c.math_approx_mode = False
    return c


for shape in [(1, 1, 64, 256), (1, 1, 32, 7168), (1, 1, 640, 6144), (1, 1, 8192, 1024)]:
    for dt, f, fid in [
        (ttnn.bfloat16, False, ttnn.MathFidelity.HiFi2),
        (ttnn.bfloat16, False, ttnn.MathFidelity.HiFi4),
        (ttnn.bfloat8_b, True, ttnn.MathFidelity.HiFi4),
        (ttnn.bfloat8_b, False, ttnn.MathFidelity.HiFi4),
    ]:
        torch.manual_seed(0)
        x = torch.randn(shape).to(torch.bfloat16)
        g = torch.randn(1, 1, 1, shape[-1]).to(torch.bfloat16)
        tx = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        tg = ttnn.from_torch(g, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        xr = ttnn.to_torch(tx).float()
        gr = ttnn.to_torch(tg).float().reshape(-1)
        exp = xr / torch.sqrt((xr * xr).mean(-1, keepdim=True) + 1e-6) * gr
        out = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=cfg(f, fid))).float()
        e = (out - exp).abs()
        p = torch.corrcoef(torch.stack([out.flatten().double(), exp.flatten().double()]))[0, 1].item()
        rr = (e.pow(2).mean().sqrt() / exp.pow(2).mean().sqrt()).item()
        print(f"R {shape} {dt} fp32_dest={f} {fid}: PCC={p:.6f} max_abs={e.max().item():.4f} rel_rms={rr:.5f}")
ttnn.close_device(dev)
