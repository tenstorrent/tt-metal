import torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn

device = ttnn.open_device(device_id=0)


def cfg(approx, fp32dest=False, fid=ttnn.MathFidelity.HiFi4):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = fid
    c.fp32_dest_acc_en = fp32dest
    c.math_approx_mode = approx
    return c


try:
    torch.manual_seed(0)
    for dt in (ttnn.float32, ttnn.bfloat16):
        for W in (32, 64, 128, 256):
            for approx in (True, False):
                for eps in (1e-12, 1e-5):
                    x = torch.randn(1, 1, 32, W, dtype=torch.float32)
                    exp = torch_rms_norm_ttnn(
                        x.to(torch.float32) if dt == ttnn.float32 else x.to(torch.bfloat16), epsilon=eps
                    )
                    tx = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
                    got = ttnn.to_torch(rms_norm_ttnn(tx, epsilon=eps, compute_kernel_config=cfg(approx))).float()
                    t = exp.float()
                    bad = (~torch.isfinite(got)).sum().item()
                    pcc = torch.corrcoef(torch.stack([got.flatten(), t.flatten()]))[0, 1].item()
                    rel = ((got - t).pow(2).mean().sqrt() / t.pow(2).mean().sqrt()).item()
                    flag = "  <== BROKEN" if (bad or not (pcc > 0.99)) else ""
                    print(
                        f"RES dt={str(dt).split('.')[-1]:9s} W={W:4d} approx={str(approx):5s} eps={eps:g} "
                        f"nonfinite={bad:4d} pcc={pcc:9.6f} rel_rms={rel:9.5f}{flag}"
                    )
finally:
    ttnn.close_device(device)
