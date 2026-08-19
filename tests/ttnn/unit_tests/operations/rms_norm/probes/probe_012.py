import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    for W in [1024, 5120, 7168, 11008]:
        shape = (1, 1, 32, W)
        tx = torch.randn(shape).to(torch.bfloat16)
        x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        x32 = tx.float()
        s_ref = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)  # per-row 1/rms
        e = x32 * s_ref
        for acc_en in [True, False]:
            cfg = ttnn.ComputeConfigDescriptor()
            cfg.math_fidelity = ttnn.MathFidelity.HiFi2
            cfg.fp32_dest_acc_en = acc_en
            cfg.math_approx_mode = False
            out = ttnn.to_torch(rms_norm(x, compute_kernel_config=cfg)).float()
            # implied per-row scale (median over the row is robust to per-element noise)
            keep = x32.abs() > 0.25
            ratio = torch.where(keep, out / x32.clamp(min=1e-9).where(x32 > 0, x32.clamp(max=-1e-9)), torch.nan)
            s_imp = ratio.nanmedian(dim=-1, keepdim=True).values
            scale_err = ((s_imp / s_ref) - 1.0).abs().max().item()
            rms_total = (out - e).pow(2).mean().sqrt().item() / e.std().item()
            # residual after removing the row-scale error => the per-element multiply noise
            resid = (out - x32 * s_imp).pow(2).mean().sqrt().item() / e.std().item()
            print(
                f"RESULT W={W:>5} fp32acc={acc_en!s:<5} rms_total={rms_total:.5f} "
                f"row_scale_err_max={scale_err:.5f} per_elem_resid={resid:.5f}"
            )
finally:
    ttnn.close_device(dev)
