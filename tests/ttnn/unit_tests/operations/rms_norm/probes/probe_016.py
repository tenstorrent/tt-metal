import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def cfg_of(acc):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    W = 7168
    tx = torch.randn((1, 1, 32, W)).to(torch.bfloat16)
    x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    x32 = tx.float()
    s_ref = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)[0, 0, :, 0]
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
    g1 = ttnn.from_torch(
        torch.ones(1, 1, 1, W).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
    )
    for acc in [True, False]:
        for tag, gg in [("nogamma", None), ("ones", g1)]:
            out = ttnn.to_torch(rms_norm(x, gamma=gg, compute_kernel_config=cfg_of(acc))).float()
            # per-row implied scale: least-squares out ~ k * x  (robust, no division blowups)
            num = (out[0, 0] * x32[0, 0]).sum(-1)
            den = (x32[0, 0] * x32[0, 0]).sum(-1)
            k = num / den
            relerr = k / s_ref - 1.0
            print(
                f"RESULT acc={acc!s:<5} {tag:<7} rowscale_relerr: mean={relerr.mean():+.5f} "
                f"absmax={relerr.abs().max():.5f} first8={[round(v,4) for v in relerr[:8].tolist()]}"
            )
            # residual once the per-row scale is removed
            resid = (out[0, 0] - k[:, None] * x32[0, 0]).pow(2).mean().sqrt().item() / normed.std().item()
            print(f"RESULT acc={acc!s:<5} {tag:<7} resid_after_rowscale={resid:.5f}")
finally:
    ttnn.close_device(dev)
