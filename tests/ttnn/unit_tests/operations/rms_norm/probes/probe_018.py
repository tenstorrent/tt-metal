import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm_program_descriptor import blocking_plan


def cfg_of(acc):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    # W=1016 / 2040 are w_non_aligned -> FORCE Regime B at small Wt.
    # W=1024 / 2048 are tile-aligned  -> Regime A at the same width.
    for W in [1016, 1024, 2040, 2048, 4088, 4096, 7168]:
        shape = (1, 1, 32, W)
        tx = torch.randn(shape).to(torch.bfloat16)
        x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        g = ttnn.from_torch(
            torch.ones(1, 1, 1, W).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
        )
        x32 = tx.float()
        s_ref = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)[0, 0, :, 0]
        for acc in [True, False]:
            p = blocking_plan(x, g, None, dev, cfg_of(acc), None)
            out = ttnn.to_torch(rms_norm(x, gamma=g, compute_kernel_config=cfg_of(acc))).float()
            num = (out[0, 0] * x32[0, 0]).sum(-1)
            den = (x32[0, 0] * x32[0, 0]).sum(-1)
            k = num / den
            bias = (k / s_ref - 1).mean().item()
            # scale bias -> implied sumsq bias
            sumsq_bias = (1.0 + bias) ** -2 - 1.0
            print(
                f"RESULT W={W:>5} Wt={p.Wt_core:>3} regime={p.regime} acc={acc!s:<5} wr={p.WT_REDUCE_BLOCK:>3} "
                f"nchunk={p.Wt_core//p.WT_REDUCE_BLOCK:>2} scale_bias={bias:+.5f} implied_sumsq_bias={sumsq_bias:+.4f}"
            )
finally:
    ttnn.close_device(dev)
