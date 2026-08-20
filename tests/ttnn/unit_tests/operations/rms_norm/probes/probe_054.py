"""Integrated-op precision A/B: pre-graduation compute+CB set vs the shipped fused one.

Row-scale bias by the SAME least-squares estimator the op's precision matrix uses
(tests/.../test_rms_norm_precision_matrix.py::_row_scale_bias), at Wt = 32/64/128/224
and fp32_dest_acc_en=False.  The widths are MASKED (W = 32*Wt - 1) because after the
W-split graduation an ALIGNED wide shape solves to Regime A on G>1 cores and never
reaches the code under test; masked widths are always Regime B, G=1.
"""
import sys, os

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath("ttnn/ttnn/operations/rms_norm/perf_experiments/fused_sumsq/graduation_ab.py")),
        "ttnn/ttnn/operations/rms_norm/perf_experiments/fused_sumsq",
    ),
)
import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as opd
import graduation_ab as gab

EPS = 1e-6


def row_scale_bias(xg, out, s_ref):
    gf = xg.reshape(-1, xg.shape[-1])
    of = out.reshape(-1, out.shape[-1])
    k = (of * gf).sum(-1) / (gf * gf).sum(-1).clamp_min(1e-30)
    return ((k / s_ref.reshape(-1)) - 1.0).mean().item()


dev = ttnn.open_device(device_id=0)
arms = gab._arms()
shipped_kd, shipped_layout, shipped_solve = opd.KERNEL_DIR, opd._cb_layout, opd._solve
try:
    print(f"{'Wt':>5s} {'W':>6s} {'arm':12s} {'plan':38s} {'pcc':>10s} {'row_scale_bias':>16s} {'rel_rms':>9s}")
    for Wt in (32, 64, 128, 224):
        W = 32 * Wt - 1
        torch.manual_seed(0)
        tx = torch.randn((1, 1, 32, W)).to(torch.bfloat16)
        tg = torch.randn(W).reshape(1, 1, 1, W).to(torch.bfloat16)
        x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        g = ttnn.from_torch(tg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        x32 = ttnn.to_torch(x).float()
        g32 = ttnn.to_torch(g).float()[..., :W]
        s_ref = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
        expected = x32 * s_ref * g32
        for arm in ("base", "fused"):
            kd, layout_fn, solve_fn = arms[arm]
            opd.KERNEL_DIR = shipped_kd if kd is None else kd
            opd._cb_layout = layout_fn
            opd._solve = solve_fn
            cfg = ttnn.ComputeConfigDescriptor()
            cfg.math_fidelity = ttnn.MathFidelity.HiFi2
            cfg.fp32_dest_acc_en = False
            cfg.math_approx_mode = False
            p = opd.blocking_plan(x, g, x, dev, cfg, None)
            out = rms_norm(x, gamma=g, epsilon=EPS, compute_kernel_config=cfg)
            a = ttnn.to_torch(out).float()
            ttnn.deallocate(out)
            err = (a - expected).abs()
            rel = err.pow(2).mean().sqrt().item() / expected.std().clamp_min(1e-30).item()
            pcc = torch.corrcoef(torch.stack([expected.flatten().double(), a.flatten().double()]))[0, 1].item()
            bias = row_scale_bias(x32 * g32, a, s_ref)
            plan = f"reg={p.regime} G={p.group_size} wr={p.WT_REDUCE_BLOCK} nchunk={p.Wt_core//p.WT_REDUCE_BLOCK} rva={p.reduce_via_add}"
            print(f"{Wt:5d} {W:6d} {arm:12s} {plan:38s} {pcc:10.6f} {bias*100:+15.4f}% {rel:9.5f}")
        ttnn.deallocate(x)
        ttnn.deallocate(g)
finally:
    opd.KERNEL_DIR, opd._cb_layout, opd._solve = shipped_kd, shipped_layout, shipped_solve
    ttnn.close_device(dev)
