#!/usr/bin/env python3
"""CPU equivalence check for the NON-Sinkhorn parts of the mHC path.

Everything in DeepSeek-V4's mHC computation except the Sinkhorn is already pure torch
(model.py: hc_post, hc_head, and the projection+reduction of hc_pre). This script runs
those expressions VERBATIM from model.py and asserts our mhc_reference.py matches them on
random inputs -- no GPU, no tilelang. Combined with check_sinkhorn_vs_kernel.py (the GPU
Sinkhorn check), this closes the loop on the entire op.

Run:  python models/demos/deepseek_v3_d_p/reference/mhc/check_computation_cpu.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mhc_reference import MHCConfig, MHCHead, MHCWrap, parametrize, sinkhorn_knopp  # noqa: E402


# --- mirrors the model's Block.hc_post (the kernel call is factored out) ---
def ds_hc_post(x, residual, post, comb):
    """Block.hc_post. Casts to the residual (entry) dtype, matching modeling_deepseek_v4.py."""
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y.type_as(residual)


def ds_hc_head(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps):
    """model.py ParallelHead.hc_head (verbatim)."""
    shape, dtype = x.size(), x.dtype
    x = x.flatten(2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y.to(dtype)


def ds_hc_pre_project(x, hc_fn, norm_eps):
    """model.py Block.hc_pre, the projection -> mixes (everything before the kernel call)."""
    x = x.flatten(2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    return F.linear(x, hc_fn) * rsqrt


def maxdiff(a, b):
    return (a - b).abs().max().item()


def main():
    torch.manual_seed(0)
    cfg = MHCConfig(dim=64, n=4)
    b, s, n, d = 2, 8, cfg.n, cfg.dim
    results = []

    # 1) hc_post: verbatim DeepSeek vs MHCWrap.hc_post  (NON-circular: their code vs ours)
    wrap = MHCWrap(cfg)
    x = torch.randn(b, s, d)
    residual = torch.randn(b, s, n, d)
    post = 2 * torch.sigmoid(torch.randn(b, s, n))
    comb = sinkhorn_knopp(torch.randn(b, s, n, n), cfg.sinkhorn_iters, cfg.eps)
    results.append(
        (
            "hc_post (expand 1->n + mix)",
            maxdiff(wrap.hc_post(x, residual, post, comb), ds_hc_post(x, residual, post, comb)),
        )
    )

    # 2) hc_head: verbatim DeepSeek vs MHCHead  (NON-circular)
    head = MHCHead(cfg)
    xh = torch.randn(b, s, n, d)
    results.append(
        (
            "hc_head (collapse n->1)",
            maxdiff(head(xh), ds_hc_head(xh, head.fn, head.scale, head.base, cfg.norm_eps, cfg.eps)),
        )
    )

    # 3) hc_pre projection identity: (X @ P) * rsqrt  ==  RMSNorm(X) @ P
    xp = torch.randn(b, s, n, d)
    xf = xp.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + cfg.norm_eps)
    results.append(
        ("hc_pre projection == RMSNorm(X)@P", maxdiff(F.linear(xf, wrap.fn) * rsqrt, F.linear(xf * rsqrt, wrap.fn)))
    )

    # 4) hc_pre structural: MHCWrap.hc_pre vs DeepSeek's hc_pre sequence (parametrize for the kernel)
    y_mine, post_mine, comb_mine = wrap.hc_pre(xp)
    mixes = ds_hc_pre_project(xp, wrap.fn, cfg.norm_eps)
    pre_ds, post_ds, comb_ds = parametrize(mixes, wrap.scale, wrap.base, cfg, "sinkhorn")
    y_ds = torch.sum(pre_ds.unsqueeze(-1) * xf.view(xp.shape), dim=2).to(xp.dtype)
    results.append(
        (
            "hc_pre (project+reduce structural)",
            max(maxdiff(y_mine, y_ds), maxdiff(post_mine, post_ds), maxdiff(comb_mine, comb_ds)),
        )
    )

    print(f"{'check':<42} {'max-abs-diff':>14}")
    print("-" * 58)
    worst = 0.0
    for name, md in results:
        worst = max(worst, md)
        print(f"{name:<42} {md:>14.2e}")
    print("-" * 58)
    ok = worst < 1e-6
    print(
        "RESULT:",
        "PASS  reference matches DeepSeek's pure-torch mHC computation"
        if ok
        else f"MISMATCH (worst {worst:.2e}) -- investigate",
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
