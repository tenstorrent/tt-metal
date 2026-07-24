#!/usr/bin/env python3
"""Isolated cross-check: DeepSeek-V4's TileLang Sinkhorn kernel vs our torch reference.

The Sinkhorn parametrization is the ONLY non-torch piece of the mHC path. This script
runs DeepSeek's actual kernel (vendored verbatim in deepseek_sinkhorn_kernel.py) on the
GPU and compares its (pre, post, comb) outputs against models/.../mhc_reference.py's
`parametrize`, on identical random inputs. No model, no projection, no F -- just
    mixes [B,S,(2+n)n], scale [3], base [(2+n)n]  ->  pre, post, comb.

Requires: CUDA GPU + tilelang.  Run:
    python models/demos/deepseek_v3_d_p/reference/mhc/check_sinkhorn_vs_kernel.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepseek_sinkhorn_kernel import hc_split_sinkhorn  # DUT (TileLang)  # noqa: E402
from mhc_reference import MHCConfig, parametrize  # torch reference  # noqa: E402


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two tensors, flattened (tt-metal's PCC metric)."""
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-30)
    return (a @ b / denom).item()


def run_case(N: int, scale_val: float, seed: int, n: int = 4, iters: int = 20, eps: float = 1e-6):
    """One (mixes, scale, base) sample through both implementations."""
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    mix_hc = (2 + n) * n
    mixes = torch.randn(1, N, mix_hc, device=dev, dtype=torch.float32, generator=g)
    scale = torch.full((3,), scale_val, device=dev, dtype=torch.float32)
    base = torch.randn(mix_hc, device=dev, dtype=torch.float32, generator=g)

    # --- device under test: DeepSeek's TileLang kernel ---
    k_pre, k_post, k_comb = hc_split_sinkhorn(mixes, scale, base, n, iters, eps)

    # --- torch reference: identical inputs ---
    cfg = MHCConfig(n=n, sinkhorn_iters=iters, eps=eps)
    r_pre, r_post, r_comb = parametrize(mixes, scale, base, cfg, constraint="sinkhorn")

    diffs = {
        name: ((k - r).abs().max().item(), pcc(k, r))
        for name, k, r in [("pre", k_pre, r_pre), ("post", k_post, r_post), ("comb", k_comb, r_comb)]
    }
    # independent sanity: is the KERNEL's comb actually doubly-stochastic?
    diffs["ds_row"] = (k_comb.sum(-1) - 1).abs().max().item()
    diffs["ds_col"] = (k_comb.sum(-2) - 1).abs().max().item()
    return diffs


def main():
    assert torch.cuda.is_available(), "this check needs a CUDA GPU"
    print(
        f"torch {torch.__version__} | tilelang {getattr(__import__('tilelang'), '__version__', '?')} "
        f"| {torch.cuda.get_device_name()}"
    )
    print(
        f"{'N':>5} {'scale':>6} {'seed':>4} | {'comb maxdiff':>12} {'comb PCC':>11} | "
        f"{'pre maxdiff':>11} {'post maxdiff':>12} | {'kern ds_row':>11} {'kern ds_col':>11}"
    )
    print("-" * 100)

    worst_pcc, worst_diff = 1.0, 0.0
    for scale_val in (0.01, 0.5, 1.0, 3.0):  # 0.01 ~ init; larger stresses the Sinkhorn
        for N in (1, 7, 128, 1000):
            for seed in (0, 1):
                d = run_case(N, scale_val, seed)
                worst_pcc = min(worst_pcc, d["comb"][1], d["pre"][1], d["post"][1])
                worst_diff = max(worst_diff, d["comb"][0], d["pre"][0], d["post"][0])
                print(
                    f"{N:>5} {scale_val:>6} {seed:>4} | {d['comb'][0]:>12.2e} {d['comb'][1]:>11.8f} | "
                    f"{d['pre'][0]:>11.2e} {d['post'][0]:>12.2e} | {d['ds_row']:>11.1e} {d['ds_col']:>11.1e}"
                )

    print("-" * 100)
    print(f"worst PCC: {worst_pcc:.8f}   worst max-abs-diff: {worst_diff:.2e}")
    ok = worst_pcc > 0.9999 and worst_diff < 1e-3
    print(
        "RESULT:",
        "PASS  torch reference matches DeepSeek's TileLang Sinkhorn"
        if ok
        else "MISMATCH -- investigate (see diffs above)",
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
