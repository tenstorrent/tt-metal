# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
# test_mhc_post_debug.py — deterministic debugging of the R2 projection precision. DO NOT DELETE.

import pytest
import torch
import ttnn

from eval.golden_tests.mhc_post.feature_spec import _case
from eval.golden_tests.mhc_post.helpers import call_mhc_post, make_case
from ttnn.operations.mhc_post import mhc_post

BF16 = dict(dtype=ttnn.bfloat16, sublayer_dtype=ttnn.bfloat16)


def _trunc(t, bits):
    i = t.float().contiguous().view(torch.int32)
    sh = 23 - bits
    return ((i >> sh) << sh).view(torch.float32)


@pytest.mark.parametrize("fid", [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2])
def test_projection_operand_bits(device, fid):
    """Which W / X' mantissa width does the device projection see? Compare the op's stats[:, :24]
    against X'_op @ W with W truncated to k bits."""
    case = make_case(_case((1, 1, 64, 4 * 640)), mode="stats", coef_source="packed", seed=61, **BF16)
    x_out, st = call_mhc_post(
        case, mode="stats", coef_source="packed", device=device, math_fidelity=fid, op=mhc_post, **BF16
    )
    xo = ttnn.to_torch(x_out).double().reshape(-1, 4 * 640)
    got = ttnn.to_torch(st).double().reshape(-1, 32)[:, :24]
    w = case.w.reshape(4 * 640, 32)
    for bits in (23, 10, 9, 8, 7, 6):
        ref = (xo @ _trunc(w, bits).double())[:, :24]
        r = ((got - ref).norm() / ref.norm()).item()
        print(f"fid={fid} W bits {bits}: rel {r:.3g}")
    for bits in (7, 6, 5, 4, 3):
        ref = (_trunc(xo.float(), bits).double() @ w.double())[:, :24]
        r = ((got - ref).norm() / ref.norm()).item()
        print(f"fid={fid} X bits {bits}: rel {r:.3g}")


def _round(t, bits):
    t = t.float().contiguous()
    i = t.view(torch.int32)
    sh = 23 - bits
    half = 1 << (sh - 1)
    lsb = (i >> sh) & 1
    return (((i + half - 1 + lsb) >> sh) << sh).view(torch.float32)


def test_projection_accumulation_model(device):
    """Where does the HiFi4 projection's ~3e-4 floor come from: rounding of per-tile partial sums, or of the
    running DEST accumulator?"""
    fid = ttnn.MathFidelity.HiFi4
    case = make_case(_case((1, 1, 64, 4 * 640)), mode="stats", coef_source="packed", seed=61, **BF16)
    x_out, st = call_mhc_post(
        case, mode="stats", coef_source="packed", device=device, math_fidelity=fid, op=mhc_post, **BF16
    )
    xo = ttnn.to_torch(x_out).double().reshape(-1, 4 * 640)
    got = ttnn.to_torch(st).double().reshape(-1, 32)[:, :24]
    w = case.w.reshape(4 * 640, 32).double()
    exact = (xo @ w)[:, :24]
    print(f"exact: rel {((got - exact).norm() / exact.norm()).item():.3g}")
    K = xo.shape[1] // 32
    for bits in (10, 12, 14, 16, 18):
        parts = sum(
            _round((xo[:, k * 32 : (k + 1) * 32] @ w[k * 32 : (k + 1) * 32]).float(), bits).double() for k in range(K)
        )[:, :24]
        print(f"tile partials rounded to {bits}: rel {((got - parts).norm() / exact.norm()).item():.3g}")
    for bits in (10, 12, 14, 16, 18):
        prods = _round((xo.unsqueeze(-1) * w.unsqueeze(0)).float(), bits).double().sum(1)[:, :24]
        print(f"products rounded to {bits}: rel {((got - prods).norm() / exact.norm()).item():.3g}")


def test_projection_error_structure(device):
    """Regress the HiFi4 projection residual on candidate dropped-bit terms."""
    fid = ttnn.MathFidelity.HiFi4
    case = make_case(_case((1, 1, 64, 4 * 640)), mode="stats", coef_source="packed", seed=61, **BF16)
    x_out, st = call_mhc_post(
        case, mode="stats", coef_source="packed", device=device, math_fidelity=fid, op=mhc_post, **BF16
    )
    xo = ttnn.to_torch(x_out).double().reshape(-1, 4 * 640)
    got = ttnn.to_torch(st).double().reshape(-1, 32)[:, :24]
    w = case.w.reshape(4 * 640, 32).double()
    exact = (xo @ w)[:, :24]
    res = got - exact
    print("res per col rel:", [round(((res[:, c]).norm() / exact[:, c].norm()).item(), 6) for c in range(0, 24, 3)])
    print(
        "res per row-tile rel:",
        [
            round((res[r * 32 : (r + 1) * 32]).norm().item() / exact[r * 32 : (r + 1) * 32].norm().item(), 6)
            for r in range(2)
        ],
    )
    cands = {
        "W_lowbit": xo @ (w - _trunc(w.float(), 8).double()),
        "W_low2": xo @ (w - _trunc(w.float(), 7).double()),
        "X_lowbit": (xo - _trunc(xo.float(), 6).double()) @ w,
        "X_low2": (xo - _trunc(xo.float(), 5).double()) @ w,
        "X_low3": (xo - _trunc(xo.float(), 4).double()) @ w,
    }
    for name, c in cands.items():
        c = c[:, :24]
        a = (res * c).sum() / (c * c).sum()
        left = (res - a * c).norm() / exact.norm()
        print(f"{name}: coef {a.item():.4f}, residual after fit {left.item():.3g}")


def test_projection_operand_is_written_x(device):
    """Is the projection's X' the written X' (RNE bf16) or a truncation of the fp32 mix?"""
    from eval.golden_tests.mhc_post.helpers import pytorch_mhc_post

    fid = ttnn.MathFidelity.HiFi4
    case = make_case(_case((1, 1, 64, 4 * 640)), mode="stats", coef_source="packed", seed=61, **BF16)
    x_out, st = call_mhc_post(
        case, mode="stats", coef_source="packed", device=device, math_fidelity=fid, op=mhc_post, **BF16
    )
    xo = ttnn.to_torch(x_out).double().reshape(-1, 4 * 640)
    mix = pytorch_mhc_post(case.f.double(), case.x.double(), case.post.double(), case.comb.double()).float()
    mix = mix.reshape(-1, 4 * 640)
    got = ttnn.to_torch(st).double().reshape(-1, 32)[:, :24]
    w = case.w.reshape(4 * 640, 32).double()
    for name, xv in (
        ("written", xo),
        ("rne(mix)", _round(mix, 7).double()),
        ("trunc(mix)", _trunc(mix, 7).double()),
        ("fp32 mix", mix.double()),
    ):
        ref = (xv @ w)[:, :24]
        print(f"{name}: rel {((got - ref).norm() / ref.norm()).item():.3g}")
    print("written==rne(mix) frac:", (xo == _round(mix, 7).double()).double().mean().item())
