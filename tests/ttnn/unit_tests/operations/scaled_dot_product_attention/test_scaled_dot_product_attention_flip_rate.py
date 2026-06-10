# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 5 — uniform/negative-input bf16-floor flip-rate tests.

The R5 regressions were single-ulp flips at the bf16 output grid (ulp_p99=1,
median_abs=0) on near-uniform-attention inputs. Root causes (probes 014–032):
  1. Dominant: the descriptor never set UnpackToDestMode::UnpackToDestFp32, so
     every copy_tile/UnaryBcast of a Float32 CB silently truncated to fp16 —
     the O accumulator (cb_o_acc) lost mantissa each KV block.
  2. Secondary: bare SFPU recip carries ~3.6e-5 relative error; Phase 11 now
     adds one Newton step (inv <- inv * (2 - l*inv)).

These tests pin both at the most sensitive operating point: Q=K=0 makes the
attention exactly uniform (P=1, l=S), so O is exactly mean(V) and any flip is
kernel error, not input-conditioning. Flip-rate gates mirror the golden
regression budget (rms <= 0.04 at sigma_ref ~ 1 ulp => flips < ~0.2%).
"""

import pytest
import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _run_zero_qk(device, V):
    shape = list(V.shape)
    Z = torch.zeros(*shape, dtype=torch.bfloat16)
    tq = ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(Z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv)).float()


@pytest.mark.parametrize("S", [128, 256, 512])
def test_uniform_attention_exact_mean_fp32_grid(device, S):
    """V on the 2^-9 bf16 grid: every row-sum is fp32-exact, so device error
    vs the RNE-rounded exact mean must be confined to exact bf16 ties (the
    packer rounds half-up while torch RNE rounds half-to-even; on the 2^-17
    mean lattice ties have measure ~2^-8). Pre-fix the fp16 O accumulator
    flipped 9–19% of NON-tie outputs (probe_031)."""
    D = 64
    shape = (1, 1, S, D)
    torch.manual_seed(0)
    V = (0.5 + torch.randint(0, 256, shape) * (2**-9)).to(torch.bfloat16)
    out = _run_zero_qk(device, V)
    exact = V.double().mean(dim=-2, keepdim=True).expand(shape)
    rne = exact.float().to(torch.bfloat16).float()
    flipped = out != rne
    # ties: exact mean precisely halfway between adjacent bf16 grid points
    lo = rne - 0.00390625
    hi = rne + 0.00390625
    is_tie = ((exact - lo.double()).abs() == 0.001953125) | ((hi.double() - exact).abs() == 0.001953125)
    nontie_flips = (flipped & ~is_tie).float().mean().item()
    assert nontie_flips == 0.0, f"S={S}: {nontie_flips*100:.2f}% non-tie flips vs exact RNE mean"


@pytest.mark.parametrize("S", [64, 512])
def test_recip_newton_exact_power_of_two(device, S):
    """O = mean(V) with V constant per column: result is exactly V (l = S
    is a power of two, recip exact after the Newton step). Pre-fix the
    ~3.6e-5 recip error pushed on-grid values off by one ulp."""
    D = 64
    shape = (1, 1, S, D)
    c = 0.50390625  # bf16-representable
    V = torch.full(shape, c, dtype=torch.bfloat16)
    out = _run_zero_qk(device, V)
    assert (out == c).all(), f"max diff {(out - c).abs().max():.3e}"


_DIST_SHAPES = [(1, 1, 32, 32), (1, 4, 128, 64), (1, 8, 256, 64), (2, 4, 128, 64), (1, 12, 512, 64)]
_IDS = [f"B{s[0]}_H{s[1]}_S{s[2]}_D{s[3]}" for s in _DIST_SHAPES]


def _flip_stats(shape, make):
    torch.manual_seed(42)
    Q, K, V = make(shape), make(shape), make(shape)
    import math

    s = torch.matmul(Q.float(), K.float().transpose(-2, -1)) / math.sqrt(shape[-1])
    ref = torch.matmul(torch.softmax(s, -1), V.float())
    return Q, K, V, ref.to(torch.bfloat16).float()


@pytest.mark.parametrize("shape", _DIST_SHAPES, ids=_IDS)
@pytest.mark.parametrize(
    "dist,make",
    [
        ("uniform", lambda s: torch.rand(s, dtype=torch.bfloat16)),
        ("negative", lambda s: -(torch.rand(s, dtype=torch.bfloat16) + 0.5)),
    ],
    ids=["uniform", "negative"],
)
def test_flip_rate_budget(device, shape, dist, make):
    """Uniform/negative inputs: single-ulp flip rate vs fp32 reference must
    stay within the golden rms=0.04 budget. Measured post-fix: 0.2–1.0%
    (golden suite green); pre-fix: 2–6%. Gate at 1.5% to catch regressions
    of either root cause without tripping on RNE-vs-half-up tie noise."""
    Q, K, V, ref = _flip_stats(shape, make)
    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv)).float()
    flips = (out != ref).float().mean().item()
    assert flips <= 0.015, f"{dist} {shape}: flip rate {flips*100:.2f}% > 1.5%"
