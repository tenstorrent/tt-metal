# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for bluestein_xl (commit 6e-2 — extended-range Bluestein with
host chirp glue + device fft_three_pass).

Coverage matrix:
  * primes whose M = next_pow2(2N-1) > 2^20, i.e. N > 524 287
  * fp32 only (host glue uses torch fp64 internally for chirp accuracy)
  * B = 1 (multi-batch deferred)

Default sweep stays under the validated three-pass envelope (M ≤ 2^22).
Aggressive cases (M up to 2^24 → N up to ~8M) are gated behind
TT_FFT_AGGRESSIVE=1 — they take minutes per call.
"""

import os
import sys

# Make the sibling helper `bluestein_xl.py` importable regardless of how
# pytest is invoked (rootdir, importlib mode, etc.). The helper lives in
# the same directory as this test file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import torch
import ttnn

from bluestein_xl import bluestein_fft_xl, _next_pow2, _bluestein_plan


pytestmark = pytest.mark.skipif(
    os.environ.get("TT_FFT_NATIVE", "1") == "0",
    reason="TT_FFT_NATIVE=1 not set; new ProgramDescriptor path is gated.",
)


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).abs().norm() / ref.abs().norm().clamp_min(1e-30))


# ─── 1. Correctness on primes that exceed the device-only cap ──────────────
#
# Each N here yields M = next_pow2(2N - 1) > 2^20, so the existing
# ttnn.experimental.bluestein_fft would TT_FATAL.  bluestein_fft_xl is
# the path that's expected to handle them.
@pytest.mark.parametrize(
    "N",
    [
        524_309,    # smallest prime with M = 2^21
        700_001,    # M = 2^21
        1_048_583,  # smallest prime > 2^20; M = 2^21
        1_500_007,  # M = 2^22
        2_000_003,  # M = 2^22
    ],
    ids=lambda v: f"N{v}",
)
def test_bluestein_xl_real_correctness(device, N):
    torch.manual_seed(N & 0x7FFF_FFFF)
    x = torch.randn(1, N, dtype=torch.float32)

    got_re, got_im = bluestein_fft_xl(device, x, N=N)
    got = torch.complex(got_re[0], got_im[0]).to(torch.complex128)

    ref = torch.fft.fft(x[0].to(torch.complex128), dim=-1)
    rel = _rel_err(got, ref)

    # Two FFT_M dispatches + chirp pre/post + B-mul + 1/M scale. The
    # dominant fp32 rounding source is the three-pass kernel itself; tol
    # mirrors the fp32 three_pass test budget plus a small Bluestein
    # margin.
    assert rel < 1e-2, (
        f"bluestein_xl[N={N}] rel err {rel:.2e} exceeds 1e-2 tol\n"
        f"  M = {_next_pow2(2 * N - 1)}"
    )


# ─── 2. Complex-input variant ──────────────────────────────────────────────
@pytest.mark.parametrize(
    "N",
    [524_309, 1_048_583, 1_500_007],
    ids=lambda v: f"N{v}",
)
def test_bluestein_xl_complex_correctness(device, N):
    torch.manual_seed((N * 7) & 0x7FFF_FFFF)
    x_re = torch.randn(1, N, dtype=torch.float32)
    x_im = torch.randn(1, N, dtype=torch.float32)

    got_re, got_im = bluestein_fft_xl(device, x_re, x_im, N=N)
    got = torch.complex(got_re[0], got_im[0]).to(torch.complex128)

    ref = torch.fft.fft(
        torch.complex(x_re[0], x_im[0]).to(torch.complex128), dim=-1
    )
    rel = _rel_err(got, ref)
    assert rel < 1e-2, (
        f"bluestein_xl complex[N={N}] rel err {rel:.2e} exceeds 1e-2 tol\n"
        f"  M = {_next_pow2(2 * N - 1)}"
    )


# ─── 3. Plan caching — second call for the same N must hit the cache ──────
def test_bluestein_xl_plan_is_cached():
    N = 524_309
    w_1, B_1, M_1 = _bluestein_plan(N)
    w_2, B_2, M_2 = _bluestein_plan(N)
    assert w_1 is w_2, "chirp w should be lru_cache shared across calls"
    assert B_1 is B_2, "B_fft should be lru_cache shared across calls"
    assert M_1 == M_2


# ─── 4. AGGRESSIVE — primes well past two_pass cap (minutes per call) ─────
_AGGRESSIVE = os.environ.get("TT_FFT_AGGRESSIVE", "0") == "1"

aggressive = pytest.mark.skipif(
    not _AGGRESSIVE,
    reason="TT_FFT_AGGRESSIVE=1 not set; slow large-N XL tests are gated.",
)


@aggressive
@pytest.mark.parametrize(
    "N",
    [
        4_194_301,   # prime, M = 2^23
        8_388_593,   # prime, M = 2^24
    ],
    ids=lambda v: f"N{v}",
)
def test_bluestein_xl_aggressive_correctness(device, N):
    torch.manual_seed(N & 0x7FFF_FFFF)
    x = torch.randn(1, N, dtype=torch.float32)

    got_re, got_im = bluestein_fft_xl(device, x, N=N)
    got = torch.complex(got_re[0], got_im[0]).to(torch.complex128)

    ref = torch.fft.fft(x[0].to(torch.complex128), dim=-1)
    rel = _rel_err(got, ref)
    # Looser tol — three rounding cycles compound across larger N.
    assert rel < 5e-2, (
        f"bluestein_xl AGGRESSIVE[N={N}] rel err {rel:.2e}\n"
        f"  M = {_next_pow2(2 * N - 1)}"
    )
