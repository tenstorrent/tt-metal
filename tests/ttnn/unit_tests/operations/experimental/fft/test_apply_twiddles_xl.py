# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for ttnn.experimental.apply_twiddles_xl — large-modulus between-pass
elementwise complex multiply (commit 5a).

This op replaces the table-based ttnn.experimental.apply_twiddles when
the twiddle modulus exceeds 1024.  The fft_three_pass composite uses it
for the between-pass-1-and-2 twiddle, where big_modulus = N1·N2 ranges
over [2^10, 2^20] for cube-balanced N up to 2^30.

Verifies:
  - twiddle row produced by the on-the-fly recurrence matches the closed-
    form numpy reference (fp32 and bf16) across the small/medium range
    that fits in test wall-time;
  - program cache hit on repeat;
  - single dispatch ⇒ Metal Trace replay works end-to-end.

Gated by TT_FFT_NATIVE=1.
"""

import os
import math
import pytest
import torch
import ttnn


pytestmark = pytest.mark.skipif(
    os.environ.get("TT_FFT_NATIVE", "1") == "0",
    reason="TT_FFT_NATIVE=1 not set; new ProgramDescriptor path is gated.",
)


# Same tolerance philosophy as apply_twiddles + fft_radix_pass: fp32 keeps
# tight, bf16 absorbs the dtype round-trip.  The recurrence accumulates
# ≤ P · ε_fp32 ≈ 1024 · 1e-7 = 1e-4 error so 5e-4 is the right fp32 bar.
_DTYPES = [
    (ttnn.float32,  torch.float32,  "fp32", 5e-4),
    (ttnn.bfloat16, torch.bfloat16, "bf16", 5e-2),
]


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).abs().norm() / ref.abs().norm().clamp_min(1e-30))


def _torch_xl_twiddle(
    y: torch.Tensor, P: int, big_modulus: int, full_N: int,
) -> torch.Tensor:
    """Reference: y[r, k] *= exp(-2πi · (r % big_modulus) · k / full_N).
    Uses double precision for the angle so the reference itself doesn't
    contribute to the error budget."""
    M = y.shape[-2]
    rows = torch.arange(M, dtype=torch.float64) % big_modulus
    cols = torch.arange(P, dtype=torch.float64)
    angle = -2.0 * math.pi * rows[:, None] * cols[None, :] / float(full_N)
    tw = torch.complex(
        angle.cos().to(torch.float32),
        angle.sin().to(torch.float32),
    ).to(torch.complex64)
    return y * tw


def _run_xl(
    device, xr: torch.Tensor, xi: torch.Tensor, tt_dtype,
    P: int, big_modulus: int, full_N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tt_xr = ttnn.from_torch(
        xr, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    tt_xi = ttnn.from_torch(
        xi, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    re, im = ttnn.experimental.apply_twiddles_xl(
        tt_xr, tt_xi, P=P, big_modulus=big_modulus, full_N=full_N,
    )
    return (
        ttnn.to_torch(re).to(torch.float32),
        ttnn.to_torch(im).to(torch.float32),
    )


# ─── 1. Correctness — small/medium big_modulus ─────────────────────────────
# Each row enumerates a distinct (n1·N2 + n2) so % big_modulus exercises
# the full modulus range.  We deliberately pick big_modulus values that
# exceed the apply_twiddles cap (1024) to demonstrate the new op's reason
# for existing.
_XL_CASES = [
    # (P=N3, big_modulus=N1·N2, full_N=N, M = big_modulus rows)
    #  → smallest, exactly at the apply_twiddles cap
    (32,   1024,  32 * 1024,         1024),
    #  → first case above the cap (2^11), N = 2^16 (64K)
    (32,   2048,  32 * 2048,         2048),
    #  → typical three-pass N = 2^22 (4M) with N1=N2=64, N3=64
    (64,   4096,  64 * 4096,         4096),
    #  → typical three-pass N = 2^24 (16M) with N1=N2=128, N3=1024
    (1024, 16384, 1024 * 16384,      16384),
]


@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("P,big_modulus,full_N,M", _XL_CASES,
                         ids=[f"P{p}_bm{bm}" for (p, bm, _, _) in _XL_CASES])
def test_apply_twiddles_xl_correctness(
    device, P, big_modulus, full_N, M, tt_dtype, torch_dtype, label, tol,
):
    torch.manual_seed(31)
    xr = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)
    xi = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)

    got_r, got_i = _run_xl(
        device, xr, xi, tt_dtype, P=P, big_modulus=big_modulus, full_N=full_N,
    )
    got = torch.complex(got_r.reshape(M, P), got_i.reshape(M, P))

    x_complex = torch.complex(
        xr.to(torch.float32), xi.to(torch.float32),
    ).to(torch.complex64)
    ref = _torch_xl_twiddle(x_complex, P=P, big_modulus=big_modulus, full_N=full_N)

    # Aggregate rel-err over the whole tensor — per-row is too expensive at
    # M=16384.  The recurrence error accumulates per-row independently so
    # the aggregate norm is a fair check.
    rel = _rel_err(got, ref)
    assert rel < tol, (
        f"[{label}] P={P} big_modulus={big_modulus} full_N={full_N} "
        f"M={M} aggregate rel err {rel:.2e}"
    )


# ─── 1b. Cap test (gated) — big_modulus at the 2^20 hard upper bound ──────
# The implementation caps big_modulus at 2^20 (= the apply_twiddles_xl
# cap, set by host delta-table size = 2^20 · 8 B = 8 MB).  This is the
# value the fft_three_pass composite uses at N = 2^30 cube-balanced.
# Default-skipped: torch reference's fp64 angle tensor is M·P·8 B which
# at (P=1024, M=2^20) is 8 GB.  Opt in with TT_FFT_AGGRESSIVE=1.
_AGGRESSIVE_GATE_XL = os.environ.get("TT_FFT_AGGRESSIVE", "0") == "1"


@pytest.mark.skipif(
    not _AGGRESSIVE_GATE_XL,
    reason="TT_FFT_AGGRESSIVE=1 not set; cap-modulus apply_twiddles_xl test is gated.",
)
def test_apply_twiddles_xl_at_cap(device):
    """big_modulus = 2^20 at the upper bound.  bf16 only — fp32 would
    need 8 GB for the input alone."""
    P, big_modulus = 1024, 1 << 20
    full_N = P * big_modulus   # 2^30 — exactly the 1G ceiling
    M = big_modulus            # one twiddle period

    torch.manual_seed(41)
    xr = torch.randn(M, P, dtype=torch.float32).to(torch.bfloat16)
    xi = torch.randn(M, P, dtype=torch.float32).to(torch.bfloat16)

    got_r, got_i = _run_xl(
        device, xr, xi, ttnn.bfloat16,
        P=P, big_modulus=big_modulus, full_N=full_N,
    )
    got = torch.complex(got_r.reshape(M, P), got_i.reshape(M, P))

    x_complex = torch.complex(
        xr.to(torch.float32), xi.to(torch.float32),
    ).to(torch.complex64)
    ref = _torch_xl_twiddle(x_complex, P=P, big_modulus=big_modulus, full_N=full_N)

    rel = _rel_err(got, ref)
    # 1e-1 because bf16 itself contributes ~2e-2 round-trip noise per
    # element and the recurrence accumulates over P=1024 steps.
    assert rel < 1e-1, (
        f"[bf16-cap] big_modulus={big_modulus} P={P} M={M} "
        f"rel err {rel:.2e}"
    )


# ─── 2. Multi-batch wraparound — verify (r % big_modulus) wraps cleanly ───
# At M = 2·big_modulus, rows [0, big_modulus) get one twiddle period,
# rows [big_modulus, 2·big_modulus) get the SAME period again (broadcast
# semantics).  Confirms the kernel's row_phase_idx wraps correctly.
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
def test_apply_twiddles_xl_wraparound(
    device, tt_dtype, torch_dtype, label, tol,
):
    P, big_modulus, full_N = 32, 1024, 32 * 1024
    M = 2 * big_modulus  # two full periods
    torch.manual_seed(37)
    xr = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)
    xi = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)

    got_r, got_i = _run_xl(
        device, xr, xi, tt_dtype, P=P, big_modulus=big_modulus, full_N=full_N,
    )
    got = torch.complex(got_r.reshape(M, P), got_i.reshape(M, P))

    x_complex = torch.complex(
        xr.to(torch.float32), xi.to(torch.float32),
    ).to(torch.complex64)
    ref = _torch_xl_twiddle(x_complex, P=P, big_modulus=big_modulus, full_N=full_N)

    rel = _rel_err(got, ref)
    assert rel < tol, f"[{label}] wraparound rel err {rel:.2e}"


# ─── 3. Program cache hit ──────────────────────────────────────────────────
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
def test_apply_twiddles_xl_program_cache_hit(
    device, tt_dtype, torch_dtype, label, tol,
):
    P, big_modulus, full_N = 32, 2048, 32 * 2048
    M = big_modulus
    torch.manual_seed(0)
    xr = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)
    xi = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)

    _run_xl(device, xr, xi, tt_dtype, P=P, big_modulus=big_modulus, full_N=full_N)
    n_after_warmup = device.num_program_cache_entries()

    _run_xl(device, xr, xi, tt_dtype, P=P, big_modulus=big_modulus, full_N=full_N)
    n_after_repeat = device.num_program_cache_entries()

    assert n_after_repeat == n_after_warmup, (
        f"[{label}] apply_twiddles_xl program cache regression: "
        f"{n_after_warmup} → {n_after_repeat}"
    )


# ─── 4. Metal Trace replay — single dispatch, must be trace-safe ──────────
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
def test_apply_twiddles_xl_metal_trace_replay(
    device, tt_dtype, torch_dtype, label, tol,
):
    P, big_modulus, full_N = 32, 1024, 32 * 1024
    M = big_modulus
    torch.manual_seed(2)
    xr = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)
    xi = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)

    tt_xr_e = ttnn.from_torch(xr, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_xi_e = ttnn.from_torch(xi, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    re_e, im_e = ttnn.experimental.apply_twiddles_xl(
        tt_xr_e, tt_xi_e, P=P, big_modulus=big_modulus, full_N=full_N,
    )
    eager_r = ttnn.to_torch(re_e).to(torch.float32).clone()
    eager_i = ttnn.to_torch(im_e).to(torch.float32).clone()

    # Warm caches.
    tt_xr_w = ttnn.from_torch(xr, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_xi_w = ttnn.from_torch(xi, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn.experimental.apply_twiddles_xl(
        tt_xr_w, tt_xi_w, P=P, big_modulus=big_modulus, full_N=full_N,
    )

    tt_xr_t = ttnn.from_torch(xr, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_xi_t = ttnn.from_torch(xi, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    re_t, im_t = ttnn.experimental.apply_twiddles_xl(
        tt_xr_t, tt_xi_t, P=P, big_modulus=big_modulus, full_N=full_N,
    )
    ttnn.end_trace_capture(device, tid, cq_id=0)

    try:
        for i in range(10):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            replay_r = ttnn.to_torch(re_t).to(torch.float32)
            replay_i = ttnn.to_torch(im_t).to(torch.float32)
            assert torch.allclose(replay_r, eager_r, rtol=tol, atol=tol), (
                f"[{label}] trace replay {i} real mismatch: max abs diff "
                f"{(replay_r - eager_r).abs().max().item():.2e}"
            )
            assert torch.allclose(replay_i, eager_i, rtol=tol, atol=tol), (
                f"[{label}] trace replay {i} imag mismatch: max abs diff "
                f"{(replay_i - eager_i).abs().max().item():.2e}"
            )
    finally:
        ttnn.release_trace(device, tid)
