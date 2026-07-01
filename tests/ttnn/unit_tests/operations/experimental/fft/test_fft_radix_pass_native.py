# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for ttnn.experimental.fft_radix_pass — the fused [batched
length-P FFT + optional post-twiddle complex multiply] device op
(commit 4 of the host-to-device refactor).

This op is a SINGLE dispatch:
  - `twiddle_N2 == 0` → pure batched length-P FFT
    (equivalent observable behaviour to ttnn.experimental.fft on (M,P))
  - `twiddle_N2  > 0` → batched FFT followed by an in-place scalar
    cmul against twiddle row (r % twiddle_N2) of T[n2,k] =
    exp(-2πi · n2 · k / (P · twiddle_N2)).
    Equivalent observable behaviour to:
        re, im = ttnn.experimental.fft(input)
        re, im = ttnn.experimental.apply_twiddles(re, im,
                                                   N1=P, N2=twiddle_N2)
    but in one dispatch with no intermediate L1↔DRAM round-trip.

All tests gated by TT_FFT_NATIVE=1 like the rest of the new path.
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


# (ttnn dtype, torch dtype, dtype label, rel-err tolerance for fused op)
# Tighter than the two-pass composite because there's no transpose
# allocation chain in between — pure single-dispatch path.
_DTYPES = [
    (ttnn.float32,  torch.float32,  "fp32", 5e-4),
    (ttnn.bfloat16, torch.bfloat16, "bf16", 5e-2),
]


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).abs().norm() / ref.abs().norm().clamp_min(1e-30))


def _torch_post_twiddle(
    y: torch.Tensor, P: int, N2: int, stride: int = 1,
) -> torch.Tensor:
    """Apply T[(r/stride)%N2, k] = exp(-2πi · row_idx · k / (P*N2))
    elementwise to a complex tensor `y` of shape (M, P)."""
    M = y.shape[-2]
    rows = (torch.arange(M, dtype=torch.float64) // stride) % N2
    cols = torch.arange(P, dtype=torch.float64)
    angle = -2.0 * math.pi * rows[:, None] * cols[None, :] / float(P * N2)
    tw = torch.complex(
        angle.cos().to(torch.float32),
        angle.sin().to(torch.float32),
    ).to(torch.complex64)
    return y * tw


def _run_radix_pass(
    device, x_real: torch.Tensor, x_imag: torch.Tensor | None,
    tt_dtype, P: int, twiddle_N2: int, stride: int = 1,
    output_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    tt_xr = ttnn.from_torch(
        x_real, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    if x_imag is None:
        re, im = ttnn.experimental.fft_radix_pass(
            tt_xr, P=P, twiddle_N2=twiddle_N2, stride=stride,
            output_scale=output_scale,
        )
    else:
        tt_xi = ttnn.from_torch(
            x_imag, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
        )
        re, im = ttnn.experimental.fft_radix_pass(
            tt_xr, tt_xi, P=P, twiddle_N2=twiddle_N2, stride=stride,
            output_scale=output_scale,
        )
    return (
        ttnn.to_torch(re).to(torch.float32),
        ttnn.to_torch(im).to(torch.float32),
    )


# ─── 1. Pure FFT mode (twiddle_N2=0) — must match batched FFT bit-for-bit ──
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("P", [32, 64, 128, 256, 512, 1024])
def test_radix_pass_pure_fft(device, M, P, tt_dtype, torch_dtype, label, tol):
    """twiddle_N2=0 → fft_radix_pass is just a batched length-P FFT.
    Should match torch.fft.fft on the same input within dtype tolerance."""
    torch.manual_seed(7)
    x_fp32 = torch.randn(M, P, dtype=torch.float32)
    x = x_fp32.to(torch_dtype)

    got_r, got_i = _run_radix_pass(
        device, x, None, tt_dtype, P=P, twiddle_N2=0,
    )
    got = torch.complex(got_r.reshape(M, P), got_i.reshape(M, P))

    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)

    for r in range(M):
        rel = _rel_err(got[r], ref[r])
        assert rel < tol, (
            f"[{label}] pure-FFT M={M} P={P} row={r} rel err {rel:.2e}"
        )


# ─── 2. Fused mode (twiddle_N2>0) — must match FFT + apply_twiddles ────────
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("N2", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("P", [32, 64, 128, 256])
def test_radix_pass_fused(device, N2, P, tt_dtype, torch_dtype, label, tol):
    """twiddle_N2>0 → fft_radix_pass = batched FFT + post-twiddle cmul.
    Cross-check against torch reference (FFT, then twiddle multiply)."""
    M = N2 * 2  # at least 2 rows per twiddle modulus so we exercise % logic
    torch.manual_seed(13)
    x_fp32 = torch.randn(M, P, dtype=torch.float32)
    x = x_fp32.to(torch_dtype)

    got_r, got_i = _run_radix_pass(
        device, x, None, tt_dtype, P=P, twiddle_N2=N2,
    )
    got = torch.complex(got_r.reshape(M, P), got_i.reshape(M, P))

    ref_fft = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    ref = _torch_post_twiddle(ref_fft, P=P, N2=N2)

    for r in range(M):
        rel = _rel_err(got[r], ref[r])
        assert rel < tol, (
            f"[{label}] fused M={M} P={P} N2={N2} row={r} rel err {rel:.2e}"
        )


# ─── 3. Complex input (real + imag) — Pass-2 of the two-pass composite ────
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("M", [2, 4, 8])
@pytest.mark.parametrize("P", [32, 64, 128])
def test_radix_pass_complex_input(
    device, M, P, tt_dtype, torch_dtype, label, tol,
):
    """fft_radix_pass with both real and imag inputs (twiddle_N2=0).
    Models the Pass-2 step of the two-pass composite."""
    torch.manual_seed(21)
    xr = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)
    xi = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)

    got_r, got_i = _run_radix_pass(
        device, xr, xi, tt_dtype, P=P, twiddle_N2=0,
    )
    got = torch.complex(got_r.reshape(M, P), got_i.reshape(M, P))

    x_complex = torch.complex(
        xr.to(torch.float32), xi.to(torch.float32),
    ).to(torch.complex64)
    ref = torch.fft.fft(x_complex, dim=-1)

    for r in range(M):
        rel = _rel_err(got[r], ref[r])
        assert rel < tol, (
            f"[{label}] complex M={M} P={P} row={r} rel err {rel:.2e}"
        )


# ─── 3b. Pass-1 surrogate — exact (P, twiddle_N2, M) tuples used by ──────
#         fft_two_pass.  Catches twiddle_N2 ≥ 64 regressions that the
#         baseline N2∈{1..32} sweep above doesn't reach.  N=2048..16K
#         covers all balanced factorizations in the two-pass gated range.
_TWOPASS_PASS1 = [
    # (B, N, N1, N2) — N1 = twiddle_N2, N2 = P
    (1,  2048,  64,  32),
    (1,  4096,  64,  64),
    (1,  8192, 128,  64),
    (1, 16384, 128, 128),
    (2,  2048,  64,  32),
    (4,  2048,  64,  32),
]


@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("B,N,N1,N2", _TWOPASS_PASS1,
                         ids=[f"B{b}N{n}" for (b, n, _, _) in _TWOPASS_PASS1])
def test_radix_pass_fused_pass1_surrogate(
    device, B, N, N1, N2, tt_dtype, torch_dtype, label, tol,
):
    """fft_radix_pass called with the exact (P=N2, twiddle_N2=N1, M=B*N1)
    tuples that fft_two_pass produces for Pass-1.  Reference is the
    explicit (length-N2 FFT) ∘ (post-twiddle) pipeline."""
    M = B * N1
    torch.manual_seed(11)
    x = torch.randn(M, N2, dtype=torch.float32).to(torch_dtype)

    got_r, got_i = _run_radix_pass(
        device, x, None, tt_dtype, P=N2, twiddle_N2=N1,
    )
    got = torch.complex(got_r.reshape(M, N2), got_i.reshape(M, N2))

    ref_fft = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    ref = _torch_post_twiddle(ref_fft, P=N2, N2=N1)

    for r in range(M):
        rel = _rel_err(got[r], ref[r])
        assert rel < tol, (
            f"[{label}] pass1-surrogate B={B} N={N} (N1={N1},N2={N2}) "
            f"row={r} rel err {rel:.2e}"
        )


# ─── 3c. Strided post-twiddle — Pass-2 of fft_three_pass (commit 5) ──────
# Three-pass enumerates rows as (b, n1, k3) at row index
#     r = b·N1·N3 + n1·N3 + k3
# i.e. n1 sits at row-stride N3.  fft_radix_pass with twiddle_N2=N1 and
# stride=N3 picks twiddle row (r / N3) % N1 = n1 — exactly the n1 factor
# of the Cooley–Tukey twiddle 2.  This test exercises that codepath
# standalone so a three-pass regression points back here.
_THREEPASS_PASS2 = [
    # (P=N2, twiddle_N2=N1, stride=N3, M = N1 · N3) — modest sizes that
    # cover one-batch-per-core, multi-batch-per-core, and a stride change.
    (32, 32,  32,  32 * 32),
    (32, 64,  32,  64 * 32),
    (64, 64,  64,  64 * 64),
    (32, 32, 128,  32 * 128),
]


@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("P,twiddle_N2,stride,M", _THREEPASS_PASS2,
                         ids=[f"P{p}_N1{n}_S{s}" for (p, n, s, _) in _THREEPASS_PASS2])
def test_radix_pass_strided_twiddle(
    device, P, twiddle_N2, stride, M, tt_dtype, torch_dtype, label, tol,
):
    """fft_radix_pass(P, twiddle_N2, stride): post-twiddle picks row
    (r/stride) % twiddle_N2.  Cross-check against torch FFT + manual
    strided twiddle multiply on a complex (Pass-2-like) input."""
    torch.manual_seed(23)
    xr = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)
    xi = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)

    got_r, got_i = _run_radix_pass(
        device, xr, xi, tt_dtype, P=P, twiddle_N2=twiddle_N2, stride=stride,
    )
    got = torch.complex(got_r.reshape(M, P), got_i.reshape(M, P))

    x_complex = torch.complex(
        xr.to(torch.float32), xi.to(torch.float32),
    ).to(torch.complex64)
    ref_fft = torch.fft.fft(x_complex, dim=-1)
    ref = _torch_post_twiddle(ref_fft, P=P, N2=twiddle_N2, stride=stride)

    for r in range(M):
        rel = _rel_err(got[r], ref[r])
        assert rel < tol, (
            f"[{label}] strided P={P} N1={twiddle_N2} stride={stride} "
            f"M={M} row={r} rel err {rel:.2e}"
        )


# ─── 3b. Output scale (commit 6c, for IFFT) ────────────────────────────────
# fft_radix_pass gained an `output_scale` param: every output element is
# multiplied by this scalar AFTER any post-twiddle (and before any bf16
# truncation).  Used by the IFFT composite to fold the 1/N scale into the
# LAST radix_pass writer with zero extra dispatch.
#
# We cross-check by running the SAME input twice — once with scale=1.0
# and once with scale=alpha — and verifying the scale-alpha output is
# exactly alpha × the scale-1.0 output to within fp32 noise.  We also
# pair scale with twiddle_N2>0 to make sure the scale multiplies AFTER
# the post-twiddle (commute-with-cmul property).
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("twiddle_N2", [0, 4])
@pytest.mark.parametrize("alpha", [0.5, 1.0 / 128.0, -2.0])
def test_radix_pass_output_scale(
    device, alpha, twiddle_N2, tt_dtype, torch_dtype, label, tol,
):
    """output_scale α scales every output element by α.  Cross-check
    against torch reference (FFT + optional post-twiddle, then × α)."""
    M, P = 8, 128
    torch.manual_seed(53)
    x = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)

    got_r, got_i = _run_radix_pass(
        device, x, None, tt_dtype,
        P=P, twiddle_N2=twiddle_N2, output_scale=alpha,
    )
    got = torch.complex(got_r.reshape(M, P), got_i.reshape(M, P))

    ref_fft = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    if twiddle_N2 > 0:
        ref_fft = _torch_post_twiddle(ref_fft, P=P, N2=twiddle_N2)
    ref = ref_fft * alpha

    for r in range(M):
        rel = _rel_err(got[r], ref[r])
        assert rel < tol, (
            f"[{label}] scale α={alpha} tw_N2={twiddle_N2} row={r} "
            f"rel err {rel:.2e} (tol {tol:.0e})"
        )


# ─── 4. Program cache hit ──────────────────────────────────────────────────
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
def test_radix_pass_program_cache_hit(
    device, tt_dtype, torch_dtype, label, tol,
):
    M, P, N2 = 16, 128, 16
    torch.manual_seed(0)
    x = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)

    _run_radix_pass(device, x, None, tt_dtype, P=P, twiddle_N2=N2)
    n_after_warmup = device.num_program_cache_entries()

    _run_radix_pass(device, x, None, tt_dtype, P=P, twiddle_N2=N2)
    n_after_repeat = device.num_program_cache_entries()

    assert n_after_repeat == n_after_warmup, (
        f"[{label}] fft_radix_pass program cache regression: "
        f"{n_after_warmup} → {n_after_repeat}"
    )


# ─── 5. Metal Trace replay — single dispatch, MUST be trace-safe ──────────
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
def test_radix_pass_metal_trace_replay(
    device, tt_dtype, torch_dtype, label, tol,
):
    """fft_radix_pass is ONE device op (no host-side reshape, no
    intermediate tensor allocations between dispatches), so Metal Trace
    capture+replay must work end-to-end.  This is the trace test the
    two-pass composite test was deferring to."""
    M, P, N2 = 8, 128, 8
    torch.manual_seed(1)
    x = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)

    tt_x = ttnn.from_torch(
        x, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    re_e, im_e = ttnn.experimental.fft_radix_pass(tt_x, P=P, twiddle_N2=N2)
    eager_r = ttnn.to_torch(re_e).to(torch.float32).clone()
    eager_i = ttnn.to_torch(im_e).to(torch.float32).clone()

    # Warm caches.
    tt_x_w = ttnn.from_torch(
        x, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    ttnn.experimental.fft_radix_pass(tt_x_w, P=P, twiddle_N2=N2)

    tt_x_t = ttnn.from_torch(
        x, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    re_t, im_t = ttnn.experimental.fft_radix_pass(tt_x_t, P=P, twiddle_N2=N2)
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
