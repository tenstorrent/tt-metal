# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for ttnn.experimental.bluestein_fft (commit 6d).

Bluestein's chirp-Z transform extends our pow-2 FFT chain to handle
**arbitrary** N (including primes and other non-pow-2 lengths).

Coverage matrix (commit 6d core):
  - small N: 3, 5, 7, 11, 13, 16, 17, 32, 33, 100
  - medium N: 257 (prime), 384 (3 · 128), 511
  - real-only and complex input
  - fp32 + bf16
  - program-cache hit (chirp + B precomputed; second call only does
    the per-N dispatch chain)

Aggressive cases (N up to commit-6d cap M ≤ 2^20) are gated behind
TT_FFT_AGGRESSIVE=1.
"""

import os
import pytest
import torch
import ttnn


pytestmark = pytest.mark.skipif(
    os.environ.get("TT_FFT_NATIVE", "1") == "0",
    reason="TT_FFT_NATIVE=1 not set; new ProgramDescriptor path is gated.",
)


_DTYPES = [
    # bf16 tol is loose for Bluestein because two FFTs + 3 cmul chain
    # accumulates rounding; the per-bin worst case is dominated by the
    # post-IFFT slice region near the chirp boundary.
    (ttnn.float32,  torch.float32,  "fp32", 5e-4),
    (ttnn.bfloat16, torch.bfloat16, "bf16", 1.5e-1),
]


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).abs().norm() / ref.abs().norm().clamp_min(1e-30))


def _run_bluestein(device, x_re, x_im, N, tt_dtype, B=1):
    """Upload (B, N) real and (optional) imag halves and call bluestein_fft."""
    tt_xr = ttnn.from_torch(
        x_re.reshape(B, N), dtype=tt_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    if x_im is not None:
        tt_xi = ttnn.from_torch(
            x_im.reshape(B, N), dtype=tt_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out_r, out_i = ttnn.experimental.bluestein_fft(tt_xr, tt_xi, N=N)
    else:
        out_r, out_i = ttnn.experimental.bluestein_fft(tt_xr, N=N)
    return (
        ttnn.to_torch(out_r).reshape(B, N),
        ttnn.to_torch(out_i).reshape(B, N),
    )


# ─── 1. Real-input correctness ──────────────────────────────────────────
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize(
    "N",
    # Primes + small composites + just under / just over a pow-2 boundary.
    [3, 5, 7, 11, 13, 16, 17, 31, 32, 33, 100, 127, 128, 129],
    ids=lambda v: f"N{v}",
)
def test_bluestein_real_correctness(device, N, tt_dtype, torch_dtype, label, tol):
    torch.manual_seed(N)
    x = torch.randn(N, dtype=torch.float32).to(torch_dtype)

    got_r, got_i = _run_bluestein(device, x, None, N, tt_dtype)
    got = torch.complex(got_r[0].to(torch.float32), got_i[0].to(torch.float32))

    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)

    rel = _rel_err(got, ref)
    assert rel < tol, (
        f"[{label}] N={N} (real input) rel err {rel:.2e} (tol {tol:.0e})"
    )


# ─── 2. Complex-input correctness ───────────────────────────────────────
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("N", [5, 13, 31, 100, 127], ids=lambda v: f"N{v}")
def test_bluestein_complex_correctness(device, N, tt_dtype, torch_dtype, label, tol):
    torch.manual_seed(N + 1)
    x_re = torch.randn(N, dtype=torch.float32).to(torch_dtype)
    x_im = torch.randn(N, dtype=torch.float32).to(torch_dtype)

    got_r, got_i = _run_bluestein(device, x_re, x_im, N, tt_dtype)
    got = torch.complex(got_r[0].to(torch.float32), got_i[0].to(torch.float32))

    x = torch.complex(x_re.to(torch.float32), x_im.to(torch.float32)).to(torch.complex64)
    ref = torch.fft.fft(x, dim=-1)

    rel = _rel_err(got, ref)
    assert rel < tol, (
        f"[{label}] N={N} (complex input) rel err {rel:.2e} (tol {tol:.0e})"
    )


# ─── 2b. Batched correctness (6e-1) ─────────────────────────────────────
# Each batch row is an INDEPENDENT length-N DFT.  We use distinct random
# data per row to confirm the chirp replication doesn't accidentally
# couple rows.
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("N", [5, 17, 33, 100, 127, 257],
                         ids=lambda v: f"N{v}")
@pytest.mark.parametrize("B", [2, 4], ids=lambda v: f"B{v}")
def test_bluestein_batched_real_correctness(
    device, N, B, tt_dtype, torch_dtype, label, tol
):
    torch.manual_seed(N * 1000 + B)
    x = torch.randn(B, N, dtype=torch.float32).to(torch_dtype)

    got_r, got_i = _run_bluestein(device, x, None, N, tt_dtype, B=B)
    got = torch.complex(got_r.to(torch.float32), got_i.to(torch.float32))

    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)

    rel = _rel_err(got, ref)
    assert rel < tol, (
        f"[{label}] B={B} N={N} (batched real) rel err {rel:.2e} (tol {tol:.0e})"
    )

    # Also verify per-row independence — row i in `got` matches row i in
    # `ref` standalone (catches accidental cross-row contamination).
    for i in range(B):
        rel_i = _rel_err(got[i], ref[i])
        assert rel_i < tol, (
            f"[{label}] B={B} N={N} row={i} per-row rel err {rel_i:.2e}"
        )


@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("N", [13, 100, 127], ids=lambda v: f"N{v}")
@pytest.mark.parametrize("B", [2, 4], ids=lambda v: f"B{v}")
def test_bluestein_batched_complex_correctness(
    device, N, B, tt_dtype, torch_dtype, label, tol
):
    torch.manual_seed(N * 7919 + B)
    x_re = torch.randn(B, N, dtype=torch.float32).to(torch_dtype)
    x_im = torch.randn(B, N, dtype=torch.float32).to(torch_dtype)

    got_r, got_i = _run_bluestein(device, x_re, x_im, N, tt_dtype, B=B)
    got = torch.complex(got_r.to(torch.float32), got_i.to(torch.float32))

    x = torch.complex(x_re.to(torch.float32), x_im.to(torch.float32)).to(torch.complex64)
    ref = torch.fft.fft(x, dim=-1)

    rel = _rel_err(got, ref)
    assert rel < tol, (
        f"[{label}] B={B} N={N} (batched complex) rel err {rel:.2e} (tol {tol:.0e})"
    )


# ─── 3. Program-cache / plan reuse ──────────────────────────────────────
# Second call with the same N should hit BOTH the JIT program cache for
# every device op AND the host-side BluesteinPlan cache (chirp + B not
# rebuilt).
def test_bluestein_program_cache_hit(device):
    N = 17  # prime, small
    torch.manual_seed(0)
    x = torch.randn(N, dtype=torch.float32)

    for trial in range(3):
        got_r, got_i = _run_bluestein(device, x, None, N, ttnn.float32)
        got = torch.complex(got_r[0].to(torch.float32), got_i[0].to(torch.float32))
        ref = torch.fft.fft(x.to(torch.complex64), dim=-1)
        rel = _rel_err(got, ref)
        assert rel < 5e-4, f"trial={trial} N={N} rel err {rel:.2e}"


# Same with B > 1 — confirms the (B, N) plan is also reused.
def test_bluestein_batched_program_cache_hit(device):
    N, B = 17, 4
    torch.manual_seed(1)
    x = torch.randn(B, N, dtype=torch.float32)

    for trial in range(3):
        got_r, got_i = _run_bluestein(device, x, None, N, ttnn.float32, B=B)
        got = torch.complex(got_r.to(torch.float32), got_i.to(torch.float32))
        ref = torch.fft.fft(x.to(torch.complex64), dim=-1)
        rel = _rel_err(got, ref)
        assert rel < 5e-4, f"trial={trial} B={B} N={N} rel err {rel:.2e}"


# ─── 4. Aggressive (gated) — larger N approaching commit-6d cap ────────
# Commit 6d cap: M = next_pow2(2*N - 1) ≤ 2^20 = 1M  →  N ≤ 524_288.
@pytest.mark.skipif(
    os.environ.get("TT_FFT_AGGRESSIVE", "0") != "1",
    reason="TT_FFT_AGGRESSIVE=1 not set; large-N Bluestein test is gated.",
)
@pytest.mark.parametrize(
    "N",
    # 1009 prime, 4097 just over pow-2, 65537 prime (Fermat), 524288 cap.
    # N=65537: M=2^18=262144; zero_pad_to_m and trim_to_n cannot be done
    # without L1 overflow because N%1024=1 (non-1024-aligned).  A dedicated
    # streaming kernel is required — tracked as a future enhancement.
    [
        1009,
        4097,
        pytest.param(
            65537,
            marks=pytest.mark.xfail(
                reason=(
                    "N=65537 Bluestein: M=262144 with non-1024-aligned N causes L1 "
                    "overflow in zero_pad_to_m (concat CB=2MB) and trim_to_n "
                    "(ttnn::slice CB=16MB). Requires a new streaming kernel."
                ),
                strict=False,
            ),
        ),
    ],
    ids=lambda v: f"N{v}",
)
def test_bluestein_aggressive(device, N):
    torch.manual_seed(N)
    x = torch.randn(N, dtype=torch.float32)

    got_r, got_i = _run_bluestein(device, x, None, N, ttnn.float32)
    got = torch.complex(got_r[0].to(torch.float32), got_i[0].to(torch.float32))
    ref = torch.fft.fft(x.to(torch.complex64), dim=-1)

    # Two FFTs + 3 cmul chain at N up to 65k accumulates a few extra ULP;
    # 2e-3 leaves room for that without masking real bugs.
    rel = _rel_err(got, ref)
    assert rel < 2e-3, f"N={N} rel err {rel:.2e}"


# ─── 6e-3: comprehensive accuracy sweep (gated) ─────────────────────────
# This block doubles as the source for the paper's accuracy table.  We
# reference against complex128 (fp64) torch.fft so the rounding floor
# isn't dominated by torch's own fp32 path; this gives us a clean
# measurement of the on-device chain's accuracy.
#
# Gated behind TT_FFT_BLUESTEIN_SWEEP=1 because the full matrix is
# ~140 cases and takes a few minutes.

_BLUESTEIN_SWEEP_N = [
    # primes (small)
    3, 5, 7, 11, 13, 17, 19, 23, 31,
    # Fermat-like primes
    257, 65537,
    # Mersenne prime at the cap
    524287,
    # pow-2 (Bluestein still works, just inefficient)
    16, 128, 1024, 16384,
    # just-around-pow-2 to stress padding logic
    127, 129, 1023, 1025, 4095, 4097,
    # composite non-pow-2 (common DSP sizes)
    100, 384, 1000, 4096 * 3 // 2,  # 6144
    # large prime well within cap
    100003,
]

_BLUESTEIN_SWEEP_DTYPES = [
    (ttnn.float32,  torch.float32,  "fp32", 1e-3),
    (ttnn.bfloat16, torch.bfloat16, "bf16", 2.0e-1),
]


def _rel_err_fp64(got_fp32: torch.Tensor, ref_fp64: torch.Tensor) -> float:
    """Use the fp64 reference but compute the relative error in fp64
    to avoid the rounding floor of fp32 norm."""
    got = got_fp32.to(torch.complex128)
    diff_norm = (got - ref_fp64).abs().to(torch.float64).norm()
    ref_norm  = ref_fp64.abs().to(torch.float64).norm().clamp_min(1e-300)
    return float(diff_norm / ref_norm)


@pytest.mark.skipif(
    os.environ.get("TT_FFT_BLUESTEIN_SWEEP", "0") != "1",
    reason="TT_FFT_BLUESTEIN_SWEEP=1 not set; comprehensive Bluestein "
           "sweep is gated (~140 cases, several minutes).",
)
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _BLUESTEIN_SWEEP_DTYPES,
                         ids=[d[2] for d in _BLUESTEIN_SWEEP_DTYPES])
@pytest.mark.parametrize("N", _BLUESTEIN_SWEEP_N, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("B", [1, 2], ids=lambda v: f"B{v}")
@pytest.mark.parametrize("inp", ["real", "complex"])
def test_bluestein_sweep_accuracy(
    device, N, B, tt_dtype, torch_dtype, label, tol, inp,
):
    """Full accuracy sweep for the paper's accuracy table.

    Reports per-case rel-err measured against torch's complex128 FFT —
    this is the most defensible reference because it removes torch's own
    fp32 rounding from the comparison.
    """
    torch.manual_seed(N * 31 + B + (0 if inp == "real" else 1))
    x_re_fp32 = torch.randn(B, N, dtype=torch.float32)

    if inp == "complex":
        x_im_fp32 = torch.randn(B, N, dtype=torch.float32)
    else:
        x_im_fp32 = None

    x_re = x_re_fp32.to(torch_dtype)
    x_im = x_im_fp32.to(torch_dtype) if x_im_fp32 is not None else None

    got_r, got_i = _run_bluestein(device, x_re, x_im, N, tt_dtype, B=B)
    got = torch.complex(got_r.to(torch.float32), got_i.to(torch.float32))

    # fp64 reference (the SOURCE of truth — torch.fft promotes
    # internally; we explicitly cast to complex128 to be sure).
    x_ref = torch.complex(
        x_re_fp32 if x_im_fp32 is None else x_re_fp32,
        torch.zeros_like(x_re_fp32) if x_im_fp32 is None else x_im_fp32,
    ).to(torch.complex128)
    ref = torch.fft.fft(x_ref, dim=-1)

    rel = _rel_err_fp64(got, ref)
    assert rel < tol, (
        f"[{label}] B={B} N={N} inp={inp} rel err {rel:.2e} (tol {tol:.0e})"
    )


# ─── 6e-3: program-cache hit across a sweep ─────────────────────────────
# Confirms that running the same (N, dtype, B) twice in a row reuses the
# cached BluesteinPlan AND every device op's JIT program-cache entry.
# We don't have direct access to cache stats from Python, so we just
# verify functional repeatability — the underlying speedup is what the
# benchmark harness will measure.
@pytest.mark.skipif(
    os.environ.get("TT_FFT_BLUESTEIN_SWEEP", "0") != "1",
    reason="TT_FFT_BLUESTEIN_SWEEP=1 not set; sweep prog-cache test gated.",
)
@pytest.mark.parametrize("N", [17, 257, 1009], ids=lambda v: f"N{v}")
@pytest.mark.parametrize("B", [1, 2], ids=lambda v: f"B{v}")
def test_bluestein_sweep_prog_cache_hit(device, N, B):
    torch.manual_seed(0xBADF00D ^ N ^ B)
    x = torch.randn(B, N, dtype=torch.float32)

    last = None
    for trial in range(3):
        got_r, got_i = _run_bluestein(device, x, None, N, ttnn.float32, B=B)
        got = torch.complex(got_r.to(torch.float32), got_i.to(torch.float32))

        if last is not None:
            # Bit-exact across calls when nothing changes (deterministic
            # on-device pipeline).
            diff = (got - last).abs().max().item()
            assert diff == 0.0, (
                f"trial={trial} N={N} B={B} non-deterministic across "
                f"prog-cache hits, max-diff={diff:.3e}"
            )
        last = got
