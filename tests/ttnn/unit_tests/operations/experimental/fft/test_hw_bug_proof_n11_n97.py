# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for bf16 Bluestein at N=11 (M=32) and N=97 (M=256).

These lengths used to fail because a real-only Stockham program for shape
(1, M) was reused for Bluestein's complex inner FFT. That is fixed by hashing
`input_imag_provided`. These tests keep the isolation ladder so a regression
fails at the first broken step rather than only at the public API.

Isolation levels:
  Level 4 — FFT of pure random input (baseline)
  Level 3 — FFT of unit-circle chirp-like input
  Level 2 — FFT of the Bluestein b_cyc kernel
  Level 1 — FFT of a_pad (Bluestein step 3)
  Level 0 — full Bluestein vs isolated step-3 FFT
  Stepwise — each Bluestein stage vs numpy
"""

import math
import struct

import pytest
import torch

import ttnn

# ---------------------------------------------------------------------------
# Pure-Python chirp builders (no numpy, no device — exact fp32/bf16 match)
# ---------------------------------------------------------------------------


def _next_pow2(v: int) -> int:
    v = int(v)
    if v <= 1:
        return 1
    v -= 1
    for s in [1, 2, 4, 8, 16]:
        v |= v >> s
    return v + 1


def _bluestein_M(N: int) -> int:
    M = _next_pow2(2 * N - 1)
    while M < 2 * N + 7 and M < (1 << 30):
        M *= 2
    return M


def _f32_to_bf16_rne(f: float) -> float:
    """Host-side fp32 → bf16 round-to-nearest-even (matches ttnn.from_torch)."""
    bits = struct.unpack("I", struct.pack("f", f))[0]
    if (bits & 0x7F800000) == 0x7F800000:  # NaN → keep NaN
        return struct.unpack("f", struct.pack("I", bits | 0x00400000))[0]
    lsb = (bits >> 16) & 1
    bits = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return struct.unpack("f", struct.pack("I", bits))[0]


def _build_chirp_n_bf16(N: int, sign: int):
    """
    Build chirp_n[n] = exp(sign·πi·(n² mod 2N)/N) quantised to bf16.
    Returns (re_list, im_list) of Python floats already in bf16 grid.
    """
    pi_over_N = math.pi / N
    re, im = [], []
    for n in range(N):
        n_sq_mod = (n * n) % (2 * N)
        angle = sign * pi_over_N * n_sq_mod
        re.append(_f32_to_bf16_rne(math.cos(angle)))
        im.append(_f32_to_bf16_rne(math.sin(angle)))
    return re, im


def _cmul_elementwise(ar, ai, br, bi):
    """Complex multiply, keeping fp32 arithmetic (no bf16 truncation)."""
    outr = [ar[i] * br[i] - ai[i] * bi[i] for i in range(len(ar))]
    outi = [ar[i] * bi[i] + ai[i] * br[i] for i in range(len(ar))]
    return outr, outi


def _build_a_pad(x_bf16, chirp_n_re, chirp_n_im, M: int):
    """
    Bluestein step 1+2: a_pad = (x * chirp_n) zero-padded to M.
    All arithmetic in fp32; result returned as bf16 torch tensor (1, M).
    """
    N = len(x_bf16)
    ar, ai = _cmul_elementwise(x_bf16, [0.0] * N, chirp_n_re, chirp_n_im)
    # zero-pad to M
    ar_pad = ar + [0.0] * (M - N)
    ai_pad = ai + [0.0] * (M - N)
    # Return as bf16 torch tensors
    re_t = torch.tensor(ar_pad, dtype=torch.bfloat16).unsqueeze(0)  # (1, M)
    im_t = torch.tensor(ai_pad, dtype=torch.bfloat16).unsqueeze(0)
    return re_t, im_t


def _build_b_cyc_bf16(N: int, M: int):
    """
    Build the cyclic Bluestein kernel b_cyc quantised to bf16.
    Returns (re_t, im_t) as (1, M) bf16 torch tensors.
    """
    pi_over_N = math.pi / N
    b_re, b_im = [], []
    for m in range(N):
        n_sq_mod = (m * m) % (2 * N)
        angle = pi_over_N * n_sq_mod  # sign = +1 for forward
        b_re.append(_f32_to_bf16_rne(math.cos(angle)))
        b_im.append(_f32_to_bf16_rne(math.sin(angle)))

    r_cyc = [0.0] * M
    i_cyc = [0.0] * M
    for m in range(N):
        r_cyc[m] = b_re[m]
        i_cyc[m] = b_im[m]
    for m in range(1, N):
        r_cyc[M - m] = b_re[m]
        i_cyc[M - m] = b_im[m]

    return (
        torch.tensor(r_cyc, dtype=torch.bfloat16).unsqueeze(0),
        torch.tensor(i_cyc, dtype=torch.bfloat16).unsqueeze(0),
    )


# ---------------------------------------------------------------------------
# Device FFT helper
# ---------------------------------------------------------------------------


def _device_fft(device, re_t: torch.Tensor, im_t: torch.Tensor):
    """
    Run ttnn.experimental.fft(re, im) on device, return (re_out, im_out) torch.
    re_t / im_t: (1, M) bfloat16 torch tensors.
    """
    tt_re = ttnn.from_torch(re_t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_im = ttnn.from_torch(im_t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    got_re_tt, got_im_tt = ttnn.experimental.fft(tt_re, tt_im)
    got_re = ttnn.to_torch(got_re_tt).to(torch.float32).squeeze(0)  # (M,)
    got_im = ttnn.to_torch(got_im_tt).to(torch.float32).squeeze(0)
    return got_re, got_im


def _rel_err(got_re, got_im, ref_re, ref_im) -> float:
    got = torch.complex(got_re, got_im)
    ref = torch.complex(ref_re, ref_im)
    return float((got - ref).abs().norm() / ref.abs().norm().clamp_min(1e-30))


# ---------------------------------------------------------------------------
# Reference FFT (numpy via torch)
# ---------------------------------------------------------------------------


def _ref_fft(re_t: torch.Tensor, im_t: torch.Tensor):
    """numpy-precision reference FFT of (re_t + i·im_t)."""
    x = torch.complex(re_t.float(), im_t.float())
    X = torch.fft.fft(x, dim=-1)
    return X.real.squeeze(0), X.imag.squeeze(0)


# ---------------------------------------------------------------------------
# Core diagnostic: compare device FFT vs numpy for a specific (re, im) input
# ---------------------------------------------------------------------------


def _check_fft(device, re_t, im_t, label: str, tol_pass=0.05, tol_fail=1.0, print_always=True):
    """
    Run device FFT on (re_t, im_t) and compare to numpy.
    Returns (rel_err, passed_soft, clearly_wrong).
    """
    got_re, got_im = _device_fft(device, re_t, im_t)
    ref_re, ref_im = _ref_fft(re_t, im_t)
    err = _rel_err(got_re, got_im, ref_re, ref_im)
    passed_soft = err < tol_pass
    clearly_wrong = err > tol_fail

    tag = "OK  " if passed_soft else ("FAIL" if clearly_wrong else "WARN")
    if print_always or not passed_soft:
        M = re_t.shape[-1]
        print(f"  [{tag}] {label:50s}  rel_err={err:.3e}  M={M}")
    return err, passed_soft, clearly_wrong


# ---------------------------------------------------------------------------
# Test: Level 4 — pure random bf16 input (baseline; must always pass)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M", [32, 256])
def test_level4_random_baseline(device, M):
    """
    Device FFT of a random (1, M) bf16 tensor must match numpy.
    This proves the Stockham kernel is CORRECT for generic inputs.
    """
    torch.manual_seed(0xBEEF)
    re_t = torch.randn(1, M).to(torch.bfloat16)
    im_t = torch.randn(1, M).to(torch.bfloat16)
    err, ok, _ = _check_fft(device, re_t, im_t, label=f"random M={M} (baseline)")
    assert ok, f"Level-4 baseline FAILED M={M}: rel_err={err:.3e} — kernel broken even for random input!"


# ---------------------------------------------------------------------------
# Test: Level 3 — unit-circle chirp-like inputs (random phase, no pairing)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N,M", [(7, 32), (11, 32), (97, 256), (101, 256)])
def test_level3_unit_circle_input(device, N, M):
    """
    FFT of a length-M bf16 vector whose first N elements are unit-circle
    chirp values (like the b_cyc kernel) and the rest are zero.
    This isolates whether the SFPU misbehaves for chirp-structured inputs.
    """
    pi_over_N = math.pi / N
    vals_re = [_f32_to_bf16_rne(math.cos(pi_over_N * ((n * n) % (2 * N)))) for n in range(N)]
    vals_im = [_f32_to_bf16_rne(math.sin(pi_over_N * ((n * n) % (2 * N)))) for n in range(N)]
    re_t = torch.zeros(1, M, dtype=torch.bfloat16)
    im_t = torch.zeros(1, M, dtype=torch.bfloat16)
    re_t[0, :N] = torch.tensor(vals_re, dtype=torch.bfloat16)
    im_t[0, :N] = torch.tensor(vals_im, dtype=torch.bfloat16)

    err, ok, _ = _check_fft(device, re_t, im_t, label=f"unit-circle chirp N={N} M={M}")
    assert ok, f"Level-3 unit-circle N={N} M={M}: rel_err={err:.3e}"


# ---------------------------------------------------------------------------
# Test: Level 2 — exact b_cyc tensor (the Bluestein kernel, cached as B_fft)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N,M", [(7, 32), (11, 32), (97, 256), (101, 256)])
def test_level2_b_cyc_fft(device, N, M):
    """
    FFT of the exact b_cyc tensor used by Bluestein's B_fft precomputation.
    """
    re_t, im_t = _build_b_cyc_bf16(N, M)
    err, ok, _ = _check_fft(device, re_t, im_t, label=f"b_cyc N={N} M={M} (Bluestein kernel)")
    assert ok, f"Level-2 b_cyc N={N} M={M}: rel_err={err:.3e}"


# ---------------------------------------------------------------------------
# Test: Level 1 — exact a_pad tensor (the per-call Bluestein FFT input)
#
# This is the CRITICAL test. a_pad = (x * chirp_n) zero-padded to M.
# The device FFT of a_pad is step 3 of Bluestein. N=11/97 share M with
# the N=7/101 controls and must meet the same accuracy bound.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N,M", [(7, 32), (11, 32), (101, 256), (97, 256)])
def test_level1_a_pad_fft(device, N, M):
    """FFT of the exact a_pad vector that Bluestein step 3 feeds the device."""
    torch.manual_seed(N)  # same seed as test_fft_all_n.py
    x_raw = torch.randn(N, dtype=torch.float32)
    x_bf16 = [float(v) for v in x_raw.to(torch.bfloat16).tolist()]

    chirp_re, chirp_im = _build_chirp_n_bf16(N, sign=-1)  # forward chirp
    re_t, im_t = _build_a_pad(x_bf16, chirp_re, chirp_im, M)

    err, ok, _ = _check_fft(device, re_t, im_t, label=f"a_pad N={N} M={M} (Bluestein step-3 input)")
    assert ok, f"Level-1 a_pad N={N} M={M}: rel_err={err:.3e}"


# ---------------------------------------------------------------------------
# Test: Level 0 — compare full Bluestein output vs step-3 isolated FFT
#
# If the full Bluestein FAILS but the isolated FFT (Level 1) PASSES,
# the bug is in Bluestein orchestration (tensor aliasing, CB reuse, etc.)
# If BOTH fail, the bug is definitively in the Stockham FFT kernel.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N,M", [(11, 32), (97, 256)])
def test_level0_full_vs_step3(device, N, M):
    """Full Bluestein and isolated step-3 FFT must both match the reference."""
    print(f"\n{'='*65}")
    print(f"  Level-0 cross-check: N={N}  M={M}  dtype=bf16")
    print(f"{'='*65}")

    torch.manual_seed(N)
    x_raw = torch.randn(N, dtype=torch.float32)
    x_bf16_t = x_raw.to(torch.bfloat16)

    # ── Full Bluestein via unified API ─────────────────────────────────────
    tt_x = ttnn.from_torch(x_bf16_t.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    got_re_tt, got_im_tt = ttnn.experimental.fft(tt_x)
    got_re = ttnn.to_torch(got_re_tt).float().squeeze(0)
    got_im = ttnn.to_torch(got_im_tt).float().squeeze(0)
    ref_full = torch.fft.fft(x_raw.to(torch.complex64))
    err_full = float((torch.complex(got_re, got_im) - ref_full).abs().norm() / ref_full.abs().norm().clamp_min(1e-30))

    # ── Isolated step-3 FFT (a_pad only) ───────────────────────────────────
    x_bf16 = [float(v) for v in x_bf16_t.tolist()]
    chirp_re, chirp_im = _build_chirp_n_bf16(N, sign=-1)
    re_t, im_t = _build_a_pad(x_bf16, chirp_re, chirp_im, M)
    err_step3, ok_step3, _ = _check_fft(device, re_t, im_t, label=f"  step-3 FFT(a_pad) N={N}", print_always=True)

    print(f"\n  SUMMARY for N={N}:")
    print(f"    Full Bluestein rel_err = {err_full:.3e}")
    print(f"    Step-3 isolated rel_err = {err_step3:.3e}")

    assert err_full <= 0.15, f"Level-0 full Bluestein N={N}: rel_err={err_full:.3e}"
    assert ok_step3, f"Level-0 step-3 FFT N={N}: rel_err={err_step3:.3e}"


# ---------------------------------------------------------------------------
# Helpers for step-by-step tests
# ---------------------------------------------------------------------------


def _upload(device, t: torch.Tensor):
    """Upload a bfloat16 torch tensor to the device as a ttnn tensor."""
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _download(tt) -> torch.Tensor:
    """Download a ttnn tensor back to a float32 torch tensor."""
    return ttnn.to_torch(tt).to(torch.float32)


def _numpy_cmul(ar, ai, br, bi):
    """Element-wise complex multiply in fp64 (reference)."""
    return ar * br - ai * bi, ar * bi + ai * br


def _numpy_fft(re, im):
    x = torch.complex(re.float(), im.float())
    X = torch.fft.fft(x, dim=-1)
    return X.real, X.imag


def _numpy_ifft(re, im):
    x = torch.complex(re.float(), im.float())
    y = torch.fft.ifft(x, dim=-1)
    return y.real, y.imag


def _rel_err_tensors(got_r, got_i, ref_r, ref_i) -> float:
    got = torch.complex(got_r.float(), got_i.float())
    ref = torch.complex(ref_r.float(), ref_i.float())
    return float((got - ref).abs().norm() / ref.abs().norm().clamp_min(1e-30))


# ---------------------------------------------------------------------------
# Test: Step-by-step Bluestein chain isolation
#
# Runs each of the 7 Bluestein steps individually on device using Python-
# accessible ttnn ops, comparing each intermediate result against numpy.
# The FIRST step whose output diverges is the one containing the bug.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N,M", [(7, 32), (11, 32), (97, 256), (101, 256)])
def test_stepwise_bluestein_chain(device, N, M):
    """
    Manually execute each Bluestein step using ttnn Python ops and compare
    every intermediate result against the numpy reference.

    Steps:
      1. a = x * chirp_n           (ttnn.experimental.complex_mul)
      2. a_pad = pad(a, M)         (torch.nn.functional.pad on CPU → upload)
      3. A = FFT(a_pad)            (ttnn.experimental.fft)
      4. C = A * B_fft             (ttnn.experimental.complex_mul)
      5. c = IFFT(C)               (ttnn.experimental.ifft)
      6. c_n = c[:N]               (slice on CPU → download)
      7. X = c_n * chirp_k         (ttnn.experimental.complex_mul)

    A mismatch at step K while step K-1 is correct pins the bug to step K.
    """
    print(f"\n{'='*65}")
    print(f"  Step-by-step Bluestein chain: N={N}  M={M}  dtype=bf16")
    print(f"{'='*65}")

    # ── Host data ────────────────────────────────────────────────────────
    torch.manual_seed(N)
    x_f32 = torch.randn(N, dtype=torch.float32)
    x_bf16 = x_f32.to(torch.bfloat16)

    chirp_re_list, chirp_im_list = _build_chirp_n_bf16(N, sign=-1)
    chirp_r = torch.tensor(chirp_re_list, dtype=torch.bfloat16).unsqueeze(0)  # (1, N)
    chirp_i = torch.tensor(chirp_im_list, dtype=torch.bfloat16).unsqueeze(0)

    b_cyc_r, b_cyc_i = _build_b_cyc_bf16(N, M)  # (1, M) bf16

    # Reference numpy B_fft (host fp64)
    B_ref_r, B_ref_i = _numpy_fft(b_cyc_r.float(), b_cyc_i.float())

    # ── Step 1: a = x * chirp_n ─────────────────────────────────────────
    x_r_tt = _upload(device, x_bf16.unsqueeze(0))  # (1, N)
    x_i_tt = _upload(device, torch.zeros(1, N, dtype=torch.bfloat16))
    c_r_tt = _upload(device, chirp_r)
    c_i_tt = _upload(device, chirp_i)

    a_r_tt, a_i_tt = ttnn.experimental.complex_mul(x_r_tt, x_i_tt, c_r_tt, c_i_tt)
    a_r = _download(a_r_tt).squeeze(0)  # (N,)
    a_i = _download(a_i_tt).squeeze(0)

    ref_a_r = x_f32 * torch.tensor(chirp_re_list)  # fp32 reference
    ref_a_i = x_f32 * torch.tensor(chirp_im_list)
    err1 = _rel_err_tensors(a_r.unsqueeze(0), a_i.unsqueeze(0), ref_a_r.unsqueeze(0), ref_a_i.unsqueeze(0))
    print(f"  Step 1 (chirp pre-mul)  rel_err = {err1:.3e}  " f"{'FAIL <<<' if err1 > 0.1 else 'ok'}")

    # ── Step 2: a_pad = zero-pad a to M ─────────────────────────────────
    # Pad is done on CPU (using the device output as base)
    a_r_full = torch.zeros(1, M, dtype=torch.float32)
    a_i_full = torch.zeros(1, M, dtype=torch.float32)
    a_r_full[0, :N] = a_r
    a_i_full[0, :N] = a_i
    a_pad_r = a_r_full.to(torch.bfloat16)
    a_pad_i = a_i_full.to(torch.bfloat16)

    ref_a_pad_r = torch.zeros(1, M)
    ref_a_pad_i = torch.zeros(1, M)
    ref_a_pad_r[0, :N] = ref_a_r
    ref_a_pad_i[0, :N] = ref_a_i
    err2 = _rel_err_tensors(a_pad_r, a_pad_i, ref_a_pad_r, ref_a_pad_i)
    print(f"  Step 2 (zero-pad)       rel_err = {err2:.3e}  " f"{'FAIL <<<' if err2 > 0.1 else 'ok'}")

    # ── Step 3: A = FFT(a_pad) ───────────────────────────────────────────
    A_r_tt, A_i_tt = _device_fft(device, a_pad_r, a_pad_i)  # (M,) float32
    ref_A_r, ref_A_i = _numpy_fft(ref_a_pad_r, ref_a_pad_i)
    A_r_tt_2d = A_r_tt.unsqueeze(0)
    A_i_tt_2d = A_i_tt.unsqueeze(0)
    err3 = _rel_err_tensors(A_r_tt_2d, A_i_tt_2d, ref_A_r, ref_A_i)
    print(f"  Step 3 (FFT a_pad)      rel_err = {err3:.3e}  " f"{'FAIL <<<' if err3 > 0.1 else 'ok'}")

    # ── Step 4: C = A * B_fft ────────────────────────────────────────────
    # Compute B_fft on device (fresh, not from Bluestein cache)
    B_r_tt_2d, B_i_tt_2d = _device_fft(device, b_cyc_r, b_cyc_i)  # (M,) float32
    B_r_tt_2d = B_r_tt_2d.unsqueeze(0)
    B_i_tt_2d = B_i_tt_2d.unsqueeze(0)

    # Upload A and B to device for complex_mul
    A_r_dev = _upload(device, A_r_tt_2d.to(torch.bfloat16))
    A_i_dev = _upload(device, A_i_tt_2d.to(torch.bfloat16))
    B_r_dev = _upload(device, B_r_tt_2d.to(torch.bfloat16))
    B_i_dev = _upload(device, B_i_tt_2d.to(torch.bfloat16))

    C_r_tt, C_i_tt = ttnn.experimental.complex_mul(A_r_dev, A_i_dev, B_r_dev, B_i_dev)
    C_r = _download(C_r_tt)  # (1, M)
    C_i = _download(C_i_tt)

    ref_C_r, ref_C_i = _numpy_cmul(ref_A_r, ref_A_i, B_ref_r, B_ref_i)
    err4 = _rel_err_tensors(C_r, C_i, ref_C_r, ref_C_i)
    print(f"  Step 4 (A * B_fft)      rel_err = {err4:.3e}  " f"{'FAIL <<<' if err4 > 0.1 else 'ok'}")

    # ── Step 5: c = IFFT(C) ──────────────────────────────────────────────
    C_r_bf16 = C_r.to(torch.bfloat16)
    C_i_bf16 = C_i.to(torch.bfloat16)
    C_r_dev = _upload(device, C_r_bf16)
    C_i_dev = _upload(device, C_i_bf16)

    c_r_tt, c_i_tt = ttnn.experimental.ifft(C_r_dev, C_i_dev)
    c_r = _download(c_r_tt)  # (1, M)
    c_i = _download(c_i_tt)

    ref_c_r, ref_c_i = _numpy_ifft(ref_C_r, ref_C_i)
    err5 = _rel_err_tensors(c_r, c_i, ref_c_r, ref_c_i)
    print(f"  Step 5 (IFFT)           rel_err = {err5:.3e}  " f"{'FAIL <<<' if err5 > 0.1 else 'ok'}")

    # ── Step 6+7: slice + chirp_k post-mul ──────────────────────────────
    c_r_n = c_r[0, :N].unsqueeze(0)  # (1, N) - slice on CPU
    c_i_n = c_i[0, :N].unsqueeze(0)
    c_r_n_bf16 = c_r_n.to(torch.bfloat16)
    c_i_n_bf16 = c_i_n.to(torch.bfloat16)

    c_r_dev = _upload(device, c_r_n_bf16)
    c_i_dev = _upload(device, c_i_n_bf16)
    ck_r_dev = _upload(device, chirp_r)
    ck_i_dev = _upload(device, chirp_i)

    X_r_tt, X_i_tt = ttnn.experimental.complex_mul(c_r_dev, c_i_dev, ck_r_dev, ck_i_dev)
    X_r = _download(X_r_tt)  # (1, N)
    X_i = _download(X_i_tt)

    ref_full = torch.fft.fft(x_f32.to(torch.complex64))
    ref_X_r = ref_full.real.unsqueeze(0)
    ref_X_i = ref_full.imag.unsqueeze(0)
    err67 = _rel_err_tensors(X_r, X_i, ref_X_r, ref_X_i)
    print(f"  Steps 6+7 (slice+post)  rel_err = {err67:.3e}  " f"{'FAIL <<<' if err67 > 0.15 else 'ok'}")

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    first_fail = None
    for step, err, tol, name in [
        (1, err1, 0.1, "chirp pre-mul"),
        (2, err2, 0.1, "zero-pad"),
        (3, err3, 0.1, "FFT(a_pad)"),
        (4, err4, 0.1, "A*B_fft cmul"),
        (5, err5, 0.1, "IFFT"),
        (67, err67, 0.15, "slice+post-mul"),
    ]:
        if err > tol and first_fail is None:
            first_fail = (step, name, err)

    if first_fail:
        s, nm, e = first_fail
        pytest.fail(f"Stepwise Bluestein N={N} first failure at step {s} ({nm}): rel_err={e:.3e}")
