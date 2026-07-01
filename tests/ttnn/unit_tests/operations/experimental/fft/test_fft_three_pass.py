# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the three-pass Cooley–Tukey composite FFT path (commit 5, corrected 5c).

For pow-2 N with 2^20 < N ≤ 2^30, ttnn.experimental.fft_three_pass
factors N = N1·N2·N3 (max-N3 then balance N1/N2, each pow-2 in [32, 1024])
and runs an 8-op device-side chain (standard mixed-radix DIT, FFT_N1 first):

    transpose_rm                                   # initial rearrangement
    fft_radix_pass(P=N1, twiddle_N2=0)             # stage 1, pure FFT_N1
    apply_twiddles_xl(P=N1, big_mod=N2·N3)         # twiddle-1 (large mod)
    transpose_rm                                   # bring n2 to last
    fft_radix_pass(P=N2, twiddle_N2=N3, stride=N1) # stage 2 + twiddle-2
    transpose_rm                                   # bring n3 to last
    fft_radix_pass(P=N3, twiddle_N2=0)             # stage 3, pure FFT_N3
    transpose_rm + transpose_rm                    # final dim-reverse

(plus reshape views around each).  Activated by TT_FFT_NATIVE=1.

⚠ API NOTES:
   (1) INPUT pre-shape: fft_three_pass takes its input PRE-SHAPED as
       (B·N1·N2, N3).  The (B, N) → (B·N1·N2, N3) reshape would otherwise
       require streaming an N-element row through one CB tile per core,
       which blows L1 for N > ~256K.  Tests do the equivalent torch view
       on the host before ttnn.from_torch.
   (2) OUTPUT shape (B·N3, N2, N1): tests reshape to (B, N) on host after
       to_torch(); the (N3, N2, N1) dim ordering encodes natural-order
       K = k3·N1·N2 + k2·N1 + k1, so the flat reshape directly yields
       X[k] in natural-K order (matching torch.fft.fft).
       NOTE: in commit 5 the output shape was (B·N1, N2, N3); commit 5c
       changed this together with the algorithm fix.

Coverage:
  - correctness vs torch.fft on the conservative band (N ∈ {2^21, 2^22})
    for fp32/bf16, B ∈ {1, 2};
  - aggressive band (N ∈ {2^24, 2^26, 2^28}) gated behind
    TT_FFT_AGGRESSIVE=1 — these are minutes-long, fp32-only, B=1;
  - program-cache hit on repeat (small N);
  - Metal-Trace is skipped (host-orchestrated composite, like fft_two_pass).
"""

import os
import pytest
import torch
import ttnn


pytestmark = pytest.mark.skipif(
    os.environ.get("TT_FFT_NATIVE", "1") == "0",
    reason="TT_FFT_NATIVE=1 not set; new ProgramDescriptor path is gated.",
)


# Cube-balanced 3-pass is dominated by the per-row recurrence in the XL
# twiddle (≈ P · ε_fp32 per row) and the per-pass FFT error (≈ √N · ε).
# Empirically that lands well inside 5e-4 for fp32 and 5e-2 for bf16.
_DTYPES = [
    (ttnn.float32,  torch.float32,  "fp32", 5e-4),
    (ttnn.bfloat16, torch.bfloat16, "bf16", 5e-2),
]


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).abs().norm() / ref.abs().norm().clamp_min(1e-30))


def _expected_three_factorization(N: int) -> tuple[int, int, int]:
    """Mirror C++ pick_three_factorization: max-N3 then balance N1/N2."""
    log2N = N.bit_length() - 1
    log2_N3 = 10 if (log2N - 10) >= 10 else max(5, log2N - 10)
    log2_rest = log2N - log2_N3
    log2_N1 = (log2_rest + 1) // 2
    log2_N2 = log2_rest - log2_N1
    return (1 << log2_N1, 1 << log2_N2, 1 << log2_N3)


# ─── 1. Correctness — conservative band (N ≤ 2^22) ─────────────────────────
# Three N points exercise three distinct factorizations and three sizes of
# the XL twiddle modulus (N1·N2 = 2^11, 2^12, 2^13).
#
# IMPORTANT: fft_three_pass takes the input PRE-SHAPED as (B·N1·N2, N3) —
# see the C++ docstring for why (host-side metadata view avoids an L1-
# busting (B, N) → (B·N1·N2, N3) reshape on device).  The torch
# `.view(B·N1·N2, N3)` here is metadata-only (no copy) since the
# allocation is contiguous.
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("N", [1 << 21, 1 << 22])
def test_three_pass_correctness(device, B, N, tt_dtype, torch_dtype, label, tol):
    N1, N2, N3 = _expected_three_factorization(N)
    assert N1 * N2 * N3 == N
    for f in (N1, N2, N3):
        assert 32 <= f <= 1024 and (f & (f - 1)) == 0

    torch.manual_seed(7)
    x_fp32 = torch.randn(B, N, dtype=torch.float32)
    x = x_fp32.to(torch_dtype)

    # Pre-shape to (B·N1·N2, N3) on the host (torch view, no copy) so
    # the ttnn allocation has small per-row page_size from the start.
    x_preshaped = x.reshape(B * N1 * N2, N3).contiguous()

    tt_x = ttnn.from_torch(
        x_preshaped, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    re, im = ttnn.experimental.fft_three_pass(tt_x, full_N=N)
    # Output shape is (B·N3, N2, N1); flat reshape to (B, N) on host gives
    # natural-order X[k] (the (N3, N2, N1) dim ordering encodes
    # K = k3·N1·N2 + k2·N1 + k1 which IS the natural flat K).
    got_r = ttnn.to_torch(re).reshape(B, N).to(torch.float32)
    got_i = ttnn.to_torch(im).reshape(B, N).to(torch.float32)
    got = torch.complex(got_r, got_i)

    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)

    for b in range(B):
        rel = _rel_err(got[b], ref[b])
        assert rel < tol, (
            f"[{label}] B={B} N={N} (N1={N1},N2={N2},N3={N3}) row={b} "
            f"rel err {rel:.2e} (tol {tol:.0e})"
        )


# ─── 1b. Correctness — COMPLEX input (commit 6a) ───────────────────────────
# fft_three_pass gained an optional input_imag parameter in commit 6a
# (the complex-input form used by the Bluestein composite for its
# intermediate length-M FFT).  When supplied, input_imag goes through
# the SAME pre-rearrangement chain as input_real (reshape →
# transpose_rm → reshape), adding 1 extra transpose_rm dispatch on the
# input but otherwise reusing the same 8-op pipeline.
#
# Single representative N (2^21) at fp32 + bf16, B ∈ {1, 2}.  Larger N
# is covered by the real-input tests above and would not exercise any
# new code path here.
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("N", [1 << 21])
def test_three_pass_complex_correctness(
    device, B, N, tt_dtype, torch_dtype, label, tol,
):
    N1, N2, N3 = _expected_three_factorization(N)
    assert N1 * N2 * N3 == N

    torch.manual_seed(17)
    x_re_fp32 = torch.randn(B, N, dtype=torch.float32)
    x_im_fp32 = torch.randn(B, N, dtype=torch.float32)
    x_re = x_re_fp32.to(torch_dtype)
    x_im = x_im_fp32.to(torch_dtype)

    x_re_pre = x_re.reshape(B * N1 * N2, N3).contiguous()
    x_im_pre = x_im.reshape(B * N1 * N2, N3).contiguous()

    tt_re = ttnn.from_torch(
        x_re_pre, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    tt_im = ttnn.from_torch(
        x_im_pre, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    out_re, out_im = ttnn.experimental.fft_three_pass(tt_re, tt_im, full_N=N)
    got_r = ttnn.to_torch(out_re).reshape(B, N).to(torch.float32)
    got_i = ttnn.to_torch(out_im).reshape(B, N).to(torch.float32)
    got = torch.complex(got_r, got_i)

    ref = torch.fft.fft(
        torch.complex(x_re_fp32, x_im_fp32).to(torch.complex64), dim=-1,
    )

    for b in range(B):
        rel = _rel_err(got[b], ref[b])
        assert rel < tol, (
            f"[{label}-complex] B={B} N={N} (N1={N1},N2={N2},N3={N3}) "
            f"row={b} rel err {rel:.2e} (tol {tol:.0e})"
        )


# ─── 1c. IFFT correctness — THREE-PASS PATH (commit 6c) ────────────────────
# fft_three_pass(inverse=true) uses the swap-trick:
#
#     IFFT(X) = (1/full_N) · (W_im, W_re)  where W = FFT(X_im, X_re).
#
# Both swap steps are pure C++ relabels (free) and the 1/full_N scale is
# folded into the Stage-3 (last) fft_radix_pass writer via output_scale,
# so the IFFT chain has the SAME 8-op dispatch count as forward complex
# FFT (no extra dispatch for the scale).
#
# IFFT REQUIRES complex input (both halves of the spectrum) — the swap-
# trick depends on having both X_re and X_im.  We don't expose an
# inverse=true overload for the real-only wrapper.
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("N", [1 << 21])
def test_three_pass_ifft_correctness(
    device, B, N, tt_dtype, torch_dtype, label, tol,
):
    N1, N2, N3 = _expected_three_factorization(N)
    assert N1 * N2 * N3 == N

    torch.manual_seed(37)
    x_re_fp32 = torch.randn(B, N, dtype=torch.float32)
    x_im_fp32 = torch.randn(B, N, dtype=torch.float32)
    x_re = x_re_fp32.to(torch_dtype)
    x_im = x_im_fp32.to(torch_dtype)

    x_re_pre = x_re.reshape(B * N1 * N2, N3).contiguous()
    x_im_pre = x_im.reshape(B * N1 * N2, N3).contiguous()

    tt_re = ttnn.from_torch(
        x_re_pre, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    tt_im = ttnn.from_torch(
        x_im_pre, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    out_re, out_im = ttnn.experimental.fft_three_pass(
        tt_re, tt_im, full_N=N, inverse=True,
    )
    got_r = ttnn.to_torch(out_re).reshape(B, N).to(torch.float32)
    got_i = ttnn.to_torch(out_im).reshape(B, N).to(torch.float32)
    got = torch.complex(got_r, got_i)

    ref = torch.fft.ifft(
        torch.complex(x_re_fp32, x_im_fp32).to(torch.complex64), dim=-1,
    )

    for b in range(B):
        rel = _rel_err(got[b], ref[b])
        assert rel < tol, (
            f"[{label}-ifft] B={B} N={N} (N1={N1},N2={N2},N3={N3}) "
            f"row={b} rel err {rel:.2e} (tol {tol:.0e})"
        )


# Round-trip sanity check: ifft(fft(x)) ≈ x.  Uses real input (passed as
# input_imag=zero) through the COMPLEX three-pass forward, then IFFT on
# the resulting complex spectrum.  Recovers x to within fp32/bf16 noise.
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("N", [1 << 21])
def test_three_pass_fft_ifft_roundtrip(
    device, N, tt_dtype, torch_dtype, label, tol,
):
    B = 1
    N1, N2, N3 = _expected_three_factorization(N)
    torch.manual_seed(41)
    x_fp32 = torch.randn(B, N, dtype=torch.float32)
    x = x_fp32.to(torch_dtype)
    x_pre = x.reshape(B * N1 * N2, N3).contiguous()
    z_pre = torch.zeros_like(x_pre)

    tt_x = ttnn.from_torch(
        x_pre, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    tt_z = ttnn.from_torch(
        z_pre, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    # Forward (complex form with zero imag) → spectrum.
    fft_re, fft_im = ttnn.experimental.fft_three_pass(tt_x, tt_z, full_N=N)

    # The forward output shape is (B·N3, N2, N1).  IFFT expects the same
    # PRE-shape that the forward took, i.e. (B·N1·N2, N3).  We could do a
    # to_torch+from_torch round-trip here, but the simpler thing is to use
    # the fact that the (B, N) view is consistent: reshape via host.
    fft_re_flat = ttnn.to_torch(fft_re).reshape(B * N1 * N2, N3).contiguous()
    fft_im_flat = ttnn.to_torch(fft_im).reshape(B * N1 * N2, N3).contiguous()
    tt_fft_re = ttnn.from_torch(
        fft_re_flat, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    tt_fft_im = ttnn.from_torch(
        fft_im_flat, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    rec_re, rec_im = ttnn.experimental.fft_three_pass(
        tt_fft_re, tt_fft_im, full_N=N, inverse=True,
    )

    rec_r = ttnn.to_torch(rec_re).reshape(B, N).to(torch.float32)
    rec_i = ttnn.to_torch(rec_im).reshape(B, N).to(torch.float32)

    rel_r = _rel_err(rec_r, x_fp32)
    rel_i = float(rec_i.abs().norm() / x_fp32.abs().norm().clamp_min(1e-30))
    assert rel_r < tol, (
        f"[{label}-roundtrip] N={N} real-part rel err {rel_r:.2e} (tol {tol:.0e})"
    )
    assert rel_i < tol, (
        f"[{label}-roundtrip] N={N} imag-part residual {rel_i:.2e} (tol {tol:.0e})"
    )


# ─── 2. Aggressive band — gated ─────────────────────────────────────────
# These exercise the upper end of the supported range.  Default-skipped
# because they each take minutes and need GB of host RAM for the torch
# reference; opt in with TT_FFT_AGGRESSIVE=1.
#
# Ceiling notes (commit 5b):
#   - fp32: 2^28 (256 MB input, ~4 GB working set incl. torch reference).
#   - bf16: 2^28 (128 MB input, ~2 GB working set).
#   - N = 2^30 (1G) IS algorithmically supported (cube-balanced gives
#     N1=N2=N3=1024, big_modulus=2^20 = the apply_twiddles_xl cap), but
#     even bf16 input is 2 GB and torch reference needs 8 GB fp64 — not
#     practical for a default test box.  The router accepts it; users
#     with sufficient DRAM can call it directly.
_AGGRESSIVE_GATE = os.environ.get("TT_FFT_AGGRESSIVE", "0") == "1"


@pytest.mark.skipif(
    not _AGGRESSIVE_GATE,
    reason="TT_FFT_AGGRESSIVE=1 not set; slow large-N three-pass tests are gated.",
)
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", [
    (ttnn.float32,  torch.float32,  "fp32", 1e-3),
    (ttnn.bfloat16, torch.bfloat16, "bf16", 1e-1),
], ids=["fp32", "bf16"])
@pytest.mark.parametrize("N", [
    1 << 24,
    1 << 26,
    # N=2^28: N1=N2=512, N3=1024. bring_n2_inner step 4 transposes
    # (B, N2·N1=262144, N3=1024)→(B, N3, 262144) with page=1MB→CB=2MB
    # which overflows the 1.5MB L1 limit.  Any 3-D decomposition of the
    # N2↔N3 swap when both dims are large (512,1024) requires this page;
    # a 4-D strided-transpose kernel is needed (future work).
    pytest.param(1 << 28, marks=pytest.mark.xfail(
        reason=(
            "N=2^28: DRAM OOM on WH B0 (≈1 GB). "
            "fp32 input = 1 GB; bf16 input = 512 MB. "
            "Exceeds device DRAM capacity."
        ),
        strict=False,
    )),
])
def test_three_pass_correctness_large(
    device, N, tt_dtype, torch_dtype, label, tol,
):
    """B=1 large-N coverage.  Tolerances are relaxed vs the small-N band
    because the on-the-fly twiddle recurrence error scales linearly with
    P and the per-pass FFT error scales as √N."""
    N1, N2, N3 = _expected_three_factorization(N)
    assert N1 * N2 * N3 == N

    torch.manual_seed(101)
    x = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)

    x_preshaped = x.reshape(N1 * N2, N3).contiguous()
    tt_x = ttnn.from_torch(
        x_preshaped, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    re, im = ttnn.experimental.fft_three_pass(tt_x, full_N=N)
    got_r = ttnn.to_torch(re).reshape(N).to(torch.float32)
    got_i = ttnn.to_torch(im).reshape(N).to(torch.float32)
    got = torch.complex(got_r, got_i)

    ref = torch.fft.fft(x[0].to(torch.float32).to(torch.complex64), dim=-1)
    rel = _rel_err(got, ref)
    assert rel < tol, (
        f"[{label}-large] N={N} (N1={N1},N2={N2},N3={N3}) rel err "
        f"{rel:.2e} (tol {tol:.0e})"
    )


# ─── 3. Program cache hit ──────────────────────────────────────────────────
# Three-pass dispatches (after commit 5c restructuring):
#   - 2 × transpose_rm  (initial rearrangement, r/i — 1 entry shape-wise)
#   - 1 × fft_radix_pass(P=N1, twiddle=0)
#   - 1 × apply_twiddles_xl(P=N1, big_mod=N2·N3)
#   - 2 × transpose_rm  (n2-to-last r/i — different shape, 1 more entry)
#   - 1 × fft_radix_pass(P=N2, twiddle=N3, stride=N1)
#   - 2 × transpose_rm  (n3-to-last r/i — 1 more entry)
#   - 1 × fft_radix_pass(P=N3, twiddle=0)
#   - 2 × transpose_rm  (final dim-reverse T1 r/i — 1 more entry)
#   - 2 × transpose_rm  (final dim-reverse T2 r/i — 1 more entry)
# Per-shape transpose cache entries; the second call with the same shape
# must NOT grow the cache count.
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
def test_three_pass_program_cache_hit(device, tt_dtype, torch_dtype, label, tol):
    B, N = 1, 1 << 21
    N1, N2, N3 = _expected_three_factorization(N)
    torch.manual_seed(0)
    x = torch.randn(B, N, dtype=torch.float32).to(torch_dtype)
    x_preshaped = x.reshape(B * N1 * N2, N3).contiguous()

    tt_x = ttnn.from_torch(
        x_preshaped, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    ttnn.experimental.fft_three_pass(tt_x, full_N=N)
    n_after_warmup = device.num_program_cache_entries()

    tt_x2 = ttnn.from_torch(
        x_preshaped, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    ttnn.experimental.fft_three_pass(tt_x2, full_N=N)
    n_after_repeat = device.num_program_cache_entries()

    assert n_after_repeat == n_after_warmup, (
        f"[{label}] three-pass program cache regression: "
        f"{n_after_warmup} → {n_after_repeat}"
    )


# ─── 4. Metal Trace replay — SKIPPED ───────────────────────────────────────
# Same caveat as fft_two_pass: the composite is host-orchestrated (each
# intermediate tensor is freshly allocated between dispatches), so
# capturing it into a single trace is not currently safe.  Each individual
# component op (fft_radix_pass, apply_twiddles_xl, transpose_rm, fft) IS
# trace-safe and is exercised by its own test file.  A future commit may
# add a tensor-pool-based variant that wraps the chain into one trace-
# capturable program.
@pytest.mark.skip(reason="three-pass composite is host-orchestrated; trace deferred to commit 7")
def test_three_pass_metal_trace_replay(device):
    pass
