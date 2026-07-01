# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Unified accuracy sweep — ttnn.experimental.fft / ifft — all N ranges.

This is the single canonical test for the PR.  It exercises EVERY routing
branch of the unified C++ router in fft.cpp through ONE public API call
(`ttnn.experimental.fft`), matching the cuFFT usage pattern.

Coverage:
  ┌──────────────────────────────────┬────────────────────────────────────┐
  │ N range                          │ Internal path                      │
  ├──────────────────────────────────┼────────────────────────────────────┤
  │ pow-2, N ≤ 1024                  │ SingleTile/BatchedStockhamFactory  │
  │ pow-2, 1024 < N ≤ 2^20          │ fft_two_pass composite             │
  │ pow-2, 2^20 < N ≤ 2^30          │ fft_three_pass_auto composite      │
  │ non-pow-2, M ≤ 2^20             │ bluestein_dispatch (two-pass inner)│
  │ non-pow-2, 2^20 < M ≤ 2^30      │ bluestein_dispatch (3-pass inner)  │
  │ non-pow-2 large (N%1024==0)      │ bluestein + rebank_rm helpers      │
  └──────────────────────────────────┴────────────────────────────────────┘

Verified hardware limits on Wormhole B0 (L1 = 1,499,136 B):
  Pow-2    fp32:  N ≤ 131,072  (2^17)  twiddle = 1.00 MB
  Pow-2    bf16:  N ≤ 262,144  (2^18)  twiddle = 1.00 MB
  Bluestein fp32: N ≤  64,512  (63×1024, M=2^17)
  Bluestein bf16: N ≤ 130,048 (127×1024, M=2^18)

Aggressive (large N) cases are gated behind TT_FFT_AGGRESSIVE=1 to keep
the default CI run fast.  Run:
    TT_FFT_AGGRESSIVE=1 pytest test_fft_all_n.py -v
to verify the full N envelope.

Tolerances:
  fp32 : 5e-4 (two-stage Bluestein can accumulate ≈1e-5 per op)
  bf16 : 5e-2 (two quantisation steps dominate)
  bf16 Bluestein : 1.5e-1 (three cmul + two FFT stages add rounding)
  large Bluestein fp32: 1e-3, bf16: 1e-1

Known Bluestein limitations (xfail):
  bf16, N=11/97: correlated bf16 rounding errors in the on-device FFTs of
    b_cyc, a_pad, and c produce catastrophically wrong output for these two
    specific prime N values.  N=13/31 (same M) are unaffected.
  fp32, N=509: FIXED — bluestein_M now guarantees ≥8 zero-pad elements,
    routing N=509 to M=2048 (fft_two_pass) instead of the problematic M=1024.
"""

import os
import math
import pytest
import torch
import ttnn

# ─── helpers ────────────────────────────────────────────────────────────────

_AGGRESSIVE = os.environ.get("TT_FFT_AGGRESSIVE", "0") == "1"


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float(
        (got - ref).abs().norm() / ref.abs().norm().clamp_min(1e-30)
    )


def _run_fft(device, x_re: torch.Tensor, tt_dtype, *, N: int, B: int = 1):
    """Upload (B, N) tensor, call ttnn.experimental.fft, return complex torch."""
    tt_x = ttnn.from_torch(
        x_re.reshape(B, N), dtype=tt_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    re, im = ttnn.experimental.fft(tt_x)
    got_r = ttnn.to_torch(re).reshape(B, N).to(torch.float32)
    got_i = ttnn.to_torch(im).reshape(B, N).to(torch.float32)
    return torch.complex(got_r, got_i)


def _run_ifft(device, x_re: torch.Tensor, x_im: torch.Tensor,
              tt_dtype, *, N: int, B: int = 1):
    """Upload (B, N) complex spectrum, call ttnn.experimental.ifft."""
    tt_r = ttnn.from_torch(
        x_re.reshape(B, N), dtype=tt_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    tt_i = ttnn.from_torch(
        x_im.reshape(B, N), dtype=tt_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    re, im = ttnn.experimental.ifft(tt_r, tt_i)
    got_r = ttnn.to_torch(re).reshape(B, N).to(torch.float32)
    got_i = ttnn.to_torch(im).reshape(B, N).to(torch.float32)
    return torch.complex(got_r, got_i)


# ─── dtype fixtures ──────────────────────────────────────────────────────────

_DTYPES_POW2 = [
    (ttnn.float32,  torch.float32,  "fp32", 5e-4),
    (ttnn.bfloat16, torch.bfloat16, "bf16", 5e-2),
]
_DTYPES_BLUESTEIN = [
    (ttnn.float32,  torch.float32,  "fp32", 5e-4),
    (ttnn.bfloat16, torch.bfloat16, "bf16", 1.5e-1),
]


# ════════════════════════════════════════════════════════════════════════════
# 1.  Stockham — pow-2, N ≤ 1024  (SingleTile / BatchedStockhamFactory)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES_POW2,
                         ids=[d[2] for d in _DTYPES_POW2])
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("N", [2, 4, 32, 64, 256, 512, 1024])
def test_stockham_fft(device, N, B, tt_dtype, torch_dtype, label, tol):
    """Stockham pow-2 N ≤ 1024 via SingleTile/BatchedStockhamFactory."""
    torch.manual_seed(N + B)
    x = torch.randn(B, N, dtype=torch.float32).to(torch_dtype)
    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    got = _run_fft(device, x, tt_dtype, N=N, B=B)
    assert _rel_err(got, ref) < tol, \
        f"Stockham N={N} B={B} {label}: rel_err={_rel_err(got, ref):.2e} > tol={tol:.2e}"


@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES_POW2,
                         ids=[d[2] for d in _DTYPES_POW2])
@pytest.mark.parametrize("N", [2, 32, 256, 1024])
def test_stockham_ifft_roundtrip(device, N, tt_dtype, torch_dtype, label, tol):
    """Forward → Inverse roundtrip for Stockham path."""
    torch.manual_seed(N)
    x = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)
    ref = x.to(torch.float32)

    tt_x = ttnn.from_torch(x, dtype=tt_dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    re_fft, im_fft = ttnn.experimental.fft(tt_x)
    got = _run_ifft(device,
                    ttnn.to_torch(re_fft).to(torch.float32),
                    ttnn.to_torch(im_fft).to(torch.float32),
                    tt_dtype, N=N)
    assert _rel_err(got.real, ref) < tol * 4, \
        f"Stockham IFFT roundtrip N={N} {label}: rel_err={_rel_err(got.real, ref):.2e}"


# ════════════════════════════════════════════════════════════════════════════
# 2.  Two-pass — pow-2, 1024 < N ≤ 2^20
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES_POW2,
                         ids=[d[2] for d in _DTYPES_POW2])
@pytest.mark.parametrize("B", [
    1,
    # B=2 doubles CB pressure in fft_radix_pass.  Gate behind AGGRESSIVE to
    # keep the default suite within safe L1 headroom on all N values.
    pytest.param(2, marks=pytest.mark.skipif(not _AGGRESSIVE,
                                             reason="TT_FFT_AGGRESSIVE not set")),
])
@pytest.mark.parametrize("N", [2048, 4096, 8192,
                                # N=2^20: page=4 MB → rebank_rm triggered in fft_two_pass
                                pytest.param(1 << 20, marks=pytest.mark.skipif(
                                    not _AGGRESSIVE, reason="TT_FFT_AGGRESSIVE not set"))])
def test_two_pass_fft(device, N, B, tt_dtype, torch_dtype, label, tol):
    """Two-pass composite pow-2 N in (1024, 1M]; B=1 default; B=2 AGGRESSIVE."""
    torch.manual_seed(N + B)
    x = torch.randn(B, N, dtype=torch.float32).to(torch_dtype)
    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    got = _run_fft(device, x, tt_dtype, N=N, B=B)
    assert _rel_err(got, ref) < tol, \
        f"TwoPass N={N} B={B} {label}: rel_err={_rel_err(got, ref):.2e} > tol={tol:.2e}"


@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES_POW2,
                         ids=[d[2] for d in _DTYPES_POW2])
@pytest.mark.parametrize("N", [2048, 8192])
def test_two_pass_ifft_roundtrip(device, N, tt_dtype, torch_dtype, label, tol):
    """Forward → Inverse roundtrip for two-pass path."""
    torch.manual_seed(N)
    x = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)
    ref = x.to(torch.float32)
    tt_x = ttnn.from_torch(x, dtype=tt_dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    re_fft, im_fft = ttnn.experimental.fft(tt_x)
    got = _run_ifft(device,
                    ttnn.to_torch(re_fft).to(torch.float32),
                    ttnn.to_torch(im_fft).to(torch.float32),
                    tt_dtype, N=N)
    assert _rel_err(got.real, ref) < tol * 4, \
        f"TwoPass IFFT roundtrip N={N} {label}: rel_err={_rel_err(got.real, ref):.2e}"


# ════════════════════════════════════════════════════════════════════════════
# 2b. rebank_rm — page-size-converting DRAM copy (commit 7)
#     Exercises the rebank_rm kernel that fft_two_pass and fft_three_pass_auto
#     call when source page > 128 KB (kRebankThresholdBytes in fft.cpp).
#
#     Threshold rationale: fft_two_pass needs simultaneous CBs for reshape,
#     transpose_rm, and radix_pass tiles.  Empirically, source page ≥ 64 KB
#     pushes combined static L1 allocation past the 1.5 MB Wormhole limit
#     for both fp32 and bf16 (bf16 N=65536 page = 128 KB > 64 KB).
#     → N=65536, fp32/bf16: page > 64 KB → rebank_rm (non-AGGRESSIVE)
#     → N=2^17+:            large pages  → rebank_rm (AGGRESSIVE)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES_POW2,
                         ids=[d[2] for d in _DTYPES_POW2])
@pytest.mark.parametrize("N", [
    # N=65536: fp32 page=256 KB / bf16 page=128 KB — both > 64 KB threshold
    #   → rebank_rm triggered in fft_two_pass; no AGGRESSIVE needed
    1 << 16,
    # N=2^17: page=512 KB → rebank_rm triggered in fft_two_pass
    pytest.param(1 << 17, marks=pytest.mark.skipif(not _AGGRESSIVE,
                                                    reason="TT_FFT_AGGRESSIVE not set")),
    # N=2^20: page=4 MB → rebank_rm triggered in fft_two_pass
    pytest.param(1 << 20, marks=pytest.mark.skipif(not _AGGRESSIVE,
                                                    reason="TT_FFT_AGGRESSIVE not set")),
    # N=2^21: page=8 MB → rebank_rm triggered in fft_three_pass_auto
    pytest.param(1 << 21, marks=pytest.mark.skipif(not _AGGRESSIVE,
                                                    reason="TT_FFT_AGGRESSIVE not set")),
])
def test_rebank_rm(device, N, tt_dtype, torch_dtype, label, tol):
    """
    Exercises the rebank_rm DRAM kernel indirectly via ttnn.experimental.fft.

    For source page > 64 KB (kRebankThresholdBytes in fft.cpp), fft_two_pass /
    fft_three_pass_auto call ttnn::prim::rebank_rm instead of ttnn::reshape to
    convert (B, N) page=N*elem → (B*N/chunk, chunk) page=chunk*elem bytes
    without filling the 1.5 MB Wormhole L1.  A correct DFT result proves
    the rebank copy was lossless AND the subsequent FFT chain was correct.
    """
    torch.manual_seed(N % (1 << 20))
    x = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)
    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    got = _run_fft(device, x, tt_dtype, N=N, B=1)
    err = _rel_err(got, ref)
    assert err < tol, (
        f"rebank_rm path N={N} {label}: rel_err={err:.2e} > tol={tol:.2e}"
    )


# ════════════════════════════════════════════════════════════════════════════
# 3.  Three-pass auto-route — pow-2, 2^20 < N ≤ 2^30
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _AGGRESSIVE, reason="TT_FFT_AGGRESSIVE not set")
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES_POW2,
                         ids=[d[2] for d in _DTYPES_POW2])
@pytest.mark.parametrize("N", [
    1 << 21,
    1 << 24,
    # N=2^27: fp32 input = 512 MB, bf16 input = 256 MB.  Either way the input
    # + merge buffers + output exceed the ~1 GB DRAM on WH B0.  This is a
    # hardware capacity limit, not a software bug.
    pytest.param(1 << 27, marks=pytest.mark.xfail(
        reason=(
            "N=2^27: DRAM OOM on WH B0 (≈1 GB). "
            "fp32 input alone = 512 MB; bf16 intermediate merge tensors "
            "add another ≥256 MB, exhausting device memory."
        ),
        strict=False,
    )),
])
def test_three_pass_fft(device, N, tt_dtype, torch_dtype, label, tol):
    """Three-pass auto-routed composite for very large pow-2 N."""
    torch.manual_seed(N % (1 << 20))
    x = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)
    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    got = _run_fft(device, x, tt_dtype, N=N, B=1)
    assert _rel_err(got, ref) < tol, \
        f"ThreePass N={N} {label}: rel_err={_rel_err(got, ref):.2e} > tol={tol:.2e}"


# ════════════════════════════════════════════════════════════════════════════
# 4.  Bluestein — non-pow-2 N, M ≤ 2^20 (two-pass inner FFTs)
# ════════════════════════════════════════════════════════════════════════════

_BLUESTEIN_SMALL = [
    # primes
    3, 5, 7, 11, 13, 17, 31, 97, 127, 257, 509,
    # composites
    6, 12, 100, 200, 384, 500,
    # just above / below a pow-2 boundary
    33, 63, 65, 128, 129, 255,
]

@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES_BLUESTEIN,
                         ids=[d[2] for d in _DTYPES_BLUESTEIN])
@pytest.mark.parametrize("N", _BLUESTEIN_SMALL)
def test_bluestein_fft_small(device, N, tt_dtype, torch_dtype, label, tol):
    """Bluestein non-pow-2 N with M ≤ 2^20 via ttnn.experimental.fft."""
    torch.manual_seed(N)
    x = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)
    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    got = _run_fft(device, x, tt_dtype, N=N, B=1)
    assert _rel_err(got, ref) < tol, \
        f"Bluestein N={N} {label}: rel_err={_rel_err(got, ref):.2e} > tol={tol:.2e}"


@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES_BLUESTEIN,
                         ids=[d[2] for d in _DTYPES_BLUESTEIN])
@pytest.mark.parametrize("N", [7, 97, 383, 997])
def test_bluestein_ifft_roundtrip(device, N, tt_dtype, torch_dtype, label, tol):
    """Bluestein forward → inverse roundtrip via unified API."""
    torch.manual_seed(N)
    x = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)
    ref = x.to(torch.float32)

    tt_x = ttnn.from_torch(x, dtype=tt_dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    re_fft, im_fft = ttnn.experimental.fft(tt_x)
    got = _run_ifft(device,
                    ttnn.to_torch(re_fft).to(torch.float32),
                    ttnn.to_torch(im_fft).to(torch.float32),
                    tt_dtype, N=N)
    assert _rel_err(got.real, ref) < tol * 4, \
        f"Bluestein IFFT roundtrip N={N} {label}: rel_err={_rel_err(got.real, ref):.2e}"


# ════════════════════════════════════════════════════════════════════════════
# 4b. Large Bluestein — verified hardware limits on Wormhole B0
#
#  For N%1024==0, bluestein_fft uses:
#    • shrink_reshape: concat-to-pow2 + rebank_rm + row-trim (CB ≤ 512 KB)
#    • zero_pad_to_m : ttnn::concat + zeros              (CB ≤ 1 MB)
#    • trim_to_n     : rebank_rm (M IS pow-2) + row-slice + reshape
#
#  Limit is the same L1 twiddle constraint as pow-2:
#    M × 2 × sizeof(dtype) ≤ 1,499,136 B
#      fp32: M ≤ 131,072 (2^17) → max N = 64,512 = 63×1024
#      bf16: M ≤ 262,144 (2^18) → max N = 130,048 = 127×1024
# ════════════════════════════════════════════════════════════════════════════

# Bluestein N values that are multiples of 1024 (required for the large-N
# rebank_rm path) and within the hardware twiddle-table limit.
_BLUESTEIN_LARGE_FP32_BF16 = [
    # M = 2^17 = 131,072:  fp32 twiddle = 1.00 MB ✓,  bf16 twiddle = 0.50 MB ✓
    pytest.param(
        64_512,   # 63 × 1024 — largest non-pow-2 1024-multiple with M=2^17
        marks=pytest.mark.skipif(not _AGGRESSIVE, reason="TT_FFT_AGGRESSIVE not set"),
    ),
]
_BLUESTEIN_LARGE_BF16_ONLY = [
    # M = 2^18 = 262,144:  fp32 twiddle = 2.00 MB ✗,  bf16 twiddle = 1.00 MB ✓
    pytest.param(
        130_048,  # 127 × 1024 — largest non-pow-2 1024-multiple with M=2^18
        marks=pytest.mark.skipif(not _AGGRESSIVE, reason="TT_FFT_AGGRESSIVE not set"),
    ),
]


@pytest.mark.parametrize("N", _BLUESTEIN_LARGE_FP32_BF16)
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", [
    (ttnn.float32,  torch.float32,  "fp32", 1e-3),
    (ttnn.bfloat16, torch.bfloat16, "bf16", 1e-1),
], ids=["fp32", "bf16"])
def test_bluestein_large_m17(device, N, tt_dtype, torch_dtype, label, tol):
    """Large Bluestein N=64,512 (M=2^17): fp32 and bf16 both within L1 limit."""
    torch.manual_seed(N % (1 << 20))
    x = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)
    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    got = _run_fft(device, x, tt_dtype, N=N, B=1)
    err = _rel_err(got, ref)
    assert err < tol, (
        f"Large Bluestein N={N} {label}: rel_err={err:.2e} > tol={tol:.2e}"
    )


@pytest.mark.parametrize("N", _BLUESTEIN_LARGE_BF16_ONLY)
def test_bluestein_large_m18_bf16(device, N):
    """Large Bluestein N=130,048 (M=2^18): bf16-only (fp32 twiddle exceeds L1)."""
    torch.manual_seed(N % (1 << 20))
    x = torch.randn(1, N, dtype=torch.float32).to(torch.bfloat16)
    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    got = _run_fft(device, x, ttnn.bfloat16, N=N, B=1)
    err = _rel_err(got, ref)
    assert err < 1e-1, (
        f"Large Bluestein bf16 N={N}: rel_err={err:.2e} > tol=1e-1"
    )


@pytest.mark.parametrize("N", _BLUESTEIN_LARGE_FP32_BF16)
def test_bluestein_large_ifft_roundtrip(device, N):
    """Large Bluestein forward→inverse roundtrip at the fp32 hardware limit."""
    torch.manual_seed(N % (1 << 20))
    x = torch.randn(1, N, dtype=torch.float32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.float32,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    re_fft, im_fft = ttnn.experimental.fft(tt_x)
    got = _run_ifft(device,
                    ttnn.to_torch(re_fft).to(torch.float32),
                    ttnn.to_torch(im_fft).to(torch.float32),
                    ttnn.float32, N=N)
    err = _rel_err(got.real, x)
    assert err < 2e-3, (
        f"Large Bluestein IFFT roundtrip N={N}: rel_err={err:.2e} > 2e-3"
    )


# ════════════════════════════════════════════════════════════════════════════
# 5.  XL Bluestein — non-pow-2, M > 2^20 (three-pass inner FFTs)
# ════════════════════════════════════════════════════════════════════════════

def _bluestein_M(N: int) -> int:
    v = 2 * N - 1
    p = 1
    while p < v:
        p <<= 1
    return p


_BLUESTEIN_XL = [
    # All values are multiples of 1024 so that complex_mul_safe's 1024-aligned
    # chunked path (complex_mul_chunked) is used instead of the Case-B pad path.
    # Non-1024-aligned N ≥ ~16K (e.g. 524289, 600000, 1000003) would overflow
    # L1 in ttnn::pad (CB ≈ 17×output_page) or ttnn::concat (CB = 2×output_page)
    # and require a new streaming kernel.  All values below exercise the identical
    # XL Bluestein code path (inner fft_three_pass_auto for M = 2^21).
    525_312,     # 513×1024, not pow-2, M = 2^21
    786_432,     # 768×1024 = 3×2^18, not pow-2, M = 2^21
    999_424,     # 976×1024, not pow-2, M = 2^21
]

@pytest.mark.skipif(not _AGGRESSIVE, reason="TT_FFT_AGGRESSIVE not set")
@pytest.mark.parametrize("N", _BLUESTEIN_XL)
def test_bluestein_xl_fft(device, N):
    """XL Bluestein: M > 2^20 — inner FFTs use fft_three_pass_auto."""
    M = _bluestein_M(N)
    assert M > (1 << 20), f"Expected M > 1M for XL case, got M={M}"
    torch.manual_seed(N % (1 << 20))
    x = torch.randn(1, N, dtype=torch.float32)
    ref = torch.fft.fft(x.to(torch.complex64), dim=-1)
    tt_x = ttnn.from_torch(x, dtype=ttnn.float32,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    re, im = ttnn.experimental.fft(tt_x)
    got_r = ttnn.to_torch(re).reshape(1, N).to(torch.float32)
    got_i = ttnn.to_torch(im).reshape(1, N).to(torch.float32)
    got = torch.complex(got_r, got_i)
    err = _rel_err(got, ref)
    assert err < 1e-3, f"XL Bluestein N={N} M={M}: rel_err={err:.2e} > 1e-3"


# ════════════════════════════════════════════════════════════════════════════
# 6.  Program cache hit — second call should NOT recompile
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("N", [256, 4096, 97])
def test_program_cache_hit(device, N):
    """Second call with same (N, dtype) must reuse the cached program."""
    x = torch.randn(1, N, dtype=torch.float32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.float32,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    device.enable_program_cache()
    ttnn.experimental.fft(tt_x)
    num_after_first = device.num_program_cache_entries()
    ttnn.experimental.fft(tt_x)
    num_after_second = device.num_program_cache_entries()
    device.disable_and_clear_program_cache()

    assert num_after_second == num_after_first, (
        f"Program cache grew on second fft call for N={N}: "
        f"{num_after_first} → {num_after_second}"
    )


# ════════════════════════════════════════════════════════════════════════════
# 7.  Single-call sanity: the public API entry point dispatches correctly
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("N", [
    # One value from each routing bucket
    512,       # Stockham
    4096,      # Two-pass
    7,         # Bluestein (prime)
    100,       # Bluestein (composite)
])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16],
                         ids=["fp32", "bf16"])
def test_unified_api_dispatch(device, N, dtype):
    """Smoke: ttnn.experimental.fft(x) produces correct DFT for every bucket."""
    tol = 5e-4 if dtype == ttnn.float32 else (5e-2 if N & (N - 1) == 0 else 1.5e-1)
    torch.manual_seed(N)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    x = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)
    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    got = _run_fft(device, x, dtype, N=N, B=1)
    assert _rel_err(got, ref) < tol, \
        f"Unified API N={N} dtype={dtype}: rel_err={_rel_err(got, ref):.2e} > tol={tol:.2e}"


# ════════════════════════════════════════════════════════════════════════════
# 8.  Routing boundary values
#     Tests the exact N values that sit at or adjacent to routing thresholds
#     to verify the router hands off correctly.
#
#   Stockham  : N ≤ 1024 (pow-2)
#   Two-pass  : 1024 < N ≤ 2²⁰  (pow-2)
#   Three-pass: 2²⁰  < N ≤ 2³⁰  (pow-2, AGGRESSIVE)
#   Bluestein : non-pow-2, M = next_pow2(2N-1)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("N,expected_bucket", [
    # ── Stockham ──
    (2, "stockham"),        # minimum valid N — passes after cache-staleness fix
    (1024, "stockham"),     # maximum Stockham N
    # ── Two-pass ──
    (2048,        "two_pass"),   # minimum two-pass N (1024*2)
    (65536,       "two_pass"),   # mid-range two-pass
    pytest.param(1 << 20, "two_pass",
                 marks=pytest.mark.skipif(not _AGGRESSIVE,
                                          reason="TT_FFT_AGGRESSIVE not set")),
    # ── Three-pass (AGGRESSIVE) ──
    pytest.param(1 << 21, "three_pass",
                 marks=pytest.mark.skipif(not _AGGRESSIVE,
                                          reason="TT_FFT_AGGRESSIVE not set")),
    pytest.param(1 << 27, "three_pass",
                 marks=[
                     pytest.mark.skipif(not _AGGRESSIVE,
                                        reason="TT_FFT_AGGRESSIVE not set"),
                     pytest.mark.xfail(
                         reason=(
                             "N=2^27: DRAM OOM on WH B0 (≈1 GB). "
                             "Input tensor alone is ≥256 MB; intermediates "
                             "exhaust device memory."
                         ),
                         strict=False,
                     ),
                 ]),
    # ── Bluestein: M just at Stockham cap (M=1024) ──
    (383,  "bluestein"),    # M = next_pow2(765) = 1024
    # ── Bluestein: M enters two-pass inner (M=2048) ──
    (997,  "bluestein"),    # M = next_pow2(1993) = 2048
    # ── Bluestein: M enters two-pass inner (M=4096) ──
    (1997, "bluestein"),    # M = next_pow2(3993) = 4096
    # ── Large Bluestein: fp32 hardware limit N=64,512 (M=2^17) (AGGRESSIVE) ──
    pytest.param(64_512, "bluestein_large_m17",
                 marks=pytest.mark.skipif(not _AGGRESSIVE,
                                          reason="TT_FFT_AGGRESSIVE not set")),
    # ── Bluestein: M just above 2^20 → three-pass inner (AGGRESSIVE) ──
    # N=525312 (513×1024) is used instead of 524289 (2^19+1) because
    # non-1024-aligned N with N*elem_bytes > 64KB overflows L1 in ttnn::pad
    # (complex_mul_safe Case B, CB ≈ 17×output_page).
    pytest.param(525_312, "bluestein_xl",
                 marks=pytest.mark.skipif(not _AGGRESSIVE,
                                          reason="TT_FFT_AGGRESSIVE not set")),
], ids=lambda x: f"N={x}" if isinstance(x, int) else x)
def test_routing_boundaries(device, N, expected_bucket):
    """Each routing boundary N produces a correct DFT via ttnn.experimental.fft."""
    if expected_bucket == "bluestein_large_m17":
        tol = 1e-3   # large Bluestein fp32 limit
    elif N & (N - 1) == 0:
        tol = 5e-4   # pow-2 paths
    else:
        tol = 1e-3   # standard Bluestein
    torch.manual_seed(N % (1 << 20))
    x = torch.randn(1, N, dtype=torch.float32)
    ref = torch.fft.fft(x.to(torch.complex64), dim=-1)
    got = _run_fft(device, x, ttnn.float32, N=N, B=1)
    err = _rel_err(got, ref)
    assert err < tol, (
        f"Boundary N={N} ({expected_bucket}): rel_err={err:.2e} > tol={tol:.2e}"
    )


# ════════════════════════════════════════════════════════════════════════════
# 9.  Complex-input FFT (re + im path)
#     Verifies fft(re, im) → correct DFT for all routing buckets.
# ════════════════════════════════════════════════════════════════════════════

def _run_fft_complex(device, x_re, x_im, tt_dtype, *, N, B=1):
    """Upload (B,N) real+imag tensors, call ttnn.experimental.fft(re, im)."""
    tt_r = ttnn.from_torch(x_re.reshape(B, N), dtype=tt_dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_i = ttnn.from_torch(x_im.reshape(B, N), dtype=tt_dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    re, im = ttnn.experimental.fft(tt_r, tt_i)
    got_r = ttnn.to_torch(re).reshape(B, N).to(torch.float32)
    got_i = ttnn.to_torch(im).reshape(B, N).to(torch.float32)
    return torch.complex(got_r, got_i)


@pytest.mark.parametrize("N,label", [
    (256,   "stockham"),
    (4096,  "two_pass"),
    (97,    "bluestein"),
    (997,   "bluestein_m2048"),   # exercises complex_mul_safe
])
@pytest.mark.parametrize("tt_dtype,torch_dtype,dtype_label,tol", _DTYPES_POW2,
                         ids=[d[2] for d in _DTYPES_POW2])
def test_complex_input_fft(device, N, label, tt_dtype, torch_dtype, dtype_label, tol):
    """fft(re, im) for a complex input (re + i*im) via the unified API."""
    torch.manual_seed(N)
    x = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)
    y = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)
    ref = torch.fft.fft(
        torch.complex(x.to(torch.float32), y.to(torch.float32)), dim=-1)
    got = _run_fft_complex(device, x, y, tt_dtype, N=N, B=1)
    err = _rel_err(got, ref)
    # Bluestein complex input has slightly higher rounding; widen tolerance.
    effective_tol = tol * 4 if label.startswith("bluestein") else tol
    assert err < effective_tol, (
        f"Complex FFT N={N} ({label}) {dtype_label}: rel_err={err:.2e} > tol={effective_tol:.2e}"
    )


# ════════════════════════════════════════════════════════════════════════════
# 10. Batched Bluestein (B > 1)
#     Ensures the B-replication in bluestein_host and the batch dimension
#     handling in bluestein_dispatch are correct.
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("N,B", [
    (7,   2),
    (97,  4),
    (383, 2),
    (997, 2),   # M=2048, exercises complex_mul_safe with B=2 → rows=4
])
def test_bluestein_batched(device, N, B):
    """Batched Bluestein: (B, N) input produces per-row correct DFT."""
    torch.manual_seed(N * B)
    x = torch.randn(B, N, dtype=torch.float32)
    ref = torch.fft.fft(x.to(torch.complex64), dim=-1)  # (B, N) batch FFT
    got = _run_fft(device, x, ttnn.float32, N=N, B=B)
    err = _rel_err(got, ref)
    assert err < 1e-3, f"Batched Bluestein N={N} B={B}: rel_err={err:.2e} > 1e-3"


# ════════════════════════════════════════════════════════════════════════════
# 11. Three-pass IFFT roundtrip (AGGRESSIVE)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _AGGRESSIVE, reason="TT_FFT_AGGRESSIVE not set")
@pytest.mark.parametrize("N", [1 << 21, 1 << 24])
def test_three_pass_ifft_roundtrip(device, N):
    """Forward → Inverse roundtrip for the three-pass composite."""
    torch.manual_seed(N % (1 << 20))
    x = torch.randn(1, N, dtype=torch.float32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.float32,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    re_fft, im_fft = ttnn.experimental.fft(tt_x)
    got = _run_ifft(device,
                    ttnn.to_torch(re_fft).to(torch.float32),
                    ttnn.to_torch(im_fft).to(torch.float32),
                    ttnn.float32, N=N)
    err = _rel_err(got.real, x)
    assert err < 5e-4, f"ThreePass IFFT roundtrip N={N}: rel_err={err:.2e} > 5e-4"


# ════════════════════════════════════════════════════════════════════════════
# 12. Metal Trace compatibility
#     Verifies that capturing and replaying a Metal Trace does not alter
#     FFT numerical results.  Exercises the ProgramDescriptor-based paths
#     (SingleTileStockham, fft_radix_pass) which must be trace-safe.
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("N", [256, 4096])
def test_metal_trace(device, N):
    """FFT result is identical when executed inside a Metal Trace replay."""
    torch.manual_seed(N)
    x = torch.randn(1, N, dtype=torch.float32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.float32,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Baseline: un-traced execution
    re_ref, im_ref = ttnn.experimental.fft(tt_x)
    ref = torch.complex(ttnn.to_torch(re_ref).to(torch.float32),
                        ttnn.to_torch(im_ref).to(torch.float32))

    # Capture trace
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    re_t, im_t = ttnn.experimental.fft(tt_x)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    # Replay trace
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    got = torch.complex(ttnn.to_torch(re_t).to(torch.float32),
                        ttnn.to_torch(im_t).to(torch.float32))
    ttnn.release_trace(device, tid)

    err = _rel_err(got, ref)
    assert err < 1e-6, f"Metal Trace N={N}: result differs from baseline: rel_err={err:.2e}"


# ════════════════════════════════════════════════════════════════════════════
# 13. bf16 prime sweep — characterise all primes in the Stockham range
#
# For non-pow-2 N in bf16, Bluestein's algorithm runs the inner FFTs on the
# Stockham kernel (M ≤ 1024).  Certain primes (e.g. N=11, N=97) suffer
# catastrophic rounding error due to worst-case chirp-phase cancellation.
#
# This sweep tests every prime 3 ≤ N ≤ 503  (bluestein_M(N) ≤ 1024)
# to identify the complete set of unstable primes on this hardware.
#
# Run:
#   TT_FFT_PRIME_SWEEP=1 pytest test_fft_all_n.py \
#       -k test_bluestein_bf16_prime_sweep -v --tb=no -q
#
# All N values now pass: the root cause (program-cache collision between
# real-only SingleTileStockhamFactory and complex BatchedStockhamFactory for
# the same shape) is fixed by including input_imag.has_value() in
# FFTDeviceOperation::compute_program_hash.
# ════════════════════════════════════════════════════════════════════════════

_PRIME_SWEEP_ENABLED = os.environ.get("TT_FFT_PRIME_SWEEP", "0") == "1"


def _sieve(limit: int):
    """Return all primes ≤ limit via Sieve of Eratosthenes."""
    is_p = [True] * (limit + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            for j in range(i * i, limit + 1, i):
                is_p[j] = False
    return [i for i in range(2, limit + 1) if is_p[i]]


# All primes 3..503 whose bluestein_M(N) ≤ 1024 (SingleTileStockham range).
# N=509 is excluded: bluestein_M(509)=2048 after the zero-pad fix.
_PRIMES_BF16_SWEEP = [p for p in _sieve(503) if p >= 3]


@pytest.mark.skipif(not _PRIME_SWEEP_ENABLED, reason="TT_FFT_PRIME_SWEEP not set")
@pytest.mark.parametrize("N", _PRIMES_BF16_SWEEP)
def test_bluestein_bf16_prime_sweep(device, N):
    """bf16 Bluestein accuracy for every prime 3 ≤ N ≤ 503 (M ≤ 1024)."""
    torch.manual_seed(N)
    x = torch.randn(1, N, dtype=torch.float32).to(torch.bfloat16)
    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    got = _run_fft(device, x, ttnn.bfloat16, N=N, B=1)
    tol = 1.5e-1  # same as _DTYPES_BLUESTEIN bf16 Bluestein tolerance
    assert _rel_err(got, ref) < tol, \
        f"Prime sweep N={N} bf16: rel_err={_rel_err(got, ref):.2e} > tol={tol:.2e}"
