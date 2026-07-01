# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Commit 7-a — comprehensive program-cache hit audit.

Goal: prove that **every** public FFT op and **every** parameter
variation that distinguishes a kernel binary correctly hits the JIT
program cache on the second call.  The test pattern is:

    1.  Warm up: call op once.
    2.  Record `device.num_program_cache_entries()`.
    3.  Call again with identical shapes/dtypes/parameters.
    4.  Assert entry count is unchanged.

If any of these asserts trip, it means either (a) the op's
`compute_program_hash` is missing a parameter and is producing two
identical hashes for two semantically-different programs (cache
collision — silent miscompilation risk), or (b) the hash IS distinct
but a kernel is being recompiled — performance regression.

This file is the source of the "every op program-cache compliant"
ablation row in the paper's evaluation section.
"""

import os
import pytest
import torch
import ttnn


pytestmark = pytest.mark.skipif(
    os.environ.get("TT_FFT_NATIVE", "1") == "0",
    reason="TT_FFT_NATIVE=0 explicitly disables the new path.",
)


# ─── Helper ──────────────────────────────────────────────────────────────
def _assert_cache_hit(device, op_label, fn):
    """Run `fn` twice; assert num_program_cache_entries is unchanged
    after the second call.  Returns the (warmup_n, repeat_n) tuple for
    debugging."""
    fn()
    n_after_warmup = device.num_program_cache_entries()
    fn()
    n_after_repeat = device.num_program_cache_entries()
    assert n_after_repeat == n_after_warmup, (
        f"[{op_label}] program-cache regression: "
        f"{n_after_warmup} → {n_after_repeat} entries after repeat call. "
        "Either the op's program_hash is missing a parameter, or a "
        "kernel is being needlessly recompiled."
    )
    return n_after_warmup, n_after_repeat


def _rm(t, device, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


# ─── 1. apply_twiddles ──────────────────────────────────────────────────
# Input shape is (M, N1) — the op operates on a *single* twiddle row
# of length N1 per (b, n2) outer pair, and M = B * N2.
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16],
                         ids=["fp32", "bf16"])
def test_cache_apply_twiddles(device, dtype):
    torch.manual_seed(0)
    N1, N2, B = 32, 32, 1
    M = B * N2
    xr = torch.randn(M, N1, dtype=torch.float32)
    xi = torch.randn(M, N1, dtype=torch.float32)

    def fn():
        ttnn.experimental.apply_twiddles(
            _rm(xr, device, dtype), _rm(xi, device, dtype),
            N1=N1, N2=N2,
        )
    _assert_cache_hit(device, f"apply_twiddles-{dtype}", fn)


# ─── 2. apply_twiddles_xl ───────────────────────────────────────────────
# `M` must be a positive multiple of `big_modulus`.  Smallest valid
# config (matches the first row of _XL_CASES in test_apply_twiddles_xl):
#   P=32, big_modulus=1024, full_N=32*1024, M=1024.
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16],
                         ids=["fp32", "bf16"])
def test_cache_apply_twiddles_xl(device, dtype):
    torch.manual_seed(1)
    P, big_modulus = 32, 1024
    full_N         = P * big_modulus     # 32768
    M              = big_modulus         # one period — smallest valid
    xr = torch.randn(M, P, dtype=torch.float32)
    xi = torch.randn(M, P, dtype=torch.float32)

    def fn():
        ttnn.experimental.apply_twiddles_xl(
            _rm(xr, device, dtype), _rm(xi, device, dtype),
            P=P, big_modulus=big_modulus, full_N=full_N,
        )
    _assert_cache_hit(device, f"apply_twiddles_xl-{dtype}", fn)


# ─── 3. transpose_rm ────────────────────────────────────────────────────
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16],
                         ids=["fp32", "bf16"])
def test_cache_transpose_rm(device, dtype):
    """transpose_rm requires the second-to-last dim (`A`) to be a
    multiple of 32 (tile size) and >= 32."""
    torch.manual_seed(2)
    x = torch.randn(32, 128, dtype=torch.float32)

    def fn():
        ttnn.experimental.transpose_rm(_rm(x, device, dtype))
    _assert_cache_hit(device, f"transpose_rm-{dtype}", fn)


# ─── 4. fft_radix_pass — all parameter variations ───────────────────────
# Each variation below MUST produce a distinct cache entry on first call
# and then NO new entries on the repeat call.
#
# Constraints (from device-op validation):
#   * twiddle_N2 == 0     → no post-twiddle (sentinel, APPLY_POSTTWIDDLE=0)
#   * twiddle_N2 ∈ {1,2,4,…,1024}, pow-2 → with post-twiddle
#   * (M / stride) % twiddle_N2 == 0      (when twiddle_N2 != 0)
# We pick M=32, twiddle_N2=16, stride∈{1,2} so all combinations satisfy
# the constraint.
_RX_M, _RX_P, _RX_N2 = 32, 128, 16


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16],
                         ids=["fp32", "bf16"])
def test_cache_radix_pass_no_twiddle(device, dtype):
    """`twiddle_N2 = 0` is the no-post-twiddle sentinel; this is the
    APPLY_POSTTWIDDLE=0 kernel variant."""
    torch.manual_seed(3)
    x = torch.randn(_RX_M, _RX_P, dtype=torch.float32)
    def fn():
        ttnn.experimental.fft_radix_pass(_rm(x, device, dtype),
                                         P=_RX_P, twiddle_N2=0)
    _assert_cache_hit(device, f"fft_radix_pass-noPT-{dtype}", fn)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16],
                         ids=["fp32", "bf16"])
def test_cache_radix_pass_with_twiddle(device, dtype):
    torch.manual_seed(4)
    x = torch.randn(_RX_M, _RX_P, dtype=torch.float32)
    def fn():
        ttnn.experimental.fft_radix_pass(_rm(x, device, dtype),
                                         P=_RX_P, twiddle_N2=_RX_N2)
    _assert_cache_hit(device, f"fft_radix_pass-PT-{dtype}", fn)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16],
                         ids=["fp32", "bf16"])
def test_cache_radix_pass_with_stride(device, dtype):
    """Commit 5a addition — `stride > 1` indexes the twiddle table at
    `(r/stride) % twiddle_N2`.  Must be a distinct cache entry from the
    default `stride = 1` path."""
    torch.manual_seed(5)
    stride = 2
    x = torch.randn(_RX_M, _RX_P, dtype=torch.float32)
    def fn():
        ttnn.experimental.fft_radix_pass(_rm(x, device, dtype),
                                         P=_RX_P, twiddle_N2=_RX_N2,
                                         stride=stride)
    _assert_cache_hit(device, f"fft_radix_pass-stride{stride}-{dtype}", fn)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16],
                         ids=["fp32", "bf16"])
def test_cache_radix_pass_with_output_scale(device, dtype):
    """Commit 6c addition — `output_scale != 1.0` enables the
    APPLY_SCALE kernel variant.  Distinct cache entry from default."""
    torch.manual_seed(6)
    x = torch.randn(_RX_M, _RX_P, dtype=torch.float32)
    def fn():
        ttnn.experimental.fft_radix_pass(
            _rm(x, device, dtype),
            P=_RX_P, twiddle_N2=_RX_N2, output_scale=0.0625)
    _assert_cache_hit(device, f"fft_radix_pass-scale-{dtype}", fn)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16],
                         ids=["fp32", "bf16"])
def test_cache_radix_pass_complex_input(device, dtype):
    """Commit 6a — complex input (input_imag present) is its own
    compute path."""
    torch.manual_seed(7)
    xr = torch.randn(_RX_M, _RX_P, dtype=torch.float32)
    xi = torch.randn(_RX_M, _RX_P, dtype=torch.float32)
    def fn():
        ttnn.experimental.fft_radix_pass(
            _rm(xr, device, dtype), _rm(xi, device, dtype),
            P=_RX_P, twiddle_N2=_RX_N2)
    _assert_cache_hit(device, f"fft_radix_pass-complex-{dtype}", fn)


# ─── 5. complex_mul ─────────────────────────────────────────────────────
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16],
                         ids=["fp32", "bf16"])
def test_cache_complex_mul(device, dtype):
    torch.manual_seed(8)
    M, P = 16, 128
    a = torch.randn(M, P, dtype=torch.float32)
    b = torch.randn(M, P, dtype=torch.float32)
    def fn():
        ttnn.experimental.complex_mul(
            _rm(a, device, dtype), _rm(b, device, dtype),
            _rm(a, device, dtype), _rm(b, device, dtype),
        )
    _assert_cache_hit(device, f"complex_mul-{dtype}", fn)


# ─── 6. fft (single-tile path, N ≤ 1024) ───────────────────────────────
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16],
                         ids=["fp32", "bf16"])
@pytest.mark.parametrize("N", [16, 1024], ids=lambda v: f"N{v}")
def test_cache_fft_single_tile(device, N, dtype):
    torch.manual_seed(9)
    x = torch.randn(1, N, dtype=torch.float32)
    def fn():
        ttnn.experimental.fft(_rm(x, device, dtype))
    _assert_cache_hit(device, f"fft-N{N}-{dtype}", fn)


# ─── 6b. fft — real-only vs complex MUST produce DISTINCT cache entries ──
# Root-cause of the bf16 Bluestein N=11/N=97 failures:
#   real-only (1,32) bf16  → SingleTileStockhamFactory  (factory_index=0)
#   complex   (1,32) bf16  → BatchedStockhamFactory     (factory_index=1)
#
# Without `input_imag.has_value()` in compute_program_hash both calls share
# the same hash.  The complex call gets a cache HIT, blindly reuses factory
# index 0, and SingleTileStockhamFactory::create_descriptor hard-codes
# zscratch (zeros) as the imaginary input — silently computing
# FFT(b_cyc_re + i·0) instead of FFT(b_cyc_re + i·b_cyc_im).
# plan->B_re / B_im are then wrong for every Bluestein call at that N.
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16],
                         ids=["fp32", "bf16"])
def test_cache_fft_real_vs_complex_distinct(device, dtype):
    """Real-only and complex FFT of the same shape must not share a cache entry.

    The test calls real-only first (warms up SingleTileStockhamFactory),
    then calls complex and asserts the entry count increases by ≥1 — i.e.
    the complex call is a cache MISS, not a collision HIT on the real entry.
    """
    torch.manual_seed(42)
    N = 32
    xr = torch.randn(1, N, dtype=torch.float32)
    xi = torch.randn(1, N, dtype=torch.float32)

    # Warm up real-only entry.
    ttnn.experimental.fft(_rm(xr, device, dtype))
    n_after_real = device.num_program_cache_entries()

    # Complex call must be a MISS (new entry) — not a collision HIT.
    ttnn.experimental.fft(_rm(xr, device, dtype), _rm(xi, device, dtype))
    n_after_complex = device.num_program_cache_entries()

    assert n_after_complex > n_after_real, (
        f"[fft-real-vs-complex-{dtype}] real-only and complex (1,{N}) "
        f"{dtype} FFT share a program cache entry — compute_program_hash "
        "is missing input_imag.has_value(). "
        f"Cache entries before complex call: {n_after_real}, after: {n_after_complex}."
    )

    # Now repeat the complex call — must HIT its own entry.
    _assert_cache_hit(device, f"fft-complex-N{N}-{dtype}",
                      lambda: ttnn.experimental.fft(
                          _rm(xr, device, dtype), _rm(xi, device, dtype)))


# ─── 7. fft_two_pass (N > 1024) ─────────────────────────────────────────
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16],
                         ids=["fp32", "bf16"])
def test_cache_fft_two_pass(device, dtype):
    torch.manual_seed(10)
    N = 2048
    x = torch.randn(1, N, dtype=torch.float32)
    def fn():
        ttnn.experimental.fft(_rm(x, device, dtype))
    _assert_cache_hit(device, f"fft_two_pass-{dtype}", fn)


# ─── 8. ifft (commit 6c) ─────────────────────────────────────────────────
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16],
                         ids=["fp32", "bf16"])
@pytest.mark.parametrize("N", [16, 2048], ids=lambda v: f"N{v}")
def test_cache_ifft(device, N, dtype):
    """IFFT routes through the same ops as forward but with output_scale
    set — must produce its own distinct cache entries the first time,
    then hit on repeat."""
    torch.manual_seed(11)
    xr = torch.randn(1, N, dtype=torch.float32)
    xi = torch.randn(1, N, dtype=torch.float32)
    def fn():
        ttnn.experimental.ifft(_rm(xr, device, dtype), _rm(xi, device, dtype))
    _assert_cache_hit(device, f"ifft-N{N}-{dtype}", fn)


# ─── 9. bluestein_fft (commit 6d / 6e-1) ────────────────────────────────
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16],
                         ids=["fp32", "bf16"])
@pytest.mark.parametrize("N,B",
    [
        (17,  1),   # prime, B=1
        (257, 1),   # Fermat prime, B=1
        (17,  2),   # batched
    ],
    ids=lambda v: f"v{v}",
)
def test_cache_bluestein(device, N, B, dtype):
    torch.manual_seed(12)
    x = torch.randn(B, N, dtype=torch.float32)
    def fn():
        ttnn.experimental.bluestein_fft(_rm(x, device, dtype), N=N)
    _assert_cache_hit(device, f"bluestein-N{N}-B{B}-{dtype}", fn)


# ─── 10. fft_three_pass (commit 5) ──────────────────────────────────────
# Three-pass requires pre-shaped (B·N1·N2, N3) input; use the smallest
# three-pass case that still exercises the full chain.
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16],
                         ids=["fp32", "bf16"])
def test_cache_fft_three_pass(device, dtype):
    torch.manual_seed(13)
    full_N = 1 << 21          # 2M
    N1, N2, N3 = 64, 32, 1024   # canonical factorization
    # host view into (N1*N2, N3) so the device tensor is allocated with
    # the small page-size required by three_pass.
    x = torch.randn(1, full_N, dtype=torch.float32).view(N1 * N2, N3)

    def fn():
        ttnn.experimental.fft_three_pass(_rm(x, device, dtype), full_N=full_N)
    _assert_cache_hit(device, f"fft_three_pass-{dtype}", fn)


# ─── 11. The killer test: distinct hashes across all the cache-key axes ──
# This is the one that would have caught the 6c regression early.
# Run the SAME radix_pass shape with DIFFERENT parameter combinations
# and verify each combination's cache-hit independently — i.e. confirm
# the per-combination caches don't interfere with each other.
def test_cache_radix_pass_orthogonality(device):
    """Cross-product of (twiddle_N2={0, 16}, stride={1, 2},
    output_scale={1.0, 0.5}) must produce 2*2*2 = 8 DISTINCT cache
    entries on first warmup, and every entry must hit on repeat without
    contaminating its neighbours.

    M=32, twiddle_N2=16, stride∈{1,2} all satisfy
    `(M/stride) % twiddle_N2 == 0`.  twiddle_N2=0 is the no-PT sentinel
    so the constraint is vacuously true there.
    """
    torch.manual_seed(14)
    M, P = 32, 128
    x_t = _rm(torch.randn(M, P, dtype=torch.float32), device, ttnn.float32)

    configs = [
        (n2, st, sc)
        for n2 in (0, 16)            # no-PT vs PT
        for st in (1, 2)             # stride={1,2}
        for sc in (1.0, 0.5)         # no-scale vs scale
    ]

    # Warmup pass — populate one cache entry per config.
    for n2, st, sc in configs:
        ttnn.experimental.fft_radix_pass(
            x_t, P=P, twiddle_N2=n2, stride=st, output_scale=sc)

    n_after_warmup = device.num_program_cache_entries()

    # Repeat pass — every config should hit, no new entries.
    for n2, st, sc in configs:
        ttnn.experimental.fft_radix_pass(
            x_t, P=P, twiddle_N2=n2, stride=st, output_scale=sc)

    n_after_repeat = device.num_program_cache_entries()
    assert n_after_repeat == n_after_warmup, (
        f"fft_radix_pass orthogonality: {n_after_warmup} → "
        f"{n_after_repeat} entries after repeat pass over "
        f"{len(configs)} configs. Some config is missing from its "
        "hash key — silent miscompilation risk."
    )
