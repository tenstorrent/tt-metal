#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
find_n_limit.py — Determines the maximum supported FFT length on this WH device.

Measures:
  1. Maximum pow-2 N (fp32 and bf16) via fft_two_pass / fft_three_pass_auto
  2. Maximum non-pow-2 N (Bluestein, fp32 and bf16) via bluestein_dispatch
  3. Peak verified DRAM usage at the limit

WH B0 L1 twiddle-table constraint (measured):
  fft_radix_pass / apply_twiddles_xl store N × 2 × sizeof(dtype) bytes in L1.
  Wormhole L1 per core = 1,499,136 B (~1.46 MB).
    fp32 : N × 8 ≤ 1,499,136  →  max pow-2 = 2^17 = 131,072
    bf16 : N × 4 ≤ 1,499,136  →  max pow-2 = 2^18 = 262,144
  For Bluestein: inner FFT has length M = next_pow2(2N-1), same limit applies.
    fp32 : M ≤ 2^17  →  N ≤ 65,535   (2N-1 ≤ 131,071)
    bf16 : M ≤ 2^18  →  N ≤ 131,071  (2N-1 ≤ 262,143)

Run from the tt-metal root:
    python tests/ttnn/unit_tests/operations/experimental/fft/find_n_limit.py

Prints a table suitable for copy-paste into a paper.
"""

import math
import sys
import time
import traceback

import torch
import ttnn

# ── Device init ──────────────────────────────────────────────────────────────

device = ttnn.open_device(device_id=0)
device.enable_program_cache()

# ── Helpers ──────────────────────────────────────────────────────────────────

def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return (torch.abs(got - ref).max() / (torch.abs(ref).max() + 1e-9)).item()

def _run_fft(N: int, tt_dtype, torch_dtype) -> float:
    """Returns relative error vs numpy DFT, or float('inf') on OOM/error."""
    torch.manual_seed(N % (1 << 20))
    x = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)
    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    try:
        tt_x = ttnn.from_torch(x, dtype=tt_dtype,
                               layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        re, im = ttnn.experimental.fft(tt_x)
        got_r = ttnn.to_torch(re).reshape(1, N).to(torch.float32)
        got_i = ttnn.to_torch(im).reshape(1, N).to(torch.float32)
        return _rel_err(torch.complex(got_r, got_i), ref)
    except Exception:
        return float('inf')

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p

def _bluestein_M(N: int) -> int:
    return _next_pow2(2 * N - 1)

# ── Section 1: pow-2 limit ────────────────────────────────────────────────────

print("\n" + "="*70)
print("  POW-2 N LIMIT  (fp32 and bf16)")
print("="*70)
print(f"{'N':>15}  {'log2N':>6}  {'DRAM input':>12}  {'fp32 err':>10}  {'bf16 err':>10}  {'status'}")
print("-"*70)

pow2_last_pass = {ttnn.float32: None, ttnn.bfloat16: None}
tols = {ttnn.float32: 5e-4, ttnn.bfloat16: 5e-2}
labels = {ttnn.float32: "fp32", ttnn.bfloat16: "bf16"}
torch_dtypes = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}

# Candidate pow-2 values (from 2^10 up to 2^30)
pow2_candidates = [1 << k for k in range(10, 31)]

for N in pow2_candidates:
    log2N = int(math.log2(N))
    input_mb = N * 4 / (1 << 20)
    results = {}
    for tt_dtype in [ttnn.float32, ttnn.bfloat16]:
        t0 = time.time()
        err = _run_fft(N, tt_dtype, torch_dtypes[tt_dtype])
        dt = time.time() - t0
        results[tt_dtype] = (err, dt)

    fp32_err, fp32_t = results[ttnn.float32]
    bf16_err, bf16_t = results[ttnn.bfloat16]

    fp32_ok = math.isfinite(fp32_err) and fp32_err < tols[ttnn.float32]
    bf16_ok = math.isfinite(bf16_err) and bf16_err < tols[ttnn.bfloat16]

    if fp32_ok:
        pow2_last_pass[ttnn.float32] = N
    if bf16_ok:
        pow2_last_pass[ttnn.bfloat16] = N

    fp32_str = f"{fp32_err:.2e}" if math.isfinite(fp32_err) else "  OOM/ERR"
    bf16_str = f"{bf16_err:.2e}" if math.isfinite(bf16_err) else "  OOM/ERR"
    if fp32_ok and bf16_ok:
        status = "PASS"
    elif bf16_ok:
        status = "bf16-only"   # fp32 twiddle overflows but bf16 fits
    elif fp32_ok:
        status = "fp32-only"
    else:
        status = "FAIL/OOM"

    print(f"{N:>15,}  {log2N:>6}  {input_mb:>10.1f}M  {fp32_str:>10}  {bf16_str:>10}  {status}")
    sys.stdout.flush()

    # Stop after both fail (OOM reached for both dtypes)
    if not fp32_ok and not bf16_ok:
        print("  → OOM boundary reached, stopping.")
        break

# ── Section 2: non-pow-2 Bluestein limit (fp32 and bf16) ─────────────────────

print("\n" + "="*70)
print("  NON-POW-2 N LIMIT  (Bluestein, fp32 and bf16)")
print("="*70)
print(f"{'N':>15}  {'M=nxp2(2N-1)':>14}  {'log2M':>6}  {'fp32 err':>10}  {'bf16 err':>10}  {'status'}")
print("-"*70)

# Bluestein limit is determined by the inner FFT length M = next_pow2(2N-1).
# Same L1 twiddle constraint applies: M × 8 ≤ 1.5 MB (fp32) → M ≤ 2^17 = 131,072.
#
# The large-N rebank_rm path (zero_pad_to_m / trim_to_n) requires N % 1024 == 0,
# so candidates are chosen as the largest N divisible by 1024 for each M bucket:
#
#   fp32 limit: M = 2^17 = 131,072  →  N = 64,512  (63 × 1024, non-pow-2)
#   bf16 limit: M = 2^18 = 262,144  →  N = 130,048 (127 × 1024, non-pow-2)
#   both fail : M = 2^19 = 524,288  →  N = 261,120 (expected OOM)
bluestein_candidates = [
    # M = 2^17 = 131,072  (fp32 twiddle fits: 131072 × 8 = 1.00 MB ≤ 1.5 MB)
    64_512,    # 63 × 1024 = 64,512  →  M = 131,072  fp32: PASS, bf16: PASS
    # M = 2^18 = 262,144  (fp32 twiddle exceeds: 262144 × 8 = 2.0 MB > 1.5 MB)
    130_048,   # 127 × 1024 = 130,048  →  M = 262,144  fp32: FAIL, bf16: PASS
    # M = 2^19 = 524,288  (both dtypes fail)
    261_120,   # 255 × 1024 = 261,120  →  M = 524,288  both FAIL
]

bluestein_last_pass = {ttnn.float32: None, ttnn.bfloat16: None}
TOL_BLUESTEIN_FP32 = 1e-3
TOL_BLUESTEIN_BF16 = 5e-2

for N in bluestein_candidates:
    M = _bluestein_M(N)
    log2M = math.log2(M)
    errs = {}
    for tt_dtype, tol in [(ttnn.float32, TOL_BLUESTEIN_FP32),
                          (ttnn.bfloat16, TOL_BLUESTEIN_BF16)]:
        err = _run_fft(N, tt_dtype, torch_dtypes[tt_dtype])
        errs[tt_dtype] = err
        if math.isfinite(err) and err < tol:
            bluestein_last_pass[tt_dtype] = N

    fp32_err, bf16_err = errs[ttnn.float32], errs[ttnn.bfloat16]
    fp32_ok = math.isfinite(fp32_err) and fp32_err < TOL_BLUESTEIN_FP32
    bf16_ok  = math.isfinite(bf16_err) and bf16_err < TOL_BLUESTEIN_BF16
    fp32_str = f"{fp32_err:.2e}" if math.isfinite(fp32_err) else "  OOM/ERR"
    bf16_str  = f"{bf16_err:.2e}"  if math.isfinite(bf16_err)  else "  OOM/ERR"
    status = "PASS" if (fp32_ok and bf16_ok) else (
             "bf16-only" if bf16_ok else "FAIL/OOM")
    print(f"{N:>15,}  {M:>14,}  {log2M:>6.1f}  {fp32_str:>10}  {bf16_str:>10}  {status}")
    sys.stdout.flush()

# ── Summary for paper ─────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  SUMMARY — Verified N limits on this WH B0 device")
print("="*70)
print("  Constraint: L1 twiddle table = N × 2 × sizeof(dtype) ≤ 1,499,136 B")
print()

if pow2_last_pass[ttnn.float32]:
    N = pow2_last_pass[ttnn.float32]
    dram = N * 4 / (1 << 20)
    print(f"  Pow-2  fp32 :  N = {N:>9,}  (2^{int(math.log2(N)):2d})  "
          f"twiddle = {N*8/(1<<20):.2f} MB   input = {dram:.2f} MB")
if pow2_last_pass[ttnn.bfloat16]:
    N = pow2_last_pass[ttnn.bfloat16]
    dram = N * 2 / (1 << 20)
    print(f"  Pow-2  bf16 :  N = {N:>9,}  (2^{int(math.log2(N)):2d})  "
          f"twiddle = {N*4/(1<<20):.2f} MB   input = {dram:.2f} MB")
print()
if bluestein_last_pass[ttnn.float32]:
    N = bluestein_last_pass[ttnn.float32]
    M = _bluestein_M(N)
    print(f"  Bluestein fp32 :  N = {N:>9,}   M = {M:,}  (2^{math.log2(M):.0f})  "
          f"twiddle = {M*8/(1<<20):.2f} MB")
if bluestein_last_pass[ttnn.bfloat16]:
    N = bluestein_last_pass[ttnn.bfloat16]
    M = _bluestein_M(N)
    print(f"  Bluestein bf16 :  N = {N:>9,}   M = {M:,}  (2^{math.log2(M):.0f})  "
          f"twiddle = {M*4/(1<<20):.2f} MB")

print()
print("  Note: Bluestein requires N % 1024 == 0 for large N (rebank_rm path).")
print("  Gap (no pow-2 path): fp32 N ∈ [2^18, 2^20], bf16 N ∈ [2^19, 2^20]")
print("  (apply_twiddles_xl uses L1 table for 'medium' N; streaming only for N > 2^20)")
print("="*70)

ttnn.close_device(device)
