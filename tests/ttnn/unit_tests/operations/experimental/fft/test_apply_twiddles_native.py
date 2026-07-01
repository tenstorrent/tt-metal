# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for ttnn.experimental.apply_twiddles — the between-pass elementwise
complex-multiply step of Cooley-Tukey two-pass FFT (commit 3b).

Gated by env var TT_FFT_NATIVE=1 (same as the rest of the new
ProgramDescriptor FFT path).

Reference math:
    out[r, k1] = in[r, k1] * exp(-2*pi*i * (r % N2) * k1 / (N1 * N2))
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


_DTYPES = [
    (ttnn.float32, torch.float32, "fp32", 1e-4),
    (ttnn.bfloat16, torch.bfloat16, "bf16", 5e-2),
]


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).abs().norm() / ref.abs().norm().clamp_min(1e-30))


def _ref_apply_twiddles(
    in_r: torch.Tensor, in_i: torch.Tensor, N1: int, N2: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference: out = in * exp(-2j*pi*(r%N2)*k1/N). Computed in fp32."""
    N = N1 * N2
    M = in_r.shape[0]
    re32 = in_r.to(torch.float32)
    im32 = in_i.to(torch.float32)

    # Broadcast twiddle row (r % N2) across all M rows.
    r = torch.arange(M, dtype=torch.int64) % N2          # (M,)
    k1 = torch.arange(N1, dtype=torch.int64)             # (N1,)
    angle = -2.0 * math.pi * r.unsqueeze(1).to(torch.float64) * \
             k1.unsqueeze(0).to(torch.float64) / float(N)
    tw_r = torch.cos(angle).to(torch.float32)            # (M, N1)
    tw_i = torch.sin(angle).to(torch.float32)            # (M, N1)

    out_r = re32 * tw_r - im32 * tw_i
    out_i = re32 * tw_i + im32 * tw_r
    return out_r, out_i


def _run_apply_twiddles(
    device, in_r: torch.Tensor, in_i: torch.Tensor, N1: int, N2: int, tt_dtype
):
    tt_r = ttnn.from_torch(in_r, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_i = ttnn.from_torch(in_i, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out_r, out_i = ttnn.experimental.apply_twiddles(tt_r, tt_i, N1=N1, N2=N2)
    return (
        ttnn.to_torch(out_r).reshape(in_r.shape).to(torch.float32),
        ttnn.to_torch(out_i).reshape(in_i.shape).to(torch.float32),
    )


# ─── 1. Correctness ─────────────────────────────────────────────────────────
# Sweep a representative set of (N1, N2, B) tuples — enough to hit:
#   - B == 1 (single twiddle row, single core)
#   - B  > 1 (broadcast across batch)
#   - non-trivial N2 wrap (B*N2 > 12 → exercises DRAM-bank-wrap on WH)
#   - N1 in {smallest, mid, max} (covers page-size override paths in
#     reader/writer)
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("B,N2,N1", [
    (1,  1,    2),     # smallest possible
    (1,  4,   32),
    (1, 16,   64),
    (1, 32,  128),
    (1, 32, 1024),     # max N1
    (1, 64,   16),
    (2,  8,  128),     # exercise B>1 broadcast
    (4, 16,   64),     # M = 64 rows → wraps past 12 banks
    (8, 32,  128),     # M = 256, larger multi-core split
])
def test_apply_twiddles_correctness(device, B, N2, N1, tt_dtype, torch_dtype, label, tol):
    torch.manual_seed(3)
    M = B * N2
    # Generate in fp32 then cast so input has well-defined values for both dtypes.
    x_r = torch.randn(M, N1, dtype=torch.float32).to(torch_dtype)
    x_i = torch.randn(M, N1, dtype=torch.float32).to(torch_dtype)

    got_r, got_i = _run_apply_twiddles(device, x_r, x_i, N1, N2, tt_dtype)
    ref_r, ref_i = _ref_apply_twiddles(x_r, x_i, N1, N2)

    got = torch.complex(got_r, got_i)
    ref = torch.complex(ref_r, ref_i)
    for r in range(M):
        rel = _rel_err(got[r], ref[r])
        assert rel < tol, (
            f"[{label}] B={B} N2={N2} N1={N1} row={r} rel err {rel:.2e}"
        )


# ─── 2. Program cache hit ──────────────────────────────────────────────────
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
def test_apply_twiddles_program_cache_hit(device, tt_dtype, torch_dtype, label, tol):
    N1, N2, B = 256, 16, 4
    M = B * N2
    torch.manual_seed(0)
    x_r = torch.randn(M, N1, dtype=torch.float32).to(torch_dtype)
    x_i = torch.randn(M, N1, dtype=torch.float32).to(torch_dtype)

    _run_apply_twiddles(device, x_r, x_i, N1, N2, tt_dtype)
    n_after_warmup = device.num_program_cache_entries()

    _run_apply_twiddles(device, x_r, x_i, N1, N2, tt_dtype)
    n_after_repeat = device.num_program_cache_entries()

    assert n_after_repeat == n_after_warmup, (
        f"[{label}] apply_twiddles program cache regression: "
        f"{n_after_warmup} → {n_after_repeat}"
    )
