# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the standalone ttnn.experimental.complex_mul device op (commit 6b).

Semantics (ROW_MAJOR elementwise complex multiply of two same-shape
complex tensors A = (a_real, a_imag) and B = (b_real, b_imag)):

    out_real = a_real * b_real  -  a_imag * b_imag
    out_imag = a_real * b_imag  +  a_imag * b_real

Single device dispatch — building block for:
  - Bluestein composite (commit 6d): chirp pre/post + H multiply
  - IFFT inverse path (commit 6c): conjugate-and-scale via length-1
    broadcast of (1/N, -1/N)

Coverage:
  - correctness vs torch complex multiply, fp32 + bf16,
    several (M, P) shapes (including non-pow-2 M)
  - program-cache hit on repeat
  - Metal-Trace replay (single device dispatch, so trace-safe)
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
    (ttnn.float32,  torch.float32,  "fp32", 5e-4),
    (ttnn.bfloat16, torch.bfloat16, "bf16", 5e-2),
]


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).abs().norm() / ref.abs().norm().clamp_min(1e-30))


def _run(device, a_re, a_im, b_re, b_im, tt_dtype):
    tt_ar = ttnn.from_torch(a_re, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_ai = ttnn.from_torch(a_im, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_br = ttnn.from_torch(b_re, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_bi = ttnn.from_torch(b_im, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out_r, out_i = ttnn.experimental.complex_mul(tt_ar, tt_ai, tt_br, tt_bi)
    return ttnn.to_torch(out_r), ttnn.to_torch(out_i)


# ─── 1. Correctness ────────────────────────────────────────────────────────
# Shapes chosen to exercise:
#   - (M=1, P=64): smallest case, single tile, single core.
#   - (M=8, P=32): tiny but multi-row (per-row dispatch path).
#   - (M=128, P=1024): full-tile row (P at the kernel cap).
#   - (M=37, P=64): non-pow-2 M to confirm the core-picker handles it.
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("M,P", [(1, 64), (8, 32), (128, 1024), (37, 64)],
                         ids=[f"M{m}_P{p}" for m, p in [(1, 64), (8, 32), (128, 1024), (37, 64)]])
def test_complex_mul_correctness(device, M, P, tt_dtype, torch_dtype, label, tol):
    torch.manual_seed(11)
    a_re = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)
    a_im = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)
    b_re = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)
    b_im = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)

    got_r, got_i = _run(device, a_re, a_im, b_re, b_im, tt_dtype)
    got = torch.complex(got_r.to(torch.float32), got_i.to(torch.float32))

    a = torch.complex(a_re.to(torch.float32), a_im.to(torch.float32)).to(torch.complex64)
    b = torch.complex(b_re.to(torch.float32), b_im.to(torch.float32)).to(torch.complex64)
    ref = a * b

    rel = _rel_err(got, ref)
    assert rel < tol, (
        f"[{label}] M={M} P={P} aggregate rel err {rel:.2e} (tol {tol:.0e})"
    )


# ─── 2. Program cache hit ──────────────────────────────────────────────────
# complex_mul is a single device op — exactly ONE new program-cache entry
# per (dtype, shape, memory_config) tuple.  Repeat call with the same
# shape/dtype must not grow the cache.
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
def test_complex_mul_program_cache_hit(device, tt_dtype, torch_dtype, label, tol):
    M, P = 16, 128
    torch.manual_seed(0)
    a_re = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)
    a_im = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)
    b_re = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)
    b_im = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)

    _run(device, a_re, a_im, b_re, b_im, tt_dtype)
    n_after_warmup = device.num_program_cache_entries()

    _run(device, a_re, a_im, b_re, b_im, tt_dtype)
    n_after_repeat = device.num_program_cache_entries()

    assert n_after_repeat == n_after_warmup, (
        f"[{label}] complex_mul program cache regression: "
        f"{n_after_warmup} → {n_after_repeat}"
    )


# ─── 3. Metal Trace replay ─────────────────────────────────────────────────
# complex_mul is a single device dispatch (no host-orchestrated reshape /
# transpose chain) so it IS trace-safe.  Verifies the captured program
# replays bit-identically to the eager invocation.
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
def test_complex_mul_metal_trace_replay(device, tt_dtype, torch_dtype, label, tol):
    M, P = 8, 64
    torch.manual_seed(31)
    a_re = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)
    a_im = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)
    b_re = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)
    b_im = torch.randn(M, P, dtype=torch.float32).to(torch_dtype)

    # Eager reference.
    eager_r, eager_i = _run(device, a_re, a_im, b_re, b_im, tt_dtype)

    # Warm program cache + bind addresses.
    tt_ar = ttnn.from_torch(a_re, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_ai = ttnn.from_torch(a_im, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_br = ttnn.from_torch(b_re, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_bi = ttnn.from_torch(b_im, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn.experimental.complex_mul(tt_ar, tt_ai, tt_br, tt_bi)

    # Capture into trace.
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    tr_r, tr_i = ttnn.experimental.complex_mul(tt_ar, tt_ai, tt_br, tt_bi)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    # Replay and compare.
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    tr_r_t = ttnn.to_torch(tr_r)
    tr_i_t = ttnn.to_torch(tr_i)

    assert torch.equal(tr_r_t, eager_r), (
        f"[{label}] complex_mul trace replay real mismatch"
    )
    assert torch.equal(tr_i_t, eager_i), (
        f"[{label}] complex_mul trace replay imag mismatch"
    )

    ttnn.release_trace(device, tid)
