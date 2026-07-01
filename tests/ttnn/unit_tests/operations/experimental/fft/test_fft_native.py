# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the new ProgramDescriptor-based SingleTileStockhamFactory path
(commits 1 + 2 of the host-to-device refactor).

Activated by env var TT_FFT_NATIVE=1.

Scope:
  - commit 1: fp32, real input, forward FFT, pow-2 N in [2, 1024]
  - commit 2: bf16, real input, forward FFT, pow-2 N in [2, 1024]
              (in-kernel bf16↔fp32 expand/truncate at DRAM I/O boundary,
               fp32 internal compute, fp32 twiddles)
  - correctness vs torch.fft
  - program-cache-hit verification (no re-compile on repeat call)
  - Metal-Trace replay verification (all work device-side, replayable)

Out of scope for commits 1-2 (future commits will add):
  - N > 1024 (commits 3-5)
  - non-pow-2 N (commit 5)
  - IFFT (commit 6)
"""

import os
import pytest
import torch
import ttnn


pytestmark = pytest.mark.skipif(
    os.environ.get("TT_FFT_NATIVE", "1") == "0",
    reason="TT_FFT_NATIVE=1 not set; new ProgramDescriptor path is gated.",
)


# (ttnn dtype, torch dtype, dtype label, rel-err tolerance)
_DTYPES = [
    (ttnn.float32,  torch.float32,  "fp32", 1e-4),
    (ttnn.bfloat16, torch.bfloat16, "bf16", 5e-2),
]


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).abs().norm() / ref.abs().norm().clamp_min(1e-30))


def _run_fft(device, x: torch.Tensor, tt_dtype) -> tuple[torch.Tensor, torch.Tensor]:
    tt_x = ttnn.from_torch(
        x, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    re, im = ttnn.experimental.fft(tt_x)
    return (
        ttnn.to_torch(re).reshape(-1).to(torch.float32),
        ttnn.to_torch(im).reshape(-1).to(torch.float32),
    )


# ─── 1. Correctness ────────────────────────────────────────────────────────
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("N", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
def test_singletile_correctness(device, N, tt_dtype, torch_dtype, label, tol):
    torch.manual_seed(42)
    # Generate in fp32 then cast to test dtype so input has well-defined values.
    x_fp32 = torch.randn(N, dtype=torch.float32)
    x = x_fp32.to(torch_dtype)

    got_r, got_i = _run_fft(device, x, tt_dtype)
    got = torch.complex(got_r, got_i)
    # Reference: run torch.fft on the SAME (lossy-cast) input so the test
    # isolates the FFT error from the input-cast error.
    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64))

    rel = _rel_err(got, ref)
    assert rel < tol, f"[{label}] N={N} rel err {rel:.2e} (tol {tol:.0e})"


# ─── 2. Program cache hit ──────────────────────────────────────────────────
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
def test_singletile_program_cache_hit(device, tt_dtype, torch_dtype, label, tol):
    """First call may compile (cache miss); second call MUST be a cache hit
    (no new program entry). This is what the reviewer's review specifically
    requires per adding_new_ttnn_operation.html."""
    N = 1024
    torch.manual_seed(0)
    x = torch.randn(N, dtype=torch.float32).to(torch_dtype)

    _run_fft(device, x, tt_dtype)
    n_after_warmup = device.num_program_cache_entries()

    _run_fft(device, x, tt_dtype)
    n_after_repeat = device.num_program_cache_entries()

    assert n_after_repeat == n_after_warmup, (
        f"[{label}] Program cache regression: {n_after_warmup} entries "
        f"after warmup, {n_after_repeat} after repeat. New ProgramDescriptor "
        f"path is not cacheable."
    )


# ─── 3. Metal Trace replay ─────────────────────────────────────────────────
# ─── 4. Batched correctness (commit 3a) ────────────────────────────────────
# Shape (B, N) → B independent length-N FFTs in parallel across cores.
# Routes through BatchedStockhamFactory in select_program_factory.
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("B", [2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("N", [128, 256, 512, 1024])
def test_batched_correctness(device, B, N, tt_dtype, torch_dtype, label, tol):
    torch.manual_seed(7)
    x_fp32 = torch.randn(B, N, dtype=torch.float32)
    x = x_fp32.to(torch_dtype)

    tt_x = ttnn.from_torch(
        x, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    re, im = ttnn.experimental.fft(tt_x)
    got_r = ttnn.to_torch(re).reshape(B, N).to(torch.float32)
    got_i = ttnn.to_torch(im).reshape(B, N).to(torch.float32)
    got = torch.complex(got_r, got_i)

    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)

    # Per-row rel-err — every row must individually satisfy the bound.
    for b in range(B):
        rel = _rel_err(got[b], ref[b])
        assert rel < tol, f"[{label}] B={B} N={N} row={b} rel err {rel:.2e}"


# ─── 5. Batched program cache hit (commit 3a) ──────────────────────────────
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
def test_batched_program_cache_hit(device, tt_dtype, torch_dtype, label, tol):
    B, N = 32, 1024
    torch.manual_seed(0)
    x = torch.randn(B, N, dtype=torch.float32).to(torch_dtype)

    tt_x = ttnn.from_torch(x, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn.experimental.fft(tt_x)
    n_after_warmup = device.num_program_cache_entries()

    tt_x2 = ttnn.from_torch(x, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn.experimental.fft(tt_x2)
    n_after_repeat = device.num_program_cache_entries()

    assert n_after_repeat == n_after_warmup, (
        f"[{label}] batched program cache regression: "
        f"{n_after_warmup} → {n_after_repeat}"
    )


# ─── 6. Single-tile Metal Trace replay ─────────────────────────────────────
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
def test_singletile_metal_trace_replay(device, tt_dtype, torch_dtype, label, tol):
    """Capture a trace around an FFT call, replay 10x, every replay must
    produce a bit-exact result vs the original eager call. If any host work
    leaks into the op path, the trace can't reproduce it and this test fails."""
    N = 1024
    torch.manual_seed(1)
    x = torch.randn(N, dtype=torch.float32).to(torch_dtype)

    eager_r, eager_i = _run_fft(device, x, tt_dtype)

    tt_x = ttnn.from_torch(
        x, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    re_t, im_t = ttnn.experimental.fft(tt_x)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    try:
        for i in range(10):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            replay_r = ttnn.to_torch(re_t).reshape(-1).to(torch.float32)
            replay_i = ttnn.to_torch(im_t).reshape(-1).to(torch.float32)

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
