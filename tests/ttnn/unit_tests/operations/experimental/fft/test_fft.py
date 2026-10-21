# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# PR-gate smoke for ttnn.experimental.fft / ifft on Wormhole B0.
# Full N / prim / cache / trace coverage lives in
# tests/ttnn/nightly/unit_tests/operations/experimental/fft/.

import pytest
import torch

import ttnn


def _rel_err(got_complex, ref_complex):
    return (torch.linalg.norm(got_complex - ref_complex) / torch.linalg.norm(ref_complex).clamp_min(1e-12)).item()


def _from_torch(t, device, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


@pytest.mark.parametrize("N", [1024, 4096])
def test_fft_returns_correct_shape_and_dtype(device, N):
    torch_in = torch.randn(N, dtype=torch.float32)
    tt_in = _from_torch(torch_in, device, ttnn.float32)
    real, imag = ttnn.experimental.fft(tt_in)
    assert real.shape == tt_in.shape, "real spectrum shape must match input"
    assert imag.shape == tt_in.shape, "imag spectrum shape must match input"
    assert real.dtype == ttnn.float32
    assert imag.dtype == ttnn.float32


@pytest.mark.parametrize(
    "N, dtype, tol",
    [
        (256, ttnn.float32, 5e-4),  # Stockham
        (4096, ttnn.float32, 1e-3),  # two-pass
        (7, ttnn.float32, 5e-3),  # Bluestein
        (256, ttnn.bfloat16, 5e-2),
    ],
)
def test_fft_matches_torch(device, N, dtype, tol):
    torch_in = torch.randn(N, dtype=torch.float32)
    tt_in = _from_torch(torch_in, device, dtype)
    tt_re, tt_im = ttnn.experimental.fft(tt_in)
    got = torch.complex(
        ttnn.to_torch(tt_re).reshape(-1).to(torch.float32),
        ttnn.to_torch(tt_im).reshape(-1).to(torch.float32),
    )
    rel = _rel_err(got, torch.fft.fft(torch_in))
    assert rel < tol, f"fft N={N} dtype={dtype} rel err {rel:.2e} exceeds {tol:.0e}"


@pytest.mark.parametrize("N, dtype, tol", [(256, ttnn.float32, 5e-4), (256, ttnn.bfloat16, 3e-2)])
def test_fft_ifft_roundtrip(device, N, dtype, tol):
    torch_in = torch.randn(N, dtype=torch.float32)
    tt_in = _from_torch(torch_in, device, dtype)
    spec_re, spec_im = ttnn.experimental.fft(tt_in)
    rec_re, rec_im = ttnn.experimental.ifft(spec_re, spec_im)
    got = ttnn.to_torch(rec_re).reshape(-1).to(torch.float32)
    err_imag = ttnn.to_torch(rec_im).reshape(-1).abs().max().item()
    rel = (torch.linalg.norm(got - torch_in) / torch.linalg.norm(torch_in)).item()
    assert rel < tol, f"roundtrip N={N} dtype={dtype} rel err {rel:.2e} exceeds {tol:.0e}"
    assert err_imag < tol, f"reconstructed imag should be ~0 (got {err_imag:.2e})"


@pytest.mark.parametrize("dtype, tol", [(ttnn.float32, 5e-4), (ttnn.bfloat16, 1.5e-1)])
def test_cache_hit_complex_fft_uses_fresh_args(device, dtype, tol):
    """Second complex FFT of the same shape must read the second imag buffer."""
    torch.manual_seed(116)
    N = 16
    ar, ai = torch.randn(1, N), torch.randn(1, N)
    br, bi = torch.randn(1, N), torch.randn(1, N)

    ttnn.experimental.fft(_from_torch(ar, device, dtype), _from_torch(ai, device, dtype))
    n_after_warmup = device.num_program_cache_entries()
    re, im = ttnn.experimental.fft(_from_torch(br, device, dtype), _from_torch(bi, device, dtype))
    n_after_hit = device.num_program_cache_entries()
    assert (
        n_after_hit == n_after_warmup
    ), f"expected a cache HIT on the 2nd complex fft; got {n_after_warmup} → {n_after_hit}"

    got = torch.complex(
        ttnn.to_torch(re).reshape(-1).to(torch.float32),
        ttnn.to_torch(im).reshape(-1).to(torch.float32),
    )
    ref = torch.fft.fft(torch.complex(br, bi).reshape(-1).to(torch.complex64))
    rel = _rel_err(got, ref)
    assert rel < tol, f"cache-hit output wrong: rel {rel:.2e} (tol {tol:.0e})"


@pytest.mark.parametrize("B, N", [(3, 1024), (5, 256), (3, 4096), (6, 2048)])
def test_non_pow2_batch_is_auto_padded(device, B, N):
    """Leading-dim product is not a power of two: op must auto-pad to next pow-2,
    run the transform, and slice the padding off so the caller sees shape (B, N)."""
    torch_in = torch.randn(B, N, dtype=torch.float32)
    tt_in = _from_torch(torch_in, device, ttnn.float32)
    real, imag = ttnn.experimental.fft(tt_in)
    assert real.shape == tt_in.shape, f"expected shape {tt_in.shape}, got {real.shape}"
    got = torch.complex(
        ttnn.to_torch(real).to(torch.float32),
        ttnn.to_torch(imag).to(torch.float32),
    )
    rel = _rel_err(got, torch.fft.fft(torch_in, dim=-1))
    assert rel < 2e-3, f"non-pow2 batch B={B} N={N} rel err {rel:.2e}"


def test_rank3_non_pow2_batch_is_auto_padded(device):
    """A rank-3 input (5, 7, 64) has leading-dim product 35, not pow-2. Auto-pad path
    must restore the original rank-3 shape on the way back."""
    torch_in = torch.randn(5, 7, 64, dtype=torch.float32)
    tt_in = _from_torch(torch_in, device, ttnn.float32)
    real, imag = ttnn.experimental.fft(tt_in)
    assert real.shape == tt_in.shape
    got = torch.complex(
        ttnn.to_torch(real).to(torch.float32),
        ttnn.to_torch(imag).to(torch.float32),
    )
    rel = _rel_err(got, torch.fft.fft(torch_in, dim=-1))
    assert rel < 2e-3, f"rank-3 non-pow2 batch rel err {rel:.2e}"


@pytest.mark.parametrize("N", [256, 1024, 4096])
def test_tile_layout_is_auto_converted(device, N):
    """TILE input used to be rejected; now auto-converts to ROW_MAJOR for the FFT
    and converts back on return so the caller gets a TILE output."""
    torch_in = torch.randn(1, N, dtype=torch.float32)
    tt_in = ttnn.from_torch(
        torch_in, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    real, imag = ttnn.experimental.fft(tt_in)
    assert real.layout == ttnn.TILE_LAYOUT, "output layout must match input layout"
    assert imag.layout == ttnn.TILE_LAYOUT
    got = torch.complex(
        ttnn.to_torch(real).reshape(-1).to(torch.float32),
        ttnn.to_torch(imag).reshape(-1).to(torch.float32),
    )
    ref = torch.fft.fft(torch_in.reshape(-1))
    rel = _rel_err(got, ref)
    assert rel < 2e-3, f"TILE auto-convert N={N} rel err {rel:.2e}"


def test_large_unaligned_length_is_rejected(device, expect_error):
    """N=44100 is a common audio length. Still rejected on this tree (fix lands in
    a follow-up PR); the error must say 'multiple of 1024'."""
    tt_in = _from_torch(torch.randn(1, 44100), device, ttnn.float32)
    with expect_error(RuntimeError, "multiple of 1024"):
        ttnn.experimental.fft(tt_in)


def test_rank3_large_n_keeps_input_shape(device):
    """The large-page merge path must restore leading dimensions, not return a flat (B, N)."""
    torch_in = torch.randn(2, 2, 32768, dtype=torch.float32)
    tt_in = _from_torch(torch_in, device, ttnn.float32)
    real, imag = ttnn.experimental.fft(tt_in)
    assert real.shape == tt_in.shape
    assert imag.shape == tt_in.shape
    got = torch.complex(
        ttnn.to_torch(real).to(torch.float32),
        ttnn.to_torch(imag).to(torch.float32),
    )
    rel = _rel_err(got, torch.fft.fft(torch_in, dim=-1))
    assert rel < 1e-3, f"rank-3 fft rel err {rel:.2e}"
