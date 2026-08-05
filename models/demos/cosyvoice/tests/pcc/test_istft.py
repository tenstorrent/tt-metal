# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P1: the iSTFT identity, on host and on device.

Two tiers, deliberately separated:

  * The `host` tests need no Tenstorrent device. They check that the algebra is
    right -- that matmul + windowed-OLA + NOLA really does reproduce torch.istft on
    the tensors the real vocoder emits. If these fail, the identity is wrong.
  * The `device` tests need silicon (or ttsim). They check that TTNN's ops behave as
    the identity assumes -- in particular that conv_transpose2d accepts the awkward
    H=1, in_ch=16, k=16, stride=4 shape. If the host tests pass and these fail, the
    math is fine and the op is the problem.

Keeping them apart is what makes a failure diagnosable rather than just red.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from models.demos.cosyvoice.tt.common import as_torch, load_golden, pcc

GOLDEN = "hift.istft"
GATE_FP32 = 0.9999
GATE_BF16 = 0.999


def _golden_inputs():
    """Magnitude/phase/waveform as the reference produced them, with the same
    clip the reference applies before transforming."""
    g = load_golden(GOLDEN)
    mag = torch.clip(as_torch(g["call0.in_magnitude"]), max=1e2)
    pha = as_torch(g["call0.in_phase"])
    ref = as_torch(g["call0.out_waveform"])
    return mag * torch.cos(pha), mag * torch.sin(pha), ref


def _have_golden() -> bool:
    from models.demos.cosyvoice.tt.common import GOLDEN_DIR

    return os.path.exists(os.path.join(GOLDEN_DIR, f"{GOLDEN}.npz"))


needs_golden = pytest.mark.skipif(not _have_golden(), reason="run scripts/gen_golden.py in the CosyVoice venv first")


# --------------------------------------------------------------------------
# host tier -- no device
# --------------------------------------------------------------------------
def test_periodic_hann_matches_reference():
    """The reference builds its window with scipy get_window(..., fftbins=True).
    The symmetric variant would divide by N-1 and silently break NOLA."""
    from models.demos.cosyvoice.tt.hifigan.istft import periodic_hann

    got = torch.from_numpy(periodic_hann(16))
    want = torch.hann_window(16, periodic=True)
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()


def test_istft_basis_inverts_rfft():
    """M = [C | -S] must invert torch.fft.rfft for arbitrary real input.

    This pins the DC/Nyquist weighting, which is the part of the derivation most
    likely to be subtly wrong: bins 0 and 8 are their own conjugates so they are
    not doubled, and they carry no imaginary component.
    """
    from models.demos.cosyvoice.tt.hifigan.istft import istft_basis

    torch.manual_seed(0)
    x = torch.randn(16, 64, dtype=torch.float64)
    spec = torch.fft.rfft(x, dim=0)  # [9, 64]

    # float64 basis: this checks the *algebra*, so it must not be limited by the
    # basis's own storage precision.
    M64 = torch.from_numpy(istft_basis(16, dtype=np.float64))
    got = M64 @ torch.cat([spec.real, spec.imag], dim=0)
    assert torch.allclose(got, x, atol=1e-12), (got - x).abs().max()

    # float32 basis: what the device actually gets built from. Checked separately
    # with a tolerance that reflects float32, not the derivation.
    M32 = torch.from_numpy(istft_basis(16)).double()
    got32 = M32 @ torch.cat([spec.real, spec.imag], dim=0)
    assert torch.allclose(got32, x, atol=1e-6), (got32 - x).abs().max()


@needs_golden
def test_host_identity_matches_golden_fp32():
    """The whole sequence against the real vocoder's magnitude/phase, fp32."""
    from models.demos.cosyvoice.tt.hifigan.istft import TtIStft, periodic_hann

    real, imag, ref = _golden_inputs()
    got = TtIStft.torch_reference(real, imag, torch.from_numpy(periodic_hann(16)))
    p = pcc(got, ref)
    print(f"\n  fp32 PCC {p:.10f}  max|d| {(got - ref).abs().max():.3e}")
    assert got.shape == ref.shape
    assert p >= GATE_FP32, p


@needs_golden
def test_host_identity_survives_bfloat16_inputs():
    """bfloat16 is the dtype the device will actually carry. The real magnitude
    spans ~14 decades, which costs about two thirds of a nine versus synthetic
    data -- so this gate is checked against real tensors, not random ones."""
    from models.demos.cosyvoice.tt.hifigan.istft import TtIStft, periodic_hann

    real, imag, ref = _golden_inputs()
    got = TtIStft.torch_reference(real.bfloat16().float(), imag.bfloat16().float(), torch.from_numpy(periodic_hann(16)))
    p = pcc(got, ref)
    print(f"\n  bf16 PCC {p:.10f}  max|d| {(got - ref).abs().max():.3e}")
    assert p >= GATE_BF16, p


@pytest.mark.parametrize("n_frames", [32, 128, 1024, 8192])
def test_host_identity_across_lengths(n_frames):
    """Random spectra at lengths spanning ~0.2 s to ~1.5 s of audio, against
    torch.istft itself rather than a captured golden."""
    from models.demos.cosyvoice.tt.hifigan.istft import TtIStft, periodic_hann

    torch.manual_seed(1234)
    window = torch.from_numpy(periodic_hann(16))
    real = torch.randn(1, 9, n_frames)
    imag = torch.randn(1, 9, n_frames)
    want = torch.istft(torch.complex(real, imag), 16, 4, 16, window=window, return_complex=False)
    got = TtIStft.torch_reference(real, imag, window)
    assert got.shape == want.shape, (got.shape, want.shape)
    assert pcc(got, want) >= 0.99999, pcc(got, want)


# --------------------------------------------------------------------------
# device tier -- needs silicon or ttsim
# --------------------------------------------------------------------------
# conv_transpose2d allocates from the L1_SMALL bank; the default l1_small_size=0
# fails with "bank size is 0 B" (CLAUDE.md DD-1). The root conftest's device
# fixture honours device_params, so unlike the DiffusionDrive demo -- which had to
# hardcode it around a shadowing local conftest -- the indirect idiom works here.
needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)


@needs_l1_small
@pytest.mark.parametrize("n_frames", [64, 512])
def test_device_istft_matches_host(device, n_frames):
    """conv_transpose2d at H=1, in_ch=16, k=16, stride=4 -- the shape P1 exists
    to de-risk. Compared against the host reference, not the golden, so a failure
    here is unambiguously about TTNN op behaviour."""
    import ttnn
    from models.demos.cosyvoice.tt.hifigan.istft import TtIStft, periodic_hann

    torch.manual_seed(7)
    real_t = torch.randn(1, 9, n_frames)
    imag_t = torch.randn(1, 9, n_frames)
    want = TtIStft.torch_reference(real_t, imag_t, torch.from_numpy(periodic_hann(16)))

    op = TtIStft(device, n_fft=16, hop=4)
    real = ttnn.from_torch(real_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    imag = ttnn.from_torch(imag_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(op(real, imag)).reshape(1, -1).float()

    p = pcc(got, want)
    print(f"\n  device T={n_frames} PCC {p:.10f}  max|d| {(got - want).abs().max():.3e}")
    assert got.shape == want.shape, (got.shape, want.shape)
    assert p >= GATE_BF16, p


@needs_golden
@needs_l1_small
def test_device_istft_matches_golden(device):
    """End of P1: the same PCC the host achieves, reproduced on silicon against
    the real captured vocoder tensors."""
    import ttnn
    from models.demos.cosyvoice.tt.hifigan.istft import TtIStft

    real_t, imag_t, ref = _golden_inputs()
    op = TtIStft(device, n_fft=16, hop=4)
    real = ttnn.from_torch(real_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    imag = ttnn.from_torch(imag_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(op(real, imag)).reshape(1, -1).float()

    p = pcc(got, ref)
    print(f"\n  device-vs-golden PCC {p:.10f}  max|d| {(got - ref).abs().max():.3e}")
    assert p >= GATE_BF16, p
