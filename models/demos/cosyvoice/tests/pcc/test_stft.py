# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Forward STFT of the NSF excitation, and the round trip through the inverse.

The round-trip test is the strongest single check on the whole frequency-domain
path: if either basis has a sign, scale or Hermitian-weight error, forward
followed by inverse stops reconstructing the input, even when each direction
looks individually plausible.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from models.demos.cosyvoice.tt.common import pcc
from models.demos.cosyvoice.tt.hifigan.istft import TtIStft, periodic_hann
from models.demos.cosyvoice.tt.hifigan.stft import TtStft, stft_basis

needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)


# --------------------------------------------------------------------------
# host tier
# --------------------------------------------------------------------------
@pytest.mark.parametrize("length", [256, 4096, 72192])
def test_host_stft_matches_torch(length):
    """72192 is the real excitation length for 3.3 s of audio, so the longest
    case here is not synthetic -- it is the shape HiFT actually transforms."""
    torch.manual_seed(0)
    w = torch.from_numpy(periodic_hann(16))
    x = torch.randn(1, length)
    spec = torch.stft(x, 16, 4, 16, window=w, return_complex=True)  # center=True by default
    want = torch.cat([spec.real, spec.imag], dim=1)
    got = TtStft.torch_reference(x, w)

    assert got.shape == want.shape, (got.shape, want.shape)
    p = pcc(got, want)
    print(f"\n  L={length} PCC {p:.10f}  max|d| {(got - want).abs().max():.3e}")
    assert p >= 0.9999, p


def test_stft_and_istft_round_trip():
    """forward -> inverse must return the original signal."""
    torch.manual_seed(1)
    w = torch.from_numpy(periodic_hann(16))
    x = torch.randn(1, 4096)
    s = TtStft.torch_reference(x, w)
    rec = TtIStft.torch_reference(s[:, :9], s[:, 9:], w)

    assert rec.shape == x.shape, (rec.shape, x.shape)
    p = pcc(rec, x)
    print(f"\n  round-trip PCC {p:.10f}  max|d| {(rec - x).abs().max():.3e}")
    assert p >= 0.99999, p


def test_forward_basis_is_not_the_inverse_transposed():
    """A tempting shortcut that is wrong: the inverse folds Hermitian doubling
    (2x on bins 1..N/2-1) and a 1/N scale into its matrix; the forward does
    neither. Asserted so nobody "simplifies" one into the other."""
    from models.demos.cosyvoice.tt.hifigan.istft import istft_basis

    fwd = stft_basis(16)  # [18, 16]
    inv = istft_basis(16)  # [16, 18]
    assert fwd.shape == (18, 16) and inv.shape == (16, 18)
    assert not np.allclose(fwd, inv.T, atol=1e-3), "bases became transposes -- Hermitian weights lost"


# --------------------------------------------------------------------------
# device tier
# --------------------------------------------------------------------------
@needs_l1_small
@pytest.mark.parametrize("length", [256, 1024])
def test_device_stft_matches_host(device, length):
    """Reflect-pad + framing conv1d + DFT matmul, all on device."""
    import ttnn

    torch.manual_seed(2)
    w = torch.from_numpy(periodic_hann(16))
    x_t = torch.randn(1, length)
    want = TtStft.torch_reference(x_t, w)

    op = TtStft(device, n_fft=16, hop=4)
    x = ttnn.from_torch(x_t.reshape(1, length, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got, n_frames = op(x, length)
    got = ttnn.to_torch(got).float()

    print(f"\n  device stft L={length} -> {n_frames} frames, PCC {pcc(got, want):.8f}")
    assert n_frames == op.n_frames(length) == want.shape[-1], (n_frames, want.shape)
    assert pcc(got, want) >= 0.99, pcc(got, want)
