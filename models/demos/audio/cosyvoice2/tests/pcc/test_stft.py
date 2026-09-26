# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""`TtStft` (the forward STFT of the NSF excitation, `tt/hifigan/stft.py`) against
`torch.stft` directly.

Regression test for a silent, length-dependent corruption: with
`ttnn.prepare_conv_weights` hoisting the framing conv's weight (the
tenstorrent/tt-metal#55545 defect class), `TtStft` was correct up to 64,000
input samples and returned garbage (PCC ~0.27, gain ~0.03) from 65,536 samples
up -- a sharp 2**16 cutoff, i.e. any utterance longer than ~2.7 s at 24 kHz. The
real 9.3 s test utterance is 222,720 samples. Nothing caught it because every
existing test that reaches `TtStft` stops at 20 mel frames (9,600 samples).

The lengths here straddle that cutoff (64,000 / 65,536) and include the real
production length. fp32, the dtype the real vocoder runs in.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.common.utility_functions import comp_pcc

GATE = 0.9999
needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)


def _excitation_like(n: int, seed: int = 0) -> torch.Tensor:
    """Voiced-looking harmonic signal plus a noise floor, roughly the scale and
    spectral shape of `SineGen2`'s output (rms ~0.03)."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n, dtype=torch.float64)
    f0 = 120.0 + 40.0 * torch.sin(2 * torch.pi * t / 24000.0)  # slowly varying pitch
    phase = 2 * torch.pi * torch.cumsum(f0 / 24000.0, 0)
    x = 0.05 * torch.sin(phase) + 0.02 * torch.sin(2 * phase)
    x = x + 0.003 * torch.randn(n, generator=g, dtype=torch.float64)
    return x.float()


@needs_l1_small
@pytest.mark.parametrize("n_samples", [4800, 64000, 65536, 222720])
def test_device_stft_matches_real_torch_stft(device, n_samples):
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.istft import periodic_hann
    from models.demos.audio.cosyvoice2.tt.hifigan.stft import TtStft

    window = torch.from_numpy(np.asarray(periodic_hann(16))).double()
    x = _excitation_like(n_samples)
    spec = torch.stft(x.double().unsqueeze(0), 16, 4, 16, window=window, return_complex=True)
    want = torch.cat([spec.real, spec.imag], dim=1)  # [1, 18, T]

    op = TtStft(device, n_fft=16, hop=4, window=window.float(), dtype=ttnn.float32)
    x_dev = ttnn.from_torch(x.reshape(1, n_samples, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out, n_frames = op(x_dev, n_samples, 1)
    got = ttnn.to_torch(out).float().reshape(1, 18, -1)

    assert got.shape == want.shape, (got.shape, want.shape)
    passed, pcc = comp_pcc(want.float(), got, GATE)
    # PCC alone would let a uniformly mis-scaled output through; the corruption this
    # guards against had gain ~0.03, so pin the level too.
    gain = float((got.double().reshape(-1) @ want.reshape(-1)) / (want.reshape(-1) @ want.reshape(-1)))
    print(f"\n  device TtStft vs torch.stft (n={n_samples}) PCC {pcc}, gain {gain:.4f}")
    assert passed, pcc
    assert abs(gain - 1.0) < 0.02, gain
