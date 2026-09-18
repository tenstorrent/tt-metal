# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""The iSTFT identity used by CosyVoice2's HiFT vocoder, on host and on device.

Two tiers, deliberately separated:

  * The `host` tests need no Tenstorrent device. They check that the algebra is
    right -- that matmul + windowed-OLA + NOLA really does reproduce torch.istft at
    CosyVoice2's n_fft=16, hop_len=4. If these fail, the identity is wrong.
  * The `device` tests need silicon. Most compare against `TtIStft.torch_reference`
    (isolates "the TTNN op behaved wrong" from "the identity is wrong": if the host
    tests pass and these fail, the math is fine and the op is the problem). One,
    `test_device_istft_matches_real_torch_istft`, instead compares straight to
    `torch.istft` -- the strongest claim in this file, with no shared derivation on
    either side to hide a bug in.

No CosyVoice2 checkpoint is available yet in this environment, so unlike a golden-based
PCC test these use synthetic magnitude/phase spanning a wide dynamic range as a stand-in
for real vocoder output. That is a weaker check than a captured golden (see istft.py's
module docstring, "Status") and should be replaced with a golden-based test once
scripts/gen_golden.py-equivalent tooling and a CosyVoice2-0.5B checkpoint exist.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.common.utility_functions import comp_pcc

GATE_FP32 = 0.9999
GATE_BF16 = 0.999


def _wide_dynamic_range_inputs(n_frames: int, bins: int = 9, decades: int = 14, seed: int = 0):
    """Synthetic magnitude/phase spanning `decades` orders of magnitude.

    Stands in for a real vocoder's captured output (1e-13 .. 1e1 in the CosyVoice1
    case) until a CosyVoice2 golden is available. Magnitude is log-uniform over the
    requested span; phase is uniform over [-pi, pi).
    """
    g = torch.Generator().manual_seed(seed)
    log_mag = torch.rand((1, bins, n_frames), generator=g) * decades - (decades - 1)  # ~[1e-(d-1), 1e1]
    mag = 10.0**log_mag
    phase = (torch.rand((1, bins, n_frames), generator=g) * 2 - 1) * torch.pi
    return mag * torch.cos(phase), mag * torch.sin(phase)


# --------------------------------------------------------------------------
# host tier -- no device
# --------------------------------------------------------------------------
def test_periodic_hann_matches_reference():
    """HiFT builds its window with scipy get_window(..., fftbins=True).
    The symmetric variant would divide by N-1 and silently break NOLA."""
    from models.demos.audio.cosyvoice2.tt.hifigan.istft import periodic_hann

    got = torch.from_numpy(periodic_hann(16))
    want = torch.hann_window(16, periodic=True)
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()


def test_istft_basis_inverts_rfft():
    """M = [C | -S] must invert torch.fft.rfft for arbitrary real input.

    This pins the DC/Nyquist weighting, which is the part of the derivation most
    likely to be subtly wrong: bins 0 and 8 are their own conjugates so they are
    not doubled, and they carry no imaginary component.
    """
    from models.demos.audio.cosyvoice2.tt.hifigan.istft import istft_basis

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


@pytest.mark.parametrize("n_frames", [32, 128, 1024, 8192])
def test_host_identity_across_lengths(n_frames):
    """Random spectra at lengths spanning ~0.13 s to ~1.4 s of 24 kHz audio
    (CosyVoice2's sampling_rate, hop_len=4), against torch.istft itself."""
    from models.demos.audio.cosyvoice2.tt.hifigan.istft import TtIStft, periodic_hann

    torch.manual_seed(1234)
    window = torch.from_numpy(periodic_hann(16))
    real = torch.randn(1, 9, n_frames)
    imag = torch.randn(1, 9, n_frames)
    want = torch.istft(torch.complex(real, imag), 16, 4, 16, window=window, return_complex=False)
    got = TtIStft.torch_reference(real, imag, window)
    assert got.shape == want.shape, (got.shape, want.shape)
    passed, pcc = comp_pcc(want, got, GATE_FP32)
    assert passed, pcc


def test_host_identity_survives_wide_dynamic_range_fp32():
    """The identity itself (torch_reference), fp32, over a 14-decade magnitude span."""
    from models.demos.audio.cosyvoice2.tt.hifigan.istft import TtIStft, periodic_hann

    real, imag = _wide_dynamic_range_inputs(n_frames=256)
    window = torch.from_numpy(periodic_hann(16))
    want = torch.istft(torch.complex(real, imag), 16, 4, 16, window=window, return_complex=False)
    got = TtIStft.torch_reference(real, imag, window)
    passed, pcc = comp_pcc(want, got, GATE_FP32)
    print(f"\n  fp32 PCC {pcc}")
    assert passed, pcc


def test_host_identity_survives_bfloat16_inputs():
    """bfloat16 is the dtype the device will actually carry the spectra in."""
    from models.demos.audio.cosyvoice2.tt.hifigan.istft import TtIStft, periodic_hann

    real, imag = _wide_dynamic_range_inputs(n_frames=256)
    window = torch.from_numpy(periodic_hann(16))
    want = torch.istft(torch.complex(real, imag), 16, 4, 16, window=window, return_complex=False)
    got = TtIStft.torch_reference(real.bfloat16().float(), imag.bfloat16().float(), window)
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  bf16 PCC {pcc}")
    assert passed, pcc


# --------------------------------------------------------------------------
# device tier -- needs silicon
# --------------------------------------------------------------------------
# conv_transpose2d allocates from the L1_SMALL bank; the default l1_small_size=0
# fails with "bank size is 0 B".
needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)


@needs_l1_small
@pytest.mark.parametrize("n_frames", [256, 8192])
def test_device_istft_matches_real_torch_istft(device, n_frames):
    """Device output vs. torch.istft directly -- zero inferential steps.

    Every other device test compares against `TtIStft.torch_reference`, which
    re-derives the same matmul/conv_transpose/NOLA identity, so a subtle error in the
    identity itself could pass those tests even with a correct TTNN port (both sides
    would share the bug). This one instead compares straight to PyTorch's own,
    independently implemented `torch.istft` -- the strongest single claim in this
    file, and the pattern later components (flow decoder, LLM) should each have one
    of: device output against the untouched framework/reference function, not just
    against this repo's own re-derivation of it.

    n_frames=8192 (~1.4s of 24kHz audio, audio_len=32768) closes the backlog item
    from the KV-cache audit: this strongest device test -- no shared derivation on
    either side -- was previously only run at n_frames=256, well short of what
    test_host_identity_across_lengths already covers on host (up to 8192) and what
    a real HiFT vocoder call produces for a several-second utterance.
    """
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.istft import TtIStft, periodic_hann

    torch.manual_seed(42)
    window = torch.from_numpy(periodic_hann(16))
    real_t = torch.randn(1, 9, n_frames)
    imag_t = torch.randn(1, 9, n_frames)
    want = torch.istft(torch.complex(real_t, imag_t), 16, 4, 16, window=window, return_complex=False)

    op = TtIStft(device, n_fft=16, hop=4)
    real = ttnn.from_torch(real_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    imag = ttnn.from_torch(imag_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(op(real, imag)).reshape(1, -1).float()

    assert got.shape == want.shape, (got.shape, want.shape)
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device vs. real torch.istft PCC {pcc}")
    assert passed, pcc


@needs_l1_small
@pytest.mark.parametrize("n_frames", [64, 512, 8192])
def test_device_istft_matches_host(device, n_frames):
    """conv_transpose2d at H=1, in_ch=16, k=16, stride=4 -- the shape this test
    exists to de-risk. Compared against the host reference, not a golden, so a failure
    here is unambiguously about TTNN op behaviour, not the identity.

    n_frames=8192 closes the same backlog item test_device_istft_matches_real_torch_istft's
    docstring explains -- this device path (specifically the conv_transpose2d leg) was
    previously only run to n_frames=512."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.istft import TtIStft, periodic_hann

    torch.manual_seed(7)
    real_t = torch.randn(1, 9, n_frames)
    imag_t = torch.randn(1, 9, n_frames)
    want = TtIStft.torch_reference(real_t, imag_t, torch.from_numpy(periodic_hann(16)))

    op = TtIStft(device, n_fft=16, hop=4)
    real = ttnn.from_torch(real_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    imag = ttnn.from_torch(imag_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(op(real, imag)).reshape(1, -1).float()

    assert got.shape == want.shape, (got.shape, want.shape)
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device T={n_frames} PCC {pcc}")
    assert passed, pcc


@needs_l1_small
def test_device_istft_survives_wide_dynamic_range(device):
    """Device path against synthetic magnitude/phase spanning 14 decades -- the
    same span the CosyVoice1 golden covered. Stand-in for a golden test; see the
    module docstring in tests for why there is no golden here yet."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.istft import TtIStft, periodic_hann

    real_t, imag_t = _wide_dynamic_range_inputs(n_frames=256, seed=99)
    want = TtIStft.torch_reference(real_t, imag_t, torch.from_numpy(periodic_hann(16)))

    op = TtIStft(device, n_fft=16, hop=4)
    real = ttnn.from_torch(real_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    imag = ttnn.from_torch(imag_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(op(real, imag)).reshape(1, -1).float()

    assert got.shape == want.shape, (got.shape, want.shape)
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device wide-dynamic-range PCC {pcc}")
    assert passed, pcc
