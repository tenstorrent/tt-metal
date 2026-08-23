# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.tt_dit.models.audio_vae.istft_ttnn import ISTFT


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 64 * 1024}],
    indirect=True,
)
def test_istft_matches_torch(device):
    torch.manual_seed(0)

    n_fft = 16
    hop_length = 4
    num_frames = 20
    n_freqs = n_fft // 2 + 1

    real = torch.randn(1, num_frames, n_freqs)
    imag = torch.randn(1, num_frames, n_freqs)

    reference = torch.istft(
        torch.complex(real, imag).transpose(1, 2),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=torch.hann_window(n_fft),
        center=True,
    )

    istft = ISTFT(
        n_fft=n_fft,
        hop_length=hop_length,
        mesh_device=device,
        dtype=ttnn.bfloat16,
    )

    state = {}
    istft._prepare_torch_state(state)
    istft.load_torch_state_dict(state)

    real_tt = ttnn.from_torch(
        real,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    imag_tt = ttnn.from_torch(
        imag,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    output_tt = istft(real_tt, imag_tt)
    output = ttnn.to_torch(output_tt).float().reshape(reference.shape)

    error = (output - reference).abs()

    pcc = torch.corrcoef(torch.stack([output.flatten(), reference.flatten()]))[0, 1].item()

    assert pcc > 0.999
    assert error.mean().item() < 0.02
