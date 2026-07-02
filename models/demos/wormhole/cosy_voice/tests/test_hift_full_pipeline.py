"""
Test for the full TtHiFTGenerator decode_full pipeline.

This tests the complete vocoder path:
    Host CPU (STFT) -> Device (Neural Network) -> Host CPU (ISTFT) -> Audio

Compares the generated audio waveform against the PyTorch reference.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../ref/CosyVoice")))
from cosyvoice.hifigan.generator import HiFTGenerator

from models.common.utility_functions import comp_pcc
from models.demos.wormhole.cosy_voice.tests.test_hift_generator import extract_hift_state_dict
from models.demos.wormhole.cosy_voice.tt.vocoder.generator import TtHiFTGenerator


@pytest.mark.parametrize("mel_length", [32])
def test_hift_decode_full(device, mel_length):
    """
    Test the full decode pipeline including STFT/ISTFT on host CPU.
    """
    torch.manual_seed(42)
    batch_size = 1
    in_channels = 80
    base_channels = 512
    upsample_rates = [8, 8]
    upsample_kernel_sizes = [16, 16]
    resblock_kernel_sizes = [3, 7, 11]
    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    source_resblock_kernel_sizes = [7, 11]
    source_resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5]]
    istft_params = {"n_fft": 16, "hop_len": 4}

    # Instantiate PyTorch model
    pt_model = HiFTGenerator(
        in_channels=in_channels,
        base_channels=base_channels,
        nb_harmonics=8,
        sampling_rate=22050,
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=upsample_kernel_sizes,
        istft_params=istft_params,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        source_resblock_kernel_sizes=source_resblock_kernel_sizes,
        source_resblock_dilation_sizes=source_resblock_dilation_sizes,
        lrelu_slope=0.1,
        f0_predictor=None,
    )
    pt_model.eval()

    # Create dummy inputs
    x_pt = torch.randn(batch_size, in_channels, mel_length)

    # Source audio length: torch.stft uses center=True by default, which pads
    # n_fft//2 on each side. With center=True: frames = floor(T / hop_len) + 1
    # We need frames = target_stft_frames, so T = (target_stft_frames - 1) * hop_len
    total_upsample = int(np.prod(upsample_rates))
    target_stft_frames = mel_length * total_upsample + 1
    source_audio_len = (target_stft_frames - 1) * istft_params["hop_len"]
    source_audio = torch.randn(batch_size, 1, source_audio_len)

    # Run PyTorch full decode
    with torch.no_grad():
        y_pt = pt_model.decode(x_pt, source_audio)

    # Extract state dict and create TTNN model
    state_dict = extract_hift_state_dict(pt_model)

    tt_model = TtHiFTGenerator(
        device,
        state_dict,
        prefix="",
        in_channels=in_channels,
        base_channels=base_channels,
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=upsample_kernel_sizes,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        source_resblock_kernel_sizes=source_resblock_kernel_sizes,
        source_resblock_dilation_sizes=source_resblock_dilation_sizes,
        istft_params=istft_params,
        lrelu_slope=0.1,
    )

    # Run TTNN full decode
    y_tt = tt_model.decode_full(x_pt, source_audio)

    # Compare audio outputs
    min_len = min(y_pt.shape[-1], y_tt.shape[-1])
    y_pt_trunc = y_pt[:, :min_len]
    y_tt_trunc = y_tt[:, :min_len]

    passing, pcc_msg = comp_pcc(y_pt_trunc, y_tt_trunc, 0.97)
    print(f"Full decode (with ISTFT) PCC: {pcc_msg}")
    assert passing, f"PCC failed: {pcc_msg}"
