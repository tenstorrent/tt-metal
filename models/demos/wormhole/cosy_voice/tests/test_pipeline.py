# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Integration test for the CosyVoice3 end-to-end TTS pipeline.

Tests the pipeline stages independently and together:
1. LLM generates speech tokens from text
2. Flow decoder converts speech tokens to mel spectrogram
3. (Future) Vocoder converts mel to audio waveform

Uses shared fixtures from conftest.py (mesh_device, flow_state_dict).
"""

import pytest
import torch

import ttnn
from models.demos.wormhole.cosy_voice.tt.flow.flow import TtCausalMaskedDiffWithDiT


def test_flow_decoder_with_dummy_tokens(device, flow_state_dict):
    """
    Test flow decoder stage in isolation with dummy speech tokens.

    This bypasses the LLM and directly tests:
    speech_tokens → flow decoder → mel spectrogram
    """
    flow = TtCausalMaskedDiffWithDiT(device, flow_state_dict, dtype=ttnn.bfloat16)

    # Create dummy inputs
    torch.manual_seed(42)
    prompt_tokens = torch.randint(0, 6561, (1, 5))
    target_tokens = torch.randint(0, 6561, (1, 10))
    prompt_feat = torch.randn(1, 10, 80)  # 5 tokens × 2 ratio = 10 mel frames
    embedding = torch.randn(1, 192)

    # Run flow decoder
    mel, _ = flow.inference(
        token=target_tokens,
        token_len=torch.tensor([10]),
        prompt_token=prompt_tokens,
        prompt_token_len=torch.tensor([5]),
        prompt_feat=prompt_feat,
        prompt_feat_len=torch.tensor([10]),
        embedding=embedding,
    )

    assert mel is not None, "Flow decoder returned None"
    assert mel.dim() == 3, f"Expected 3D mel, got {mel.dim()}D"
    assert mel.shape[1] == 80, f"Expected 80 mel channels, got {mel.shape[1]}"
    expected_frames = 10 * 2  # target_tokens × token_mel_ratio
    assert mel.shape[2] == expected_frames, f"Expected {expected_frames} mel frames, got {mel.shape[2]}"
    print(f"Flow decoder output: {mel.shape}, mean={mel.mean():.4f}, std={mel.std():.4f}")


def test_flow_to_vocoder(device, flow_state_dict, hift_state_dict):
    """
    Test Flow Decoder → Vocoder pipeline: speech tokens → mel → audio.

    Uses TTNN flow decoder for mel generation, then reference PyTorch
    vocoder on host for mel → audio (f0 predictor not yet ported to TTNN).
    """
    import sys

    # Load flow decoder on device
    flow = TtCausalMaskedDiffWithDiT(device, flow_state_dict, dtype=ttnn.bfloat16)

    # Generate mel from dummy tokens
    torch.manual_seed(42)
    prompt_tokens = torch.randint(0, 6561, (1, 5))
    target_tokens = torch.randint(0, 6561, (1, 10))
    prompt_feat = torch.randn(1, 10, 80)
    embedding = torch.randn(1, 192)

    mel, _ = flow.inference(
        token=target_tokens,
        token_len=torch.tensor([10]),
        prompt_token=prompt_tokens,
        prompt_token_len=torch.tensor([5]),
        prompt_feat=prompt_feat,
        prompt_feat_len=torch.tensor([10]),
        embedding=embedding,
    )
    print(f"Flow mel output: {mel.shape}")

    # Load reference PyTorch vocoder for mel → audio
    sys.path.insert(0, "models/demos/wormhole/cosy_voice/ref/CosyVoice")
    try:
        from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
        from cosyvoice.hifigan.generator import CausalHiFTGenerator
    except ImportError:
        pytest.skip("Reference CosyVoice not importable — skipping vocoder integration")

    # Build reference vocoder
    f0_predictor = CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)
    vocoder = CausalHiFTGenerator(
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=24000,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        audio_limit=0.99,
        conv_pre_look_right=4,
        f0_predictor=f0_predictor,
    )

    # Load weights (hift_state_dict already has 'generator.' prefix stripped)
    vocoder.load_state_dict(hift_state_dict, strict=True)
    vocoder.eval()

    # Run vocoder: mel → audio
    with torch.no_grad():
        audio, source = vocoder.inference(mel.float(), finalize=True)

    assert audio is not None, "Vocoder returned None"
    assert audio.dim() == 2, f"Expected 2D audio, got {audio.dim()}D"
    print(f"Audio output: {audio.shape}, duration={audio.shape[1]/24000:.3f}s")
    print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

    # Basic sanity checks
    assert audio.shape[1] > 0, "Audio has zero length"
    assert audio.abs().max() <= 1.0, f"Audio exceeds [-1,1] range: {audio.abs().max()}"
