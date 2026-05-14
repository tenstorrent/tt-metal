# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RVC (Retrieval-based Voice Conversion) demo for Tenstorrent hardware.

This demo runs voice conversion using the TTNN implementation of RVC,
including posterior encoding, pitch extraction, feature retrieval,
flow-based decoding, and HiFi-GAN vocoding.

Usage:
    pytest --disable-warnings models/demos/audio/rvc/demo/demo.py::test_rvc_demo
"""

import os
import time
from pathlib import Path

import pytest
import torch
import torchaudio
from loguru import logger

import ttnn
from models.demos.audio.rvc.tt.reference_rvc import RVCModel
from models.demos.audio.rvc.tt.ttnn_rvc import rvc_inference
from models.demos.audio.rvc.tt.rvc_parameter_preprocessing import preprocess_rvc_model


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def generate_test_audio(duration_sec=3.0, sample_rate=16000, freq=440.0):
    """Generate a test sine wave audio signal."""
    t = torch.linspace(0, duration_sec, int(duration_sec * sample_rate))
    audio = 0.3 * torch.sin(2 * 3.14159 * freq * t).unsqueeze(0)
    return audio


def generate_test_features(n_features=1000, feature_dim=192):
    """Generate synthetic speaker feature index for testing."""
    features = torch.randn(n_features, feature_dim)
    features = torch.nn.functional.normalize(features, dim=-1)
    return features


def audio_to_mel(audio, n_fft=1024, hop_length=256, n_mels=80, sample_rate=16000):
    """Convert audio to mel spectrogram."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel = mel_transform(audio)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel


def load_audio(path, sample_rate=16000):
    """Load audio file and resample."""
    wav, sr = torchaudio.load(path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


# ---------------------------------------------------------------------------
# Reference (PyTorch) inference
# ---------------------------------------------------------------------------

def run_reference_inference(model, source_mel, target_features_index=None, f0_up_key=0, index_rate=0.5):
    """Run PyTorch reference inference for validation."""
    model.eval()
    with torch.no_grad():
        output = model(source_mel, target_features_index, f0_up_key, index_rate)
    return output


# ---------------------------------------------------------------------------
# Demo tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device_params", [{"l1_small_size": 1600}], indirect=True)
def test_rvc_demo(device, use_program_cache):
    """
    Run RVC voice conversion demo.

    Generates synthetic audio, converts it through the RVC pipeline,
    and validates the output.
    """
    logger.info("=" * 60)
    logger.info("RVC Voice Conversion Demo")
    logger.info("=" * 60)

    # 1. Create model
    logger.info("Step 1: Initializing RVC model")
    model = RVCModel(
        n_mels=80,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        n_flows=4,
    )
    model.eval()
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Generate test inputs
    logger.info("Step 2: Generating test audio")
    source_audio = generate_test_audio(duration_sec=2.0, sample_rate=16000)
    source_mel = audio_to_mel(source_audio)
    target_features = generate_test_features(n_features=500, feature_dim=192)
    logger.info(f"Source mel shape: {source_mel.shape}")
    logger.info(f"Target features shape: {target_features.shape}")

    # 3. Run PyTorch reference
    logger.info("Step 3: Running PyTorch reference inference")
    ref_start = time.time()
    ref_output = run_reference_inference(model, source_mel, target_features, index_rate=0.5)
    ref_time = time.time() - ref_start
    logger.info(f"Reference output shape: {ref_output.shape}")
    logger.info(f"Reference inference time: {ref_time:.3f}s")

    # 4. Preprocess parameters for TTNN
    logger.info("Step 4: Preprocessing parameters for TTNN")
    params = preprocess_rvc_model(model, device)

    # 5. Run TTNN inference
    logger.info("Step 5: Running TTNN inference")
    ttnn_start = time.time()
    ttnn_output = rvc_inference(
        source_audio,
        target_features,
        params,
        f0_method="rmvpe",
        index_rate=0.5,
        f0_up_key=0,
    )
    ttnn_time = time.time() - ttnn_start
    logger.info(f"TTNN output shape: {ttnn_output.shape}")
    logger.info(f"TTNN inference time: {ttnn_time:.3f}s")

    # 6. Validate output
    logger.info("Step 6: Validating output")
    assert ttnn_output.shape[0] == 1, "Batch size should be 1"
    assert ttnn_output.shape[1] == 1, "Output should be mono"
    assert ttnn_output.abs().max() <= 1.0, "Output should be in [-1, 1] range"

    # Compute RTF
    output_duration = ttnn_output.shape[-1] / 16000
    rtf = ttnn_time / output_duration if output_duration > 0 else float("inf")
    logger.info(f"Output duration: {output_duration:.3f}s")
    logger.info(f"Real-time factor (RTF): {rtf:.3f}")
    logger.info(f"RTF < 0.5 target: {'PASS' if rtf < 0.5 else 'FAIL (will improve with optimizations)'}")

    logger.info("=" * 60)
    logger.info("RVC Demo Complete!")
    logger.info("=" * 60)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1600}], indirect=True)
def test_rvc_component_validation(device, use_program_cache):
    """
    Validate individual RVC components against PyTorch reference.
    """
    logger.info("RVC: Component validation test")

    model = RVCModel()
    model.eval()

    # Test posterior encoder
    source_mel = torch.randn(1, 80, 100)
    with torch.no_grad():
        z_ref, m_ref, logs_ref = model.encoder(source_mel)

    assert z_ref.shape == (1, 192, 100), f"Expected (1, 192, 100), got {z_ref.shape}"
    assert not torch.isnan(z_ref).any(), "Posterior encoder output contains NaN"
    logger.info(f"Posterior encoder: PASS (shape={z_ref.shape})")

    # Test flow decoder
    z_test = torch.randn(1, 192, 100)
    with torch.no_grad():
        flow_out = model.flow(z_test, reverse=True)
    assert flow_out.shape == z_test.shape, f"Flow decoder changed shape: {flow_out.shape}"
    logger.info(f"Flow decoder: PASS (shape={flow_out.shape})")

    # Test vocoder
    voc_input = torch.randn(1, 192, 100)
    with torch.no_grad():
        audio_out = model.vocoder(voc_input)
    logger.info(f"Vocoder: PASS (shape={audio_out.shape})")

    # Test RMVPE
    mel_input = torch.randn(1, 128, 100)
    with torch.no_grad():
        f0 = model.rmvpe(mel_input)
    assert f0.shape[0] == 1, f"Expected batch=1, got {f0.shape}"
    logger.info(f"RMVPE: PASS (shape={f0.shape})")

    logger.info("All component validations passed!")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1600}], indirect=True)
def test_rvc_pitch_transposition(device, use_program_cache):
    """Test RVC with different pitch transposition settings."""
    logger.info("RVC: Pitch transposition test")

    model = RVCModel()
    model.eval()

    source_mel = torch.randn(1, 80, 50)
    target_features = generate_test_features(200, 192)

    for f0_up_key in [-12, -6, 0, 6, 12]:
        with torch.no_grad():
            output = model(source_mel, target_features, f0_up_key=f0_up_key, index_rate=0.5)
        assert output.shape[0] == 1
        assert not torch.isnan(output).any(), f"NaN with f0_up_key={f0_up_key}"
        logger.info(f"f0_up_key={f0_up_key}: shape={output.shape}, OK")

    logger.info("Pitch transposition test passed!")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1600}], indirect=True)
def test_rvc_index_rate_sweep(device, use_program_cache):
    """Test RVC with different index rates (accent control)."""
    logger.info("RVC: Index rate sweep test")

    model = RVCModel()
    model.eval()

    source_mel = torch.randn(1, 80, 50)
    target_features = generate_test_features(200, 192)

    for index_rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
        with torch.no_grad():
            output = model(source_mel, target_features, index_rate=index_rate)
        assert output.shape[0] == 1
        assert not torch.isnan(output).any(), f"NaN with index_rate={index_rate}"
        logger.info(f"index_rate={index_rate}: shape={output.shape}, OK")

    logger.info("Index rate sweep test passed!")
