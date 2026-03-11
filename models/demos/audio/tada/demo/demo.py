# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TADA 1B Text-to-Speech demo on Tenstorrent hardware.

Usage:
    pytest models/demos/audio/tada/demo/demo.py -v

Requires:
    - HumeAI/tada-1b model weights (auto-downloaded from HuggingFace Hub)
    - HumeAI/tada-codec weights (auto-downloaded)
"""

import os

import pytest
import torch
from loguru import logger
from scipy.io import wavfile

from models.demos.audio.tada.tt.tada_generator import TadaGenerator, TadaInferenceOptions

TADA_L1_SMALL_SIZE = 24576
TADA_SAMPLE_RATE = 24000
TADA_MODEL_PATH = os.environ.get("TADA_MODEL_PATH", "HumeAI/tada-1b")
TADA_CODEC_PATH = os.environ.get("TADA_CODEC_PATH", "HumeAI/tada-codec")


def save_audio(filepath: str, audio: torch.Tensor, sample_rate: int = TADA_SAMPLE_RATE):
    """Save audio tensor to WAV file."""
    if audio.dim() == 3:
        audio = audio.squeeze(0).squeeze(0)
    elif audio.dim() == 2:
        audio = audio.squeeze(0)
    # Clamp to [-1, 1] and convert to int16
    audio = audio.clamp(-1.0, 1.0)
    audio_np = (audio.float().cpu().numpy() * 32767).astype("int16")
    wavfile.write(filepath, sample_rate, audio_np)
    logger.info(f"Saved audio to {filepath} ({audio_np.shape[0] / sample_rate:.2f}s)")


def load_audio(filepath: str, target_sr: int = TADA_SAMPLE_RATE) -> torch.Tensor:
    """Load audio file and resample to target sample rate. Returns (1, 1, T)."""
    import torchaudio

    waveform, sr = torchaudio.load(filepath)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.dim() == 2:
        waveform = waveform[:1]  # Mono
        waveform = waveform.unsqueeze(0)  # (1, 1, T)
    return waveform


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_tada_tts_demo(mesh_device):
    """
    End-to-end TADA TTS demo: generate speech from text.
    """
    generator = TadaGenerator(
        mesh_device,
        tada_model_id=TADA_MODEL_PATH,
        codec_model_id=TADA_CODEC_PATH,
        max_seq_len=2048,
        max_batch_size=1,
    )

    # Generate unconditional (no prompt audio)
    opts = TadaInferenceOptions(
        num_flow_matching_steps=20,
        acoustic_cfg_scale=1.6,
        noise_temperature=0.9,
        random_seed=42,
    )

    result = generator.generate(
        prompt_audio=None,
        prompt_text="",
        generation_text="This is a test of text to speech on Tenstorrent hardware.",
        inference_options=opts,
        max_steps=200,
    )

    audio = result["audio"]
    assert audio is not None, "No audio generated"
    assert audio.numel() > 0, "Empty audio output"

    # Save output
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "demo_output.wav")
    save_audio(output_path, audio)

    # Save reference decoder output for comparison
    if "audio_reference" in result and result["audio_reference"] is not None:
        ref_path = os.path.join(output_dir, "demo_output_reference.wav")
        save_audio(ref_path, result["audio_reference"])
        logger.info(f"Saved reference audio to {ref_path}")

    logger.info(f"Generated {audio.numel() / TADA_SAMPLE_RATE:.2f}s of audio")
    logger.info(f"Generated text tokens: {result.get('text', '')[:200]}")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "cfg_scale,suffix",
    [(1.0, "no_cfg"), (1.2, "low_cfg"), (1.6, "default_cfg")],
    ids=["no_cfg", "low_cfg", "default_cfg"],
)
def test_tada_tts_cfg_variants(mesh_device, cfg_scale, suffix):
    """
    Test TADA TTS with different CFG scales.

    For unconditional generation (no prompt audio), CFG's negative conditioning
    from pad tokens may be harmful. This test compares:
    - CFG=1.0 (no CFG)
    - CFG=1.2 (reduced CFG)
    - CFG=1.6 (default)
    """
    generator = TadaGenerator(
        mesh_device,
        tada_model_id=TADA_MODEL_PATH,
        codec_model_id=TADA_CODEC_PATH,
        max_seq_len=2048,
        max_batch_size=1,
    )

    opts = TadaInferenceOptions(
        num_flow_matching_steps=20,
        acoustic_cfg_scale=cfg_scale,
        noise_temperature=0.9,
        random_seed=42,
    )

    result = generator.generate(
        prompt_audio=None,
        prompt_text="",
        generation_text="This is a test of text to speech on Tenstorrent hardware.",
        inference_options=opts,
        max_steps=200,
    )

    audio = result["audio"]
    assert audio is not None, "No audio generated"
    assert audio.numel() > 0, "Empty audio output"

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"demo_output_{suffix}.wav")
    save_audio(output_path, audio)
    logger.info(f"Generated {audio.numel() / TADA_SAMPLE_RATE:.2f}s of audio (CFG={cfg_scale})")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_tada_tts_with_prompt(mesh_device):
    """
    TADA TTS demo with voice cloning from a prompt audio.
    Requires a prompt WAV file at the expected path.
    """
    prompt_path = os.path.join(os.path.dirname(__file__), "prompt.wav")
    if not os.path.exists(prompt_path):
        pytest.skip(f"Prompt audio not found at {prompt_path}")

    prompt_audio = load_audio(prompt_path)
    prompt_text = "Hello world"  # Text spoken in the prompt

    generator = TadaGenerator(
        mesh_device,
        tada_model_id=TADA_MODEL_PATH,
        codec_model_id=TADA_CODEC_PATH,
    )

    result = generator.generate(
        prompt_audio=prompt_audio,
        prompt_text=prompt_text,
        generation_text="This is a test of voice cloning on Tenstorrent hardware.",
        max_steps=300,
    )

    audio = result["audio"]
    assert audio is not None and audio.numel() > 0

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "demo_prompted_output.wav")
    save_audio(output_path, audio)
    logger.info(f"Generated {audio.numel() / TADA_SAMPLE_RATE:.2f}s of prompted audio")
