# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test decoder by directly importing model components from HuggingFace model files.

This bypasses the qwen_tts package dependencies (torchaudio, sox) that are hard to install.
"""

import sys
from pathlib import Path

import soundfile as sf
import torch


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    a = a.flatten().float()
    b = b.flatten().float()
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    cov = (a_centered * b_centered).sum()
    std_a = torch.sqrt((a_centered**2).sum())
    std_b = torch.sqrt((b_centered**2).sum())
    if std_a == 0 or std_b == 0:
        return 0.0
    return (cov / (std_a * std_b)).item()


def test_decoder_direct():
    """
    Test by directly loading the speech tokenizer decoder model.
    """
    print("=" * 80)
    print("Testing Decoder by Direct Model Loading")
    print("=" * 80)

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from models.demos.qwen3_tts.reference.functional import (
        SpeechTokenizerDecoderConfig,
        speech_tokenizer_decoder_forward,
    )

    # Load weights
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)
    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    raw_dict = load_file(speech_tokenizer_path)

    # Strip decoder prefix
    state_dict = {}
    for k, v in raw_dict.items():
        if k.startswith("decoder."):
            state_dict[k[8:]] = v

    config = SpeechTokenizerDecoderConfig()

    # Get test codes
    voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
    if voice_clone_path.exists():
        data = torch.load(voice_clone_path)
        ref_code = data["ref_code"]
        codes = ref_code.T.unsqueeze(0)  # [1, 16, 101]
        print(f"  Loaded ref_code from voice clone: {codes.shape}")
    else:
        codes = torch.randint(0, 2048, (1, 16, 50))
        print(f"  Using random codes: {codes.shape}")

    # Run our reference decoder
    print("\n--- Running Reference Decoder ---")
    with torch.no_grad():
        ref_audio = speech_tokenizer_decoder_forward(codes, state_dict, config)
        print(f"  Reference audio shape: {ref_audio.shape}")
        print(f"  Reference audio range: [{ref_audio.min():.4f}, {ref_audio.max():.4f}]")
        print(f"  Reference audio std: {ref_audio.std():.4f}")

    # Save audio
    audio_np = ref_audio.squeeze().cpu().float().numpy()
    output_path = "/tmp/reference_decoder_output.wav"
    sf.write(output_path, audio_np, 24000)
    print(f"\n  Saved audio to: {output_path}")

    # Try to load official decoder directly without full qwen_tts
    print("\n--- Attempting to load official decoder components ---")

    try:
        # Try importing decoder components directly
        hf_model_path = Path(model_path)
        decoder_py_path = hf_model_path / "speech_tokenizer"

        if decoder_py_path.exists():
            sys.path.insert(0, str(decoder_py_path))
            print(f"  Added {decoder_py_path} to path")

        # Try loading config
        config_path = hf_model_path / "speech_tokenizer" / "config.json"
        if config_path.exists():
            import json

            with open(config_path) as f:
                official_config = json.load(f)
            print(f"  Loaded config: {official_config}")
    except Exception as e:
        print(f"  Could not load official components: {e}")

    # Audio analysis
    print("\n--- Audio Quality Analysis ---")
    audio_tensor = torch.tensor(audio_np)

    # Basic stats
    std = audio_tensor.std().item()
    mean_abs = audio_tensor.abs().mean().item()
    max_val = audio_tensor.abs().max().item()

    print(f"  Standard deviation: {std:.4f}")
    print(f"  Mean absolute value: {mean_abs:.4f}")
    print(f"  Max absolute value: {max_val:.4f}")

    # Check for silence
    silence_ratio = (audio_tensor.abs() < 0.01).float().mean().item()
    print(f"  Silence ratio (|x| < 0.01): {silence_ratio:.2%}")

    # Check for clipping
    clipping_ratio = (audio_tensor.abs() > 0.99).float().mean().item()
    print(f"  Clipping ratio (|x| > 0.99): {clipping_ratio:.2%}")

    # Simple voice activity detection
    # Real speech typically has energy variation
    frame_size = 1024
    num_frames = len(audio_np) // frame_size
    if num_frames > 0:
        frames = audio_np[: num_frames * frame_size].reshape(num_frames, frame_size)
        frame_energy = (frames**2).mean(axis=1)
        energy_std = frame_energy.std()
        energy_mean = frame_energy.mean()
        energy_ratio = energy_std / (energy_mean + 1e-8)
        print(f"  Frame energy variation (std/mean): {energy_ratio:.4f}")

        if energy_ratio > 0.5:
            print("  ✓ Energy variation suggests speech-like content")
        else:
            print("  ⚠ Low energy variation - may be noise or silence")

    print("\n" + "=" * 80)
    print("IMPORTANT: Please listen to the audio file to verify!")
    print(f"  File: {output_path}")
    print("=" * 80)

    return ref_audio


if __name__ == "__main__":
    test_decoder_direct()
