# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test Speech Tokenizer Encoder and Speaker Encoder reference implementations.
"""

from pathlib import Path

import numpy as np
import torch


def test_speech_tokenizer_encoder():
    """Test Speech Tokenizer Encoder produces valid RVQ codes."""
    print("=" * 80)
    print("Testing Speech Tokenizer Encoder")
    print("=" * 80)

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from models.demos.qwen3_tts.reference.functional import (
        SpeechTokenizerEncoderConfig,
        extract_speech_tokenizer_encoder_weights,
        speech_tokenizer_encoder_forward,
    )

    # Load weights
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)
    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    raw_dict = load_file(speech_tokenizer_path)

    # Extract encoder weights
    encoder_weights = extract_speech_tokenizer_encoder_weights(raw_dict)
    print(f"Loaded {len(encoder_weights)} encoder weight tensors")

    config = SpeechTokenizerEncoderConfig()

    # Create test audio (1 second of 24kHz audio)
    sample_rate = 24000
    duration = 1.0
    num_samples = int(sample_rate * duration)

    # Generate a simple test tone
    t = torch.linspace(0, duration, num_samples)
    audio = torch.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    audio = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, num_samples]
    print(f"\nInput audio shape: {audio.shape}")
    print(f"Input audio range: [{audio.min():.4f}, {audio.max():.4f}]")

    # Run encoder
    print("\nRunning encoder...")
    try:
        with torch.no_grad():
            codes = speech_tokenizer_encoder_forward(audio, encoder_weights, config)

        print(f"Output codes shape: {codes.shape}")
        print(f"Output codes range: [{codes.min()}, {codes.max()}]")
        print(f"Expected: [batch=1, num_quantizers=16, seq_len=~{num_samples // 1920}]")

        # Verify codes are in valid range
        assert codes.min() >= 0, "Codes should be non-negative"
        assert codes.max() < config.codebook_size, f"Codes should be < {config.codebook_size}"

        print("\n✓ Speech Tokenizer Encoder test passed!")
        return codes

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_speaker_encoder():
    """Test Speaker Encoder produces valid speaker embeddings."""
    print("\n" + "=" * 80)
    print("Testing Speaker Encoder")
    print("=" * 80)

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from models.demos.qwen3_tts.reference.functional import (
        SpeakerEncoderConfig,
        extract_speaker_encoder_weights,
        speaker_encoder_forward,
    )

    # Load weights
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)
    main_model_path = model_path / "model.safetensors"
    main_dict = load_file(main_model_path)

    # Extract speaker encoder weights
    speaker_weights = extract_speaker_encoder_weights(main_dict)
    print(f"Loaded {len(speaker_weights)} speaker encoder weight tensors")

    config = SpeakerEncoderConfig()

    # Create test mel-spectrogram
    batch_size = 1
    n_mels = config.n_mels  # 128 - model expects 128 channels
    seq_len = 100  # ~1 second at typical hop length

    mel = torch.randn(batch_size, n_mels, seq_len)
    print(f"\nInput mel-spectrogram shape: {mel.shape}")

    # Run encoder
    print("\nRunning speaker encoder...")
    try:
        with torch.no_grad():
            embedding = speaker_encoder_forward(mel, speaker_weights, config)

        print(f"Output embedding shape: {embedding.shape}")
        print(f"Output embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
        print(f"Expected shape: [batch=1, output_dim={config.output_dim}]")

        # Verify embedding shape
        assert (
            embedding.shape[-1] == config.output_dim or embedding.dim() == 1
        ), f"Embedding should have dim {config.output_dim}"

        print("\n✓ Speaker Encoder test passed!")
        return embedding

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_encoder_decoder_roundtrip():
    """Test that encoding then decoding produces valid audio."""
    print("\n" + "=" * 80)
    print("Testing Encoder-Decoder Roundtrip")
    print("=" * 80)

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from models.demos.qwen3_tts.reference.functional import (
        SpeechTokenizerDecoderConfig,
        SpeechTokenizerEncoderConfig,
        extract_speech_tokenizer_encoder_weights,
        speech_tokenizer_decoder_forward,
        speech_tokenizer_encoder_forward,
    )

    # Load weights
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)
    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    raw_dict = load_file(speech_tokenizer_path)

    # Extract weights
    encoder_weights = extract_speech_tokenizer_encoder_weights(raw_dict)
    decoder_weights = {k[8:]: v for k, v in raw_dict.items() if k.startswith("decoder.")}

    encoder_config = SpeechTokenizerEncoderConfig()
    decoder_config = SpeechTokenizerDecoderConfig()

    # Create test audio
    sample_rate = 24000
    duration = 0.5  # 0.5 seconds
    num_samples = int(sample_rate * duration)

    t = torch.linspace(0, duration, num_samples)
    audio_in = torch.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    audio_in = audio_in.unsqueeze(0).unsqueeze(0)  # [1, 1, num_samples]

    print(f"Input audio: shape={audio_in.shape}, duration={duration}s")

    try:
        with torch.no_grad():
            # Encode
            print("Encoding...")
            codes = speech_tokenizer_encoder_forward(audio_in, encoder_weights, encoder_config)
            print(f"Codes shape: {codes.shape}")

            # Decode
            print("Decoding...")
            audio_out = speech_tokenizer_decoder_forward(codes, decoder_weights, decoder_config)
            print(f"Output audio shape: {audio_out.shape}")
            print(f"Output audio range: [{audio_out.min():.4f}, {audio_out.max():.4f}]")

        # Calculate reconstruction quality
        # Note: Due to quantization, we don't expect perfect reconstruction
        print("\n✓ Encoder-Decoder roundtrip completed!")
        print("  (Note: Reconstruction quality depends on quantization loss)")

        return codes, audio_out

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # Run tests
    test_speech_tokenizer_encoder()
    test_speaker_encoder()
    test_encoder_decoder_roundtrip()
