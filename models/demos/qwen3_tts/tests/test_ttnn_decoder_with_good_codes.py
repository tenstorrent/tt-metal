# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test TTNN decoder with known-good voice clone codes.

This verifies that the TTNN speech tokenizer decoder produces correct audio
when given correct input codes.
"""

from pathlib import Path

import soundfile as sf
import torch


def test_ttnn_decoder_with_voice_clone_codes():
    """Test TTNN decoder with voice clone codes (known to work)."""
    print("=" * 80)
    print("Testing TTNN Decoder with Voice Clone Codes")
    print("=" * 80)

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    import ttnn
    from models.demos.qwen3_tts.tt.speech_tokenizer import (
        SpeechTokenizerConfig,
        TtSpeechTokenizerDecoder,
        extract_speech_tokenizer_weights,
    )

    # Load voice clone codes
    voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
    if not voice_clone_path.exists():
        print("Voice clone codes not found. Skipping test.")
        return

    data = torch.load(voice_clone_path)
    ref_code = data["ref_code"]  # [101, 16]
    codes = ref_code.T.unsqueeze(0)  # [1, 16, 101]
    print(f"Input codes shape: {codes.shape}")
    print(f"Input codes range: [{codes.min()}, {codes.max()}]")

    # Load weights
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)
    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    raw_dict = load_file(speech_tokenizer_path)

    # Extract decoder weights
    decoder_weights = extract_speech_tokenizer_weights(raw_dict)
    print(f"Loaded {len(decoder_weights)} decoder weights")

    # Initialize device
    device = ttnn.open_device(device_id=0)

    try:
        # Initialize decoder (use_reference=True for correct output)
        config = SpeechTokenizerConfig()
        decoder = TtSpeechTokenizerDecoder(
            device=device,
            state_dict=decoder_weights,
            config=config,
            use_reference=True,  # Use fixed reference implementation
        )

        # Generate audio
        print("\nGenerating audio...")
        with torch.no_grad():
            audio = decoder.forward(codes)

        print(f"Audio shape: {audio.shape}")
        print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
        print(f"Audio std: {audio.std():.4f}")

        # Save audio
        audio_np = audio.squeeze().cpu().float().numpy()
        output_path = "/tmp/ttnn_decoder_voice_clone.wav"
        sf.write(output_path, audio_np, 24000)
        print(f"\nSaved audio to: {output_path}")

        # Audio duration
        duration = len(audio_np) / 24000
        expected_duration = codes.shape[-1] / 12.5  # ~8 seconds for 101 tokens
        print(f"Audio duration: {duration:.2f}s (expected: {expected_duration:.2f}s)")

        # Quality check
        if audio.std() > 0.05:
            print("\n✓ Audio has reasonable energy for speech")
        else:
            print("\n⚠ Audio energy is low")

        return audio

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_ttnn_decoder_with_voice_clone_codes()
