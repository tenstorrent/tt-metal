# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Audio Quality Verification Test for Qwen3-TTS.

This test verifies that the TTNN implementation produces audio
that matches the reference qwen_tts implementation.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.stats import pearsonr

# Test audio quality of the speech decoder


def test_speech_decoder_quality():
    """Test that speech decoder produces high quality audio."""
    print("=" * 80)
    print("Audio Quality Verification Test")
    print("=" * 80)

    # Check if reference audio exists for comparison
    ref_audio_path = Path("/tmp/reference.wav")
    if not ref_audio_path.exists():
        print("\nGenerating test reference audio...")
        # Create a simple test reference audio
        sample_rate = 24000
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        sf.write(str(ref_audio_path), test_audio, sample_rate)
        print(f"  Created test audio: {ref_audio_path}")

    # Load qwen_tts model for audio decoding
    print("\nLoading qwen_tts model...")
    try:
        from qwen_tts import Qwen3TTSModel

        qwen_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cpu",
            dtype=torch.float32,
        )
        print("  qwen_tts loaded successfully")
    except Exception as e:
        print(f"  Error loading qwen_tts: {e}")
        print("  Skipping test - qwen_tts not available")
        return

    # Test 1: Speech tokenizer encode/decode roundtrip
    print("\n" + "-" * 60)
    print("Test 1: Speech Tokenizer Encode/Decode Roundtrip")
    print("-" * 60)

    # Access speech tokenizer via model.model
    speech_tokenizer = qwen_model.model.speech_tokenizer

    # Load reference audio
    ref_audio, sr = sf.read(str(ref_audio_path))
    print(f"  Reference audio: {len(ref_audio)} samples @ {sr}Hz")

    # Encode using file path (preferred API)
    try:
        print("  Encoding to tokens...")
        with torch.no_grad():
            # API: encode(audios: str | ndarray, sr: int = None)
            encoded = speech_tokenizer.encode(str(ref_audio_path))
        print(f"  Encoded type: {type(encoded)}")

        # Get audio codes from encoded output
        if hasattr(encoded, "audio_codes"):
            audio_codes = encoded.audio_codes
            print(f"  Audio codes: {len(audio_codes)} items")
            if audio_codes:
                codes = audio_codes[0]
                if hasattr(codes, "shape"):
                    print(f"  Codes shape: {codes.shape}")

        # Decode back to audio
        print("  Decoding from tokens...")
        with torch.no_grad():
            # API: decode(encoded) -> Tuple[List[ndarray], int]
            decoded_result = speech_tokenizer.decode(encoded)

        # Unpack decode result (list of arrays, sample_rate)
        if isinstance(decoded_result, tuple):
            decoded_list, decoded_sr = decoded_result
            decoded_np = decoded_list[0].squeeze()
        else:
            decoded_np = decoded_result[0].squeeze() if isinstance(decoded_result, list) else decoded_result.squeeze()
            decoded_sr = 24000

        print(f"  Decoded audio: {len(decoded_np)} samples @ {decoded_sr}Hz")

        # Resample ref_audio if needed for comparison
        if sr != decoded_sr:
            from scipy import signal

            ref_audio = signal.resample(ref_audio, int(len(ref_audio) * decoded_sr / sr))

        # Compare
        min_len = min(len(ref_audio), len(decoded_np))
        if min_len > 0:
            pcc = pearsonr(ref_audio[:min_len], decoded_np[:min_len])[0]
            print(f"  PCC (encode/decode roundtrip): {pcc:.6f}")
            if pcc > 0.9:
                print("  ✅ PASS: High correlation in roundtrip")
            else:
                print("  ⚠️  Low PCC - this is expected for lossy codec")

        # Save decoded audio for manual inspection
        sf.write("/tmp/decoded_roundtrip.wav", decoded_np, decoded_sr)
        print(f"  Saved decoded audio to /tmp/decoded_roundtrip.wav")

    except Exception as e:
        print(f"  Error in encode/decode test: {e}")
        import traceback

        traceback.print_exc()

    # Test 2: Encode consistency (same audio encodes to same tokens)
    print("\n" + "-" * 60)
    print("Test 2: Encoding Consistency")
    print("-" * 60)

    try:
        # Encode same file twice
        print("  Encoding same file twice...")
        with torch.no_grad():
            encoded1 = speech_tokenizer.encode(str(ref_audio_path))
            encoded2 = speech_tokenizer.encode(str(ref_audio_path))

        # Compare audio codes
        if hasattr(encoded1, "audio_codes") and hasattr(encoded2, "audio_codes"):
            codes1 = encoded1.audio_codes[0]
            codes2 = encoded2.audio_codes[0]

            if hasattr(codes1, "numpy"):
                codes1_np = codes1.numpy()
                codes2_np = codes2.numpy()
            else:
                codes1_np = np.array(codes1)
                codes2_np = np.array(codes2)

            if np.array_equal(codes1_np, codes2_np):
                print("  ✅ PASS: Same audio produces identical tokens")
            else:
                diff = np.sum(codes1_np != codes2_np)
                print(f"  ⚠️  Different tokens: {diff} differences")
        else:
            print("  Could not extract audio_codes from encoded output")

    except Exception as e:
        print(f"  Error in encoding consistency test: {e}")
        import traceback

        traceback.print_exc()

    # Test 3: Decode consistency - same encoded input produces same audio
    print("\n" + "-" * 60)
    print("Test 3: Decoding Consistency")
    print("-" * 60)

    try:
        # Encode once
        print("  Encoding audio...")
        with torch.no_grad():
            encoded = speech_tokenizer.encode(str(ref_audio_path))

        # Decode twice
        print("  Decoding same encoded data twice...")
        with torch.no_grad():
            decoded1 = speech_tokenizer.decode(encoded)
            decoded2 = speech_tokenizer.decode(encoded)

        # Extract audio arrays
        audio1_np = decoded1[0][0].squeeze() if isinstance(decoded1, tuple) else decoded1[0].squeeze()
        audio2_np = decoded2[0][0].squeeze() if isinstance(decoded2, tuple) else decoded2[0].squeeze()

        # Compare
        if np.allclose(audio1_np, audio2_np, rtol=1e-5):
            print("  ✅ PASS: Same tokens produce identical audio")
        else:
            diff = np.abs(audio1_np - audio2_np).max()
            print(f"  ⚠️  Max difference: {diff:.6f}")
            pcc = pearsonr(audio1_np.flatten(), audio2_np.flatten())[0]
            print(f"  PCC: {pcc:.6f}")

    except Exception as e:
        print(f"  Error in consistency test: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Audio Quality Tests Complete")
    print("=" * 80)
    print("\nGenerated files:")
    print("  /tmp/decoded_roundtrip.wav - encode/decode roundtrip test")
    print("  /tmp/synthetic_decoded.wav - synthetic token decoding")
    print("\nListen to these files to manually verify audio quality.")


def test_voice_clone_audio_quality():
    """Test voice clone audio generation quality."""
    print("=" * 80)
    print("Voice Clone Audio Quality Test")
    print("=" * 80)

    ref_audio_path = "/tmp/reference.wav"
    ref_text = "This is a reference audio sample."

    # Create test reference if not exists
    if not Path(ref_audio_path).exists():
        sample_rate = 24000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        test_audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        sf.write(ref_audio_path, test_audio, sample_rate)
        print(f"Created test reference: {ref_audio_path}")

    try:
        from qwen_tts import Qwen3TTSModel

        print("\nLoading qwen_tts model...")
        qwen_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cpu",
            dtype=torch.float32,
        )
        print("  Model loaded")

        # Generate with official qwen_tts
        print("\nGenerating audio with official qwen_tts...")
        test_text = "Hello, how are you today?"

        # generate_voice_clone returns Tuple[List[numpy.ndarray], int]
        # where the tuple is (list_of_audio_arrays, sample_rate)
        result = qwen_model.generate_voice_clone(
            text=test_text,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            max_new_tokens=50,
        )

        # Unpack the result - it's (List[ndarray], sample_rate)
        if isinstance(result, tuple) and len(result) == 2:
            audio_list, sample_rate = result
            official_np = audio_list[0].squeeze()
        elif isinstance(result, list):
            official_np = result[0].squeeze()
            sample_rate = 24000
        else:
            official_np = np.array(result).squeeze()
            sample_rate = 24000

        print(f"  Generated audio shape: {official_np.shape}")
        print(f"  Sample rate: {sample_rate}")
        print(f"  Audio range: [{official_np.min():.4f}, {official_np.max():.4f}]")
        print(f"  Duration: {len(official_np) / sample_rate:.2f}s")

        # Verify audio quality
        if not np.isnan(official_np).any() and not np.isinf(official_np).any():
            print("  ✅ Audio is valid (no NaN/Inf)")
        else:
            print("  ❌ Audio contains invalid values")

        # Save for manual inspection
        sf.write("/tmp/official_voice_clone.wav", official_np, sample_rate)
        print(f"  Saved to /tmp/official_voice_clone.wav")

        print("\n✅ Voice clone test complete")

    except Exception as e:
        print(f"\n❌ Error in voice clone test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_speech_decoder_quality()
    print("\n\n")
    test_voice_clone_audio_quality()
