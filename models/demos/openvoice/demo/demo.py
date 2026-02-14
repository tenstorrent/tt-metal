# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
OpenVoice V2 TTNN Demo

Demonstrates voice cloning and TTS capabilities on Tenstorrent hardware.

Usage:
    # Run all demos
    pytest models/demos/openvoice/demo/demo.py -v

    # Run specific demo
    pytest models/demos/openvoice/demo/demo.py::test_voice_cloning -v
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    print("Warning: TTNN not available, using PyTorch fallback")


# Try to import MeloTTS
try:
    from melo.api import TTS as MeloTTS

    MELO_AVAILABLE = True
except ImportError:
    MELO_AVAILABLE = False


def get_device():
    """Get TTNN device or None for CPU fallback."""
    if TTNN_AVAILABLE:
        try:
            return ttnn.open_device(device_id=0)
        except Exception as e:
            print(f"Could not open TTNN device: {e}")
    return None


def close_device(device):
    """Close TTNN device if open."""
    if device is not None and TTNN_AVAILABLE:
        ttnn.close_device(device)


@pytest.fixture(scope="module")
def device():
    """Pytest fixture for TTNN device."""
    dev = get_device()
    yield dev
    close_device(dev)


@pytest.fixture(scope="module")
def voice_converter(device):
    """Load voice converter model."""
    from models.demos.openvoice.tt.tone_color_converter import TTNNToneColorConverter

    checkpoint_dir = Path("checkpoints/openvoice/converter")
    if not checkpoint_dir.exists():
        pytest.skip('Checkpoint not found. Download with: python -c "from huggingface_hub import hf_hub_download; ..."')

    converter = TTNNToneColorConverter(checkpoint_dir / "config.json", device=device)
    converter.load_checkpoint(checkpoint_dir / "checkpoint.pth")
    return converter


@pytest.fixture(scope="module")
def tts_model():
    """Load MeloTTS model."""
    if not MELO_AVAILABLE:
        pytest.skip("MeloTTS not available. Install with: pip install melotts")

    return MeloTTS(language="EN", device="cpu")


def test_voice_cloning(device, voice_converter):
    """
    Test voice cloning: Convert source audio to target speaker's voice.

    This demonstrates the core voice conversion capability.
    """
    print("\n" + "=" * 60)
    print("Test: Voice Cloning")
    print("=" * 60)

    # Create test audio (sine wave)
    sample_rate = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

    # Save temp files
    import tempfile

    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        source_path = f.name
        sf.write(source_path, test_audio, sample_rate)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        ref_path = f.name
        # Slightly different frequency for reference
        ref_audio = (np.sin(2 * np.pi * 330 * t) * 0.3).astype(np.float32)
        sf.write(ref_path, ref_audio, sample_rate)

    try:
        # Extract embeddings
        print("Extracting speaker embeddings...")
        start_time = time.time()
        src_se = voice_converter.extract_se([source_path])
        tgt_se = voice_converter.extract_se([ref_path])
        extraction_time = time.time() - start_time
        print(f"  Extraction time: {extraction_time*1000:.2f}ms")

        # Convert voice
        print("Converting voice...")
        start_time = time.time()
        converted_audio = voice_converter.convert(
            source_audio=source_path,
            src_se=src_se,
            tgt_se=tgt_se,
            tau=0.3,
        )
        conversion_time = time.time() - start_time

        # Calculate RTF
        rtf = conversion_time / duration
        print(f"  Conversion time: {conversion_time*1000:.2f}ms")
        print(f"  RTF: {rtf:.4f} ({1/rtf:.1f}x real-time)")

        # Verify output
        assert converted_audio is not None
        assert len(converted_audio) > 0
        assert rtf < 0.6, f"RTF {rtf:.4f} exceeds target 0.6"

        print(f"\nResult: PASS (RTF={rtf:.4f} < 0.6)")

    finally:
        os.unlink(source_path)
        os.unlink(ref_path)


def test_tts_basic(tts_model):
    """
    Test basic TTS: Generate speech from text.

    Uses MeloTTS for text-to-speech synthesis.
    """
    print("\n" + "=" * 60)
    print("Test: Basic TTS")
    print("=" * 60)

    text = "Hello, this is a test of the text to speech system."
    print(f"Input: '{text}'")

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_path = f.name

    try:
        start_time = time.time()
        tts_model.tts_to_file(text, 0, output_path=output_path, speed=1.0, quiet=True)
        tts_time = time.time() - start_time

        # Get audio duration
        import librosa

        audio, sr = librosa.load(output_path, sr=None)
        duration = len(audio) / sr
        rtf = tts_time / duration

        print(f"  TTS time: {tts_time:.3f}s")
        print(f"  Audio duration: {duration:.2f}s")
        print(f"  RTF: {rtf:.4f}")

        assert duration > 0
        assert rtf < 0.6, f"RTF {rtf:.4f} exceeds target 0.6"

        print(f"\nResult: PASS (RTF={rtf:.4f} < 0.6)")

    finally:
        os.unlink(output_path)


def test_full_pipeline(device, voice_converter, tts_model):
    """
    Test full pipeline: TTS + Voice Cloning

    1. Generate base speech with MeloTTS
    2. Clone voice to match reference speaker
    """
    print("\n" + "=" * 60)
    print("Test: Full Pipeline (TTS + Voice Cloning)")
    print("=" * 60)

    text = "OpenVoice provides instant voice cloning with high quality."
    print(f"Input: '{text}'")

    import tempfile

    import soundfile as sf

    # Create reference audio
    sample_rate = 22050
    duration_ref = 3.0
    t = np.linspace(0, duration_ref, int(sample_rate * duration_ref))
    ref_audio = (np.sin(2 * np.pi * 330 * t) * 0.3).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        ref_path = f.name
        sf.write(ref_path, ref_audio, sample_rate)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tts_path = f.name

    try:
        # Step 1: TTS
        print("\nStep 1: Text-to-Speech")
        start_time = time.time()
        tts_model.tts_to_file(text, 0, output_path=tts_path, speed=1.0, quiet=True)
        tts_time = time.time() - start_time
        print(f"  TTS time: {tts_time:.3f}s")

        # Step 2: Voice Cloning
        print("\nStep 2: Voice Cloning")
        start_time = time.time()

        src_se = voice_converter.extract_se([tts_path])
        tgt_se = voice_converter.extract_se([ref_path])
        extraction_time = time.time() - start_time

        start_time = time.time()
        converted_audio = voice_converter.convert(
            source_audio=tts_path,
            src_se=src_se,
            tgt_se=tgt_se,
            tau=0.3,
        )
        conversion_time = time.time() - start_time

        print(f"  Extraction time: {extraction_time*1000:.2f}ms")
        print(f"  Conversion time: {conversion_time*1000:.2f}ms")

        # Total metrics
        import librosa

        audio, sr = librosa.load(tts_path, sr=None)
        audio_duration = len(audio) / sr

        total_time = tts_time + extraction_time + conversion_time
        total_rtf = total_time / audio_duration

        print(f"\nTotal Pipeline:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  End-to-end RTF: {total_rtf:.4f}")

        assert converted_audio is not None
        assert len(converted_audio) > 0

        print(f"\nResult: PASS")

    finally:
        os.unlink(ref_path)
        os.unlink(tts_path)


def test_multilingual(tts_model):
    """
    Test multi-lingual TTS for all 6 supported languages.
    """
    print("\n" + "=" * 60)
    print("Test: Multi-lingual TTS")
    print("=" * 60)

    test_texts = {
        "EN": "Hello, this is a test.",
        "ES": "Hola, esta es una prueba.",
        "FR": "Bonjour, ceci est un test.",
        "ZH": "你好，这是一个测试。",
        "JP": "こんにちは、これはテストです。",
        "KR": "안녕하세요, 이것은 테스트입니다.",
    }

    results = {}

    for lang, text in test_texts.items():
        print(f"\n[{lang}] {text}")

        try:
            tts = MeloTTS(language=lang, device="cpu")

            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name

            start_time = time.time()
            tts.tts_to_file(text, 0, output_path=output_path, speed=1.0, quiet=True)
            tts_time = time.time() - start_time

            import librosa

            audio, sr = librosa.load(output_path, sr=None)
            duration = len(audio) / sr
            rtf = tts_time / duration

            results[lang] = {"success": True, "rtf": rtf}
            print(f"  Duration: {duration:.2f}s, RTF: {rtf:.4f}")

            os.unlink(output_path)

        except Exception as e:
            results[lang] = {"success": False, "error": str(e)}
            print(f"  Error: {e}")

    # Summary
    print("\n" + "-" * 40)
    print("Summary:")
    passed = sum(1 for r in results.values() if r["success"])
    print(f"  Languages passed: {passed}/6")

    for lang, result in results.items():
        status = "PASS" if result["success"] else "FAIL"
        rtf = f"RTF={result['rtf']:.4f}" if result["success"] else result.get("error", "")
        print(f"  {lang}: {status} ({rtf})")

    assert passed >= 6, f"Only {passed}/6 languages passed"


def test_latency_benchmark(device, voice_converter):
    """
    Benchmark latency for voice cloning.

    Target: Total clone latency < 2000ms
    """
    print("\n" + "=" * 60)
    print("Test: Latency Benchmark")
    print("=" * 60)

    # Create test audio
    sample_rate = 22050
    duration = 5.0  # 5 second audio
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

    import tempfile

    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        source_path = f.name
        sf.write(f.name, test_audio, sample_rate)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        ref_path = f.name
        sf.write(f.name, test_audio, sample_rate)

    try:
        # Warm up
        _ = voice_converter.extract_se([source_path])

        # Benchmark extraction
        start_time = time.time()
        src_se = voice_converter.extract_se([source_path])
        tgt_se = voice_converter.extract_se([ref_path])
        extraction_time = (time.time() - start_time) * 1000

        # Benchmark conversion
        start_time = time.time()
        _ = voice_converter.convert(
            source_audio=source_path,
            src_se=src_se,
            tgt_se=tgt_se,
            tau=0.3,
        )
        conversion_time = (time.time() - start_time) * 1000

        total_time = extraction_time + conversion_time

        print(f"\nLatency Results:")
        print(f"  Extraction: {extraction_time:.2f}ms")
        print(f"  Conversion: {conversion_time:.2f}ms")
        print(f"  Total: {total_time:.2f}ms")
        print(f"\nTarget: < 2000ms")
        print(f"Status: {'PASS' if total_time < 2000 else 'FAIL'}")

        assert total_time < 2000, f"Total latency {total_time:.2f}ms exceeds 2000ms target"

    finally:
        os.unlink(source_path)
        os.unlink(ref_path)


if __name__ == "__main__":
    # Run demos directly
    device = get_device()

    try:
        from models.demos.openvoice.tt.tone_color_converter import TTNNToneColorConverter

        checkpoint_dir = Path("checkpoints/openvoice/converter")
        if checkpoint_dir.exists():
            converter = TTNNToneColorConverter(checkpoint_dir / "config.json", device=device)
            converter.load_checkpoint(checkpoint_dir / "checkpoint.pth")

            print("Running voice cloning test...")
            # Manual test execution
            test_voice_cloning(device, converter)

    finally:
        close_device(device)
