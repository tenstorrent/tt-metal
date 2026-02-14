# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests for OpenVoice V2 TTNN implementation.

These tests verify the complete pipeline works correctly on TTNN hardware.

Usage:
    pytest models/demos/openvoice/tests/test_openvoice.py -v
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False


@pytest.fixture(scope="module")
def device():
    """Get TTNN device."""
    if not TTNN_AVAILABLE:
        pytest.skip("TTNN not available")

    try:
        dev = ttnn.open_device(device_id=0)
        yield dev
        ttnn.close_device(dev)
    except Exception as e:
        pytest.skip(f"Could not open TTNN device: {e}")


@pytest.fixture(scope="module")
def converter(device):
    """Load voice converter."""
    from models.demos.openvoice.tt.tone_color_converter import TTNNToneColorConverter

    checkpoint_dir = Path("checkpoints/openvoice/converter")
    if not checkpoint_dir.exists():
        pytest.skip("Checkpoint not found")

    conv = TTNNToneColorConverter(checkpoint_dir / "config.json", device=device)
    conv.load_checkpoint(checkpoint_dir / "checkpoint.pth")
    return conv


class TestVoiceConversion:
    """Tests for voice conversion pipeline."""

    def test_embedding_extraction(self, converter):
        """Test speaker embedding extraction."""
        import tempfile

        import soundfile as sf

        # Create test audio
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            temp_path = f.name

        try:
            start = time.time()
            se = converter.extract_se([temp_path])
            extraction_time = (time.time() - start) * 1000

            assert se is not None
            assert se.shape[1] == 256  # gin_channels
            assert extraction_time < 100, f"Extraction took {extraction_time:.2f}ms (target <100ms)"

        finally:
            os.unlink(temp_path)

    def test_voice_conversion_rtf(self, converter):
        """Test voice conversion meets RTF target."""
        import tempfile

        import soundfile as sf

        sample_rate = 22050
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            source_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            ref_path = f.name

        try:
            src_se = converter.extract_se([source_path])
            tgt_se = converter.extract_se([ref_path])

            start = time.time()
            result = converter.convert(
                source_audio=source_path,
                src_se=src_se,
                tgt_se=tgt_se,
                tau=0.3,
            )
            conversion_time = time.time() - start

            rtf = conversion_time / duration

            assert result is not None
            assert len(result) > 0
            assert rtf < 0.6, f"RTF {rtf:.4f} exceeds 0.6 target"

        finally:
            os.unlink(source_path)
            os.unlink(ref_path)

    def test_latency_target(self, converter):
        """Test total clone latency < 2000ms."""
        import tempfile

        import soundfile as sf

        sample_rate = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            source_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            ref_path = f.name

        try:
            start = time.time()
            src_se = converter.extract_se([source_path])
            tgt_se = converter.extract_se([ref_path])
            _ = converter.convert(
                source_audio=source_path,
                src_se=src_se,
                tgt_se=tgt_se,
                tau=0.3,
            )
            total_time = (time.time() - start) * 1000

            assert total_time < 2000, f"Total latency {total_time:.2f}ms exceeds 2000ms"

        finally:
            os.unlink(source_path)
            os.unlink(ref_path)


class TestBatchProcessing:
    """Tests for batch processing capabilities."""

    def test_batch_conversion(self, converter):
        """Test batch voice conversion."""
        import tempfile

        import soundfile as sf

        from models.demos.openvoice.tt.tone_color_converter import BatchConversionItem

        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        # Create temp files
        temp_files = []
        output_dir = tempfile.mkdtemp()

        try:
            for i in range(4):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, audio, sample_rate)
                    temp_files.append(f.name)

            # Create batch items
            items = [
                BatchConversionItem(
                    source_audio=temp_files[i],
                    reference_audio=temp_files[(i + 1) % 4],
                    output_path=os.path.join(output_dir, f"output_{i}.wav"),
                    tau=0.3,
                )
                for i in range(4)
            ]

            # Run batch conversion
            results = converter.convert_batch(items, num_workers=2)

            successful = sum(1 for item in results if item.error is None)
            assert successful == 4, f"Only {successful}/4 conversions succeeded"

        finally:
            for f in temp_files:
                if os.path.exists(f):
                    os.unlink(f)
            import shutil

            shutil.rmtree(output_dir, ignore_errors=True)


class TestQuality:
    """Tests for output quality metrics."""

    def test_speaker_similarity(self, converter):
        """
        Test that converted audio maintains speaker similarity.

        Note: This is a basic shape test. Full similarity testing
        requires speaker verification models.
        """
        import tempfile

        import soundfile as sf

        sample_rate = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            audio_path = f.name

        try:
            se = converter.extract_se([audio_path])

            # Embedding should be valid
            assert se is not None
            assert not torch.isnan(se).any()
            assert not torch.isinf(se).any()

            # Embedding should have reasonable magnitude
            assert se.abs().mean() > 0.01
            assert se.abs().max() < 100

        finally:
            os.unlink(audio_path)
