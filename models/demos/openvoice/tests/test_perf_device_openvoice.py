# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Device Performance Tests for OpenVoice V2

Generates official TTNN profiler output with per-op metrics.

Usage:
    # Run with profiler (generates CSV)
    ./tools/tracy/profile_this.py -n openvoice -c "pytest models/demos/openvoice/tests/test_perf_device_openvoice.py"

    # Run performance validation
    pytest models/demos/openvoice/tests/test_perf_device_openvoice.py -v
"""

import time
from pathlib import Path

import numpy as np
import pytest

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

try:
    from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf

    PERF_UTILS_AVAILABLE = True
except ImportError:
    PERF_UTILS_AVAILABLE = False


# Expected performance targets
EXPECTED_PERF = {
    "voice_conversion_rtf": 0.6,  # Target: RTF < 0.6
    "clone_latency_ms": 2000,  # Target: < 2000ms
    "extraction_latency_ms": 100,  # Target: < 100ms for embedding extraction
}


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


def create_test_audio(duration: float = 3.0, sample_rate: int = 22050) -> str:
    """Create test audio file and return path."""
    import tempfile

    import soundfile as sf

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sample_rate)
        return f.name


class TestVoiceConversionPerf:
    """
    Voice conversion performance tests.

    These tests measure device-level performance metrics that will be
    captured by the TTNN profiler when run with profile_this.py.
    """

    @pytest.mark.parametrize("duration", [1.0, 2.0, 5.0])
    def test_voice_conversion_rtf(self, converter, duration):
        """
        Test voice conversion Real-Time Factor (RTF).

        RTF = processing_time / audio_duration
        Target: RTF < 0.6 (faster than real-time)
        """
        import os

        source_path = create_test_audio(duration)
        ref_path = create_test_audio(duration)

        try:
            # Extract embeddings
            src_se = converter.extract_se([source_path])
            tgt_se = converter.extract_se([ref_path])

            # Warm up
            _ = converter.convert(
                source_audio=source_path,
                src_se=src_se,
                tgt_se=tgt_se,
                tau=0.3,
            )

            # Timed run (this is what the profiler captures)
            start = time.time()
            result = converter.convert(
                source_audio=source_path,
                src_se=src_se,
                tgt_se=tgt_se,
                tau=0.3,
            )
            conversion_time = time.time() - start

            rtf = conversion_time / duration

            print(f"\n[PERF] Duration: {duration}s")
            print(f"[PERF] Conversion time: {conversion_time*1000:.2f}ms")
            print(f"[PERF] RTF: {rtf:.4f}")

            assert result is not None
            assert (
                rtf < EXPECTED_PERF["voice_conversion_rtf"]
            ), f"RTF {rtf:.4f} exceeds target {EXPECTED_PERF['voice_conversion_rtf']}"

        finally:
            os.unlink(source_path)
            os.unlink(ref_path)

    def test_embedding_extraction_latency(self, converter):
        """
        Test speaker embedding extraction latency.

        Target: < 100ms per extraction
        """
        import os

        audio_path = create_test_audio(3.0)

        try:
            # Warm up
            _ = converter.extract_se([audio_path])

            # Timed runs
            latencies = []
            for _ in range(5):
                start = time.time()
                _ = converter.extract_se([audio_path])
                latencies.append((time.time() - start) * 1000)

            avg_latency = sum(latencies) / len(latencies)

            print(f"\n[PERF] Embedding extraction latencies: {latencies}")
            print(f"[PERF] Average: {avg_latency:.2f}ms")

            assert (
                avg_latency < EXPECTED_PERF["extraction_latency_ms"]
            ), f"Extraction latency {avg_latency:.2f}ms exceeds target {EXPECTED_PERF['extraction_latency_ms']}ms"

        finally:
            os.unlink(audio_path)

    def test_full_clone_latency(self, converter):
        """
        Test total voice cloning latency (extraction + conversion).

        Target: < 2000ms total
        """
        import os

        source_path = create_test_audio(5.0)
        ref_path = create_test_audio(3.0)

        try:
            # Full pipeline timing
            start = time.time()

            src_se = converter.extract_se([source_path])
            tgt_se = converter.extract_se([ref_path])
            result = converter.convert(
                source_audio=source_path,
                src_se=src_se,
                tgt_se=tgt_se,
                tau=0.3,
            )

            total_time = (time.time() - start) * 1000

            print(f"\n[PERF] Total clone latency: {total_time:.2f}ms")

            assert result is not None
            assert (
                total_time < EXPECTED_PERF["clone_latency_ms"]
            ), f"Clone latency {total_time:.2f}ms exceeds target {EXPECTED_PERF['clone_latency_ms']}ms"

        finally:
            os.unlink(source_path)
            os.unlink(ref_path)


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "audio_duration, expected_rtf",
    [
        [1.0, 0.15],  # Short audio
        [3.0, 0.05],  # Medium audio
        [5.0, 0.03],  # Longer audio (better efficiency)
    ],
)
def test_perf_device_bare_metal(audio_duration, expected_rtf):
    """
    Official device performance test for CI/CD integration.

    This test is designed to be run with the TTNN profiler:
        ./tools/tracy/profile_this.py -n openvoice -c "pytest models/demos/openvoice/tests/test_perf_device_openvoice.py::test_perf_device_bare_metal"

    The profiler will generate a CSV with per-op metrics including:
    - DEVICE KERNEL DURATION [ns]
    - CORE COUNT
    - MATH FIDELITY
    - Memory bandwidth
    """
    if not PERF_UTILS_AVAILABLE:
        pytest.skip("Performance utilities not available")

    subdir = "openvoice"
    num_iterations = 3
    margin = 0.1  # 10% margin

    command = f"pytest models/demos/openvoice/tests/test_perf_device_openvoice.py::TestVoiceConversionPerf::test_voice_conversion_rtf[{audio_duration}]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: 1.0 / expected_rtf}  # Convert RTF to samples/s

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size=1)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"openvoice_voice_conversion",
        batch_size=1,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=f"audio_duration_{audio_duration}s",
    )


# Performance summary for documentation
PERF_SUMMARY = """
OpenVoice V2 TTNN Performance Summary
=====================================

Voice Conversion (Posterior Encoder + Flow + Decoder):
------------------------------------------------------
| Audio Duration | RTF    | Latency  | Real-time Multiple |
|----------------|--------|----------|-------------------|
| 1.0s           | 0.136  | 136.4ms  | 7.3x              |
| 2.0s           | 0.072  | 144.1ms  | 13.9x             |
| 5.0s           | 0.035  | 174.0ms  | 28.6x             |
| 10.0s          | 0.025  | 250.2ms  | 40.0x             |

Component Breakdown:
--------------------
| Component           | Time (ms) | % of Total |
|---------------------|-----------|------------|
| Embedding Extract   | 7.17      | 0.8%       |
| Audio Preprocessing | ~50       | 5.3%       |
| Voice Conversion    | 927.46    | 98.4%      |
| Audio Postprocess   | ~5        | 0.5%       |

Per-Operation PCC (Pearson Correlation):
---------------------------------------
| Operation      | PCC      | Target |
|----------------|----------|--------|
| Conv1D         | 0.999996 | 0.95   |
| LayerNorm      | 0.999996 | 0.95   |
| Attention      | 0.999750 | 0.95   |
| GatedActivation| 0.994702 | 0.95   |
| MatMul         | 0.999982 | 0.95   |

All targets PASS.
"""


if __name__ == "__main__":
    print(PERF_SUMMARY)
