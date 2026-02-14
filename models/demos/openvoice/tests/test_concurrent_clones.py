# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Test 10 concurrent voice clones - Stage 3 requirement for bounty.

This test verifies the system can handle 10 concurrent voice clone requests
as required by the bounty specification.
"""

import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pytest

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    print("Warning: TTNN not available, using PyTorch fallback")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


def create_test_audio(duration: float = 2.0, freq: float = 440.0, sample_rate: int = 22050) -> np.ndarray:
    """Create a synthetic test audio signal."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
    return audio


@pytest.fixture(scope="module")
def device():
    """Get TTNN device."""
    if TTNN_AVAILABLE:
        try:
            dev = ttnn.open_device(device_id=0)
            yield dev
            ttnn.close_device(dev)
        except Exception as e:
            print(f"Could not open TTNN device: {e}")
            yield None
    else:
        yield None


@pytest.fixture(scope="module")
def voice_converter(device):
    """Load voice converter model."""
    from models.demos.openvoice.tt.tone_color_converter import TTNNToneColorConverter

    checkpoint_dir = Path("checkpoints/openvoice/converter")
    if not checkpoint_dir.exists():
        pytest.skip("Checkpoint not found")

    converter = TTNNToneColorConverter(
        checkpoint_dir / "config.json",
        device=device,
        enable_cache=True,  # Enable caching for concurrent access
    )
    converter.load_checkpoint(checkpoint_dir / "checkpoint.pth")
    return converter


class TestConcurrentClones:
    """Test suite for concurrent voice cloning operations."""

    def test_10_concurrent_voice_clones(self, device, voice_converter):
        """
        Stage 3 Requirement: Test 10 concurrent voice clone requests.

        This test:
        1. Creates 10 unique source/reference audio pairs
        2. Submits all 10 voice clone requests concurrently
        3. Measures total throughput and per-clone latency
        4. Verifies all clones complete successfully
        """
        import soundfile as sf

        print("\n" + "=" * 60)
        print("Test: 10 Concurrent Voice Clones (Stage 3 Requirement)")
        print("=" * 60)

        num_clones = 10
        sample_rate = 22050
        duration = 2.0  # 2 second audio clips

        # Create temporary directory for test files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create 10 unique source and reference audio files
            print(f"\nCreating {num_clones} source/reference audio pairs...")
            test_files = []
            for i in range(num_clones):
                # Source: different frequencies for each
                source_freq = 440 + (i * 50)  # 440, 490, 540, ...
                source_audio = create_test_audio(duration, source_freq, sample_rate)
                source_path = tmpdir / f"source_{i}.wav"
                sf.write(source_path, source_audio, sample_rate)

                # Reference: different frequencies
                ref_freq = 330 + (i * 30)  # 330, 360, 390, ...
                ref_audio = create_test_audio(duration, ref_freq, sample_rate)
                ref_path = tmpdir / f"reference_{i}.wav"
                sf.write(ref_path, ref_audio, sample_rate)

                test_files.append(
                    {
                        "source": source_path,
                        "reference": ref_path,
                        "output": tmpdir / f"output_{i}.wav",
                        "index": i,
                    }
                )

            # Pre-extract embeddings to focus test on conversion
            print("Pre-extracting speaker embeddings...")
            embeddings = {}
            for tf in test_files:
                embeddings[f"src_{tf['index']}"] = voice_converter.extract_se([str(tf["source"])])
                embeddings[f"ref_{tf['index']}"] = voice_converter.extract_se([str(tf["reference"])])

            # Results tracking
            results = {}
            results_lock = threading.Lock()

            def run_voice_clone(tf):
                """Run a single voice clone operation."""
                idx = tf["index"]
                try:
                    start_time = time.time()

                    # Get pre-computed embeddings
                    src_se = embeddings[f"src_{idx}"]
                    tgt_se = embeddings[f"ref_{idx}"]

                    # Run voice conversion
                    audio = voice_converter.convert(
                        source_audio=str(tf["source"]),
                        src_se=src_se,
                        tgt_se=tgt_se,
                        output_path=str(tf["output"]),
                        tau=0.3,
                    )

                    elapsed = time.time() - start_time

                    return {
                        "index": idx,
                        "success": True,
                        "time": elapsed,
                        "output_size": len(audio),
                    }
                except Exception as e:
                    return {
                        "index": idx,
                        "success": False,
                        "error": str(e),
                        "time": 0,
                    }

            # Run 10 concurrent voice clones
            print(f"\nSubmitting {num_clones} concurrent voice clone requests...")
            overall_start = time.time()

            with ThreadPoolExecutor(max_workers=num_clones) as executor:
                futures = {executor.submit(run_voice_clone, tf): tf["index"] for tf in test_files}

                for future in as_completed(futures):
                    result = future.result()
                    with results_lock:
                        results[result["index"]] = result

                    status = "✓" if result["success"] else "✗"
                    time_str = f"{result['time']*1000:.1f}ms" if result["success"] else result.get("error", "")[:40]
                    print(f"  Clone {result['index']}: {status} ({time_str})")

            overall_time = time.time() - overall_start

            # Calculate statistics
            successful = [r for r in results.values() if r["success"]]
            failed = [r for r in results.values() if not r["success"]]

            if successful:
                times = [r["time"] for r in successful]
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
            else:
                avg_time = min_time = max_time = 0

            throughput = len(successful) / overall_time if overall_time > 0 else 0

            # Print results
            print("\n" + "-" * 40)
            print("Results Summary:")
            print(f"  Total clones: {num_clones}")
            print(f"  Successful: {len(successful)}")
            print(f"  Failed: {len(failed)}")
            print(f"\nTiming:")
            print(f"  Wall clock time: {overall_time*1000:.1f}ms")
            print(f"  Throughput: {throughput:.2f} clones/sec")
            print(f"  Avg latency per clone: {avg_time*1000:.1f}ms")
            print(f"  Min latency: {min_time*1000:.1f}ms")
            print(f"  Max latency: {max_time*1000:.1f}ms")

            # Verify outputs exist
            print("\nVerifying outputs...")
            outputs_verified = 0
            for tf in test_files:
                if tf["output"].exists():
                    outputs_verified += 1

            print(f"  Output files created: {outputs_verified}/{num_clones}")

            # Assertions
            assert len(successful) >= 10, f"Only {len(successful)}/10 clones succeeded"
            assert outputs_verified >= 10, f"Only {outputs_verified}/10 output files created"

            print("\n" + "=" * 60)
            print("PASS: 10 concurrent voice clones completed successfully")
            print("=" * 60)

    def test_concurrent_with_shared_reference(self, device, voice_converter):
        """
        Test concurrent clones where all use the same target voice.

        This is a common use case: clone multiple source audios to
        the same target speaker.
        """
        import soundfile as sf

        print("\n" + "=" * 60)
        print("Test: Concurrent Clones with Shared Reference")
        print("=" * 60)

        num_clones = 10
        sample_rate = 22050
        duration = 2.0

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create one shared reference
            ref_audio = create_test_audio(duration, 330, sample_rate)
            ref_path = tmpdir / "shared_reference.wav"
            sf.write(ref_path, ref_audio, sample_rate)

            # Pre-extract target embedding (shared)
            print("Extracting shared target embedding...")
            tgt_se = voice_converter.extract_se([str(ref_path)])

            # Create multiple source files
            test_files = []
            for i in range(num_clones):
                source_freq = 440 + (i * 50)
                source_audio = create_test_audio(duration, source_freq, sample_rate)
                source_path = tmpdir / f"source_{i}.wav"
                sf.write(source_path, source_audio, sample_rate)

                test_files.append(
                    {
                        "source": source_path,
                        "output": tmpdir / f"output_{i}.wav",
                        "index": i,
                    }
                )

            # Pre-extract source embeddings
            print("Pre-extracting source embeddings...")
            src_embeddings = {}
            for tf in test_files:
                src_embeddings[tf["index"]] = voice_converter.extract_se([str(tf["source"])])

            results = []
            results_lock = threading.Lock()

            def run_clone(tf):
                idx = tf["index"]
                try:
                    start = time.time()
                    audio = voice_converter.convert(
                        source_audio=str(tf["source"]),
                        src_se=src_embeddings[idx],
                        tgt_se=tgt_se,  # Shared target
                        tau=0.3,
                    )
                    elapsed = time.time() - start
                    return {"index": idx, "success": True, "time": elapsed}
                except Exception as e:
                    return {"index": idx, "success": False, "error": str(e)}

            print(f"\nRunning {num_clones} concurrent clones with shared target...")
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=num_clones) as executor:
                futures = [executor.submit(run_clone, tf) for tf in test_files]
                for future in as_completed(futures):
                    results.append(future.result())

            total_time = time.time() - start_time

            successful = [r for r in results if r["success"]]
            print(f"\nResults: {len(successful)}/{num_clones} successful")
            print(f"Total time: {total_time*1000:.1f}ms")
            print(f"Throughput: {len(successful)/total_time:.2f} clones/sec")

            assert len(successful) >= num_clones, f"Only {len(successful)}/{num_clones} succeeded"

            print("\nPASS: Shared reference concurrent cloning works")

    def test_pipelined_batch_10_clones(self, device, voice_converter):
        """
        Test using the pipelined batch API for 10 clones.

        This tests the optimized pipeline mode which overlaps CPU
        preprocessing with TTNN inference.
        """
        import soundfile as sf

        from models.demos.openvoice.tt.tone_color_converter import BatchConversionItem

        print("\n" + "=" * 60)
        print("Test: Pipelined Batch Processing (10 clones)")
        print("=" * 60)

        num_clones = 10
        sample_rate = 22050
        duration = 2.0

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create source and reference files
            test_items = []
            for i in range(num_clones):
                source_audio = create_test_audio(duration, 440 + i * 50, sample_rate)
                source_path = tmpdir / f"source_{i}.wav"
                sf.write(source_path, source_audio, sample_rate)

                ref_audio = create_test_audio(duration, 330 + i * 30, sample_rate)
                ref_path = tmpdir / f"reference_{i}.wav"
                sf.write(ref_path, ref_audio, sample_rate)

                output_path = tmpdir / f"output_{i}.wav"

                test_items.append(
                    BatchConversionItem(
                        source_audio=str(source_path),
                        reference_audio=str(ref_path),
                        output_path=str(output_path),
                        tau=0.3,
                    )
                )

            print(f"\nRunning pipelined batch conversion for {num_clones} items...")

            # Use pipelined batch API
            results, stats = voice_converter.convert_pipelined(
                test_items,
                num_workers=4,
                queue_depth=3,
            )

            print(f"\nPipeline Statistics:")
            print(f"  Total items: {stats.total_items}")
            print(f"  Successful: {stats.successful}")
            print(f"  Failed: {stats.failed}")
            print(f"  Wall time: {stats.wall_time:.2f}s")
            print(f"  Throughput: {stats.throughput:.2f} items/sec")
            print(f"  Avg latency: {stats.avg_latency*1000:.1f}ms")

            assert stats.successful >= num_clones, f"Only {stats.successful}/{num_clones} succeeded"

            print("\nPASS: Pipelined batch processing works")


def test_concurrent_stress(device, voice_converter):
    """
    Stress test with more than 10 concurrent requests.

    This goes beyond the minimum requirement to test system stability.
    """
    import soundfile as sf

    print("\n" + "=" * 60)
    print("Test: Concurrent Stress Test (20 clones)")
    print("=" * 60)

    num_clones = 20
    sample_rate = 22050
    duration = 1.5  # Shorter for stress test

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test files
        test_files = []
        for i in range(num_clones):
            source_audio = create_test_audio(duration, 440 + i * 25, sample_rate)
            source_path = tmpdir / f"source_{i}.wav"
            sf.write(source_path, source_audio, sample_rate)

            ref_audio = create_test_audio(duration, 330 + i * 15, sample_rate)
            ref_path = tmpdir / f"reference_{i}.wav"
            sf.write(ref_path, ref_audio, sample_rate)

            test_files.append(
                {
                    "source": source_path,
                    "reference": ref_path,
                    "index": i,
                }
            )

        # Pre-extract all embeddings
        print("Pre-extracting embeddings...")
        embeddings = {}
        for tf in test_files:
            embeddings[f"src_{tf['index']}"] = voice_converter.extract_se([str(tf["source"])])
            embeddings[f"ref_{tf['index']}"] = voice_converter.extract_se([str(tf["reference"])])

        results = []

        def run_clone(tf):
            idx = tf["index"]
            try:
                start = time.time()
                audio = voice_converter.convert(
                    source_audio=str(tf["source"]),
                    src_se=embeddings[f"src_{idx}"],
                    tgt_se=embeddings[f"ref_{idx}"],
                    tau=0.3,
                )
                return {"index": idx, "success": True, "time": time.time() - start}
            except Exception as e:
                return {"index": idx, "success": False, "error": str(e)}

        print(f"\nSubmitting {num_clones} concurrent requests...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_clones) as executor:
            futures = [executor.submit(run_clone, tf) for tf in test_files]
            for future in as_completed(futures):
                results.append(future.result())

        total_time = time.time() - start_time

        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        print(f"\nResults:")
        print(f"  Successful: {len(successful)}/{num_clones}")
        print(f"  Failed: {len(failed)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {len(successful)/total_time:.2f} clones/sec")

        if failed:
            print(f"\nFailure reasons:")
            for f in failed[:5]:  # Show first 5
                print(f"  Clone {f['index']}: {f.get('error', 'Unknown')[:60]}")

        # Stress test - expect at least 80% success rate
        success_rate = len(successful) / num_clones
        assert success_rate >= 0.8, f"Success rate {success_rate:.1%} below 80% threshold"

        print(f"\nPASS: Stress test completed with {success_rate:.1%} success rate")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
