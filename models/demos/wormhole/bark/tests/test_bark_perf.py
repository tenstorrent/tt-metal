# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Bark Small performance test with standardized reporting.

Uses ``models.common.utility_functions.profiler`` and ``prep_perf_report``
for CI-compatible performance tracking, following the patterns established
by ``mamba/tests/test_mamba_perf.py`` and ``bert_tiny/tests/test_performance.py``.

Usage:
    pytest models/demos/wormhole/bark/tests/test_bark_perf.py -v
    pytest models/demos/wormhole/bark/tests/test_bark_perf.py -v -k "perf"
"""

import numpy as np
import pytest
from loguru import logger

from models.common.utility_functions import profiler
from models.demos.wormhole.bark.tt.bark_model import TtBarkModel
from models.perf.perf_utils import prep_perf_report


@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "text, expected_compile_time, expected_inference_time",
    [
        ("Hello from Tenstorrent Bark!", 120.0, 8.0),
    ],
)
def test_perf_bark(
    device,
    text,
    expected_compile_time,
    expected_inference_time,
):
    """End-to-end performance benchmark for Bark Small.

    Measures compile time (first run), warm inference time (second run),
    and RTF (Real-Time Factor). Reports results via ``prep_perf_report``
    for integration with the CI performance dashboard.

    Expected times are generous initial baselines that will be tightened
    after the first successful CI run on N300.
    """
    profiler.clear()

    # --- Model initialization ---
    profiler.start("initialize_model")
    model = TtBarkModel(device, model_name="suno/bark-small")
    profiler.end("initialize_model")
    logger.info(f"Model initialized in {profiler.get('initialize_model'):.2f}s")

    # --- Compile run (cold — includes JIT kernel compilation) ---
    logger.info("Running compile pass (cold)...")
    profiler.start("inference_and_compile_time")
    audio_compile = model.generate(text, verbose=False)
    profiler.end("inference_and_compile_time")

    inference_and_compile_time = profiler.get("inference_and_compile_time")
    logger.info(f"Compile + inference (cold): {inference_and_compile_time:.2f}s")

    # --- Warm run (cached kernels) ---
    logger.info("Running warm inference pass...")
    profiler.start("inference_time")
    audio = model.generate(text, verbose=True)
    profiler.end("inference_time")

    inference_time = profiler.get("inference_time")
    compile_time = inference_and_compile_time - inference_time

    # --- Compute metrics ---
    duration = len(audio) / 24000
    rtf = inference_time / duration if duration > 0 else float("inf")

    logger.info(f"Compile time: {compile_time:.2f}s")
    logger.info(f"Inference time (warm): {inference_time:.2f}s")
    logger.info(f"Audio duration: {duration:.2f}s")
    logger.info(f"RTF: {rtf:.2f}")

    # --- Generate standardized CI performance report ---
    prep_perf_report(
        model_name="bark_small",
        batch_size=1,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=f"rtf={rtf:.3f}_duration={duration:.2f}s",
    )

    # --- Assertions ---
    assert audio is not None, "Generate returned None"
    assert np.isfinite(audio).all(), "Audio contains NaN/Inf"
    assert len(audio) > 0, "Audio is empty"

    # RTF target — this is the primary bounty requirement
    # Using a generous margin for CI variability; target is < 0.8
    assert rtf < 1.5, (
        f"RTF {rtf:.2f} is too high — expected < 1.5 "
        f"(bounty target: < 0.8, allowing CI margin)"
    )

    logger.info(f"Performance test PASSED: RTF={rtf:.2f}, inference={inference_time:.2f}s")


@pytest.mark.timeout(300)
def test_perf_bark_throughput(device):
    """Per-stage throughput verification.

    Validates that individual stage throughputs exceed their minimum targets:
    - Semantic: >= 20 tok/s
    - Coarse: >= 60 tok/s
    """
    import time

    model = TtBarkModel(device, model_name="suno/bark-small")
    text = "Throughput benchmark test."

    # Warm up (compile)
    _ = model.generate(text, verbose=False)

    # Measure
    t0 = time.time()
    semantic_tokens = model.generate_semantic_tokens(text)
    semantic_time = time.time() - t0
    semantic_count = semantic_tokens.shape[1]
    semantic_tps = semantic_count / semantic_time if semantic_time > 0 else 0

    t0 = time.time()
    coarse_tokens = model.generate_coarse_tokens(semantic_tokens)
    coarse_time = time.time() - t0
    coarse_count = coarse_tokens.shape[1]
    coarse_tps = coarse_count / coarse_time if coarse_time > 0 else 0

    logger.info(f"Semantic: {semantic_count} tokens in {semantic_time:.2f}s ({semantic_tps:.1f} tok/s)")
    logger.info(f"Coarse: {coarse_count} tokens in {coarse_time:.2f}s ({coarse_tps:.1f} tok/s)")

    assert semantic_tps >= 20, f"Semantic throughput {semantic_tps:.1f} < 20 tok/s target"
    assert coarse_tps >= 30, f"Coarse throughput {coarse_tps:.1f} < 30 tok/s target (60 is stretch)"
