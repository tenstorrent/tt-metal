# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end performance tests for PI0 with Denoising Loop Trace.

Tests the performant runner against baseline and measures improvement.

Usage:
    pytest models/experimental/pi0/tests/perf/test_e2e_performant.py -v

    # Or run directly
    python models/experimental/pi0/tests/perf/test_e2e_performant.py
"""

import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN
from models.experimental.pi0.runner.performant_runner import (
    PerformantRunner,
    PI0TraceConfig,
    create_device_for_performant_runner,
)


MODEL_PATH = Path(__file__).parent.parent.parent / "weights" / "pi0_base"


def create_test_inputs(config: PI0TraceConfig):
    """Create test inputs matching fixed configuration."""
    images = [torch.randn(config.batch_size, 3, config.image_size, config.image_size) for _ in range(config.num_images)]
    img_masks = [torch.ones(config.batch_size) for _ in range(config.num_images)]
    lang_tokens = torch.zeros(config.batch_size, config.max_lang_tokens, dtype=torch.long)
    lang_masks = torch.ones(config.batch_size, config.max_lang_tokens)
    state = torch.zeros(config.batch_size, config.state_dim)

    return images, img_masks, lang_tokens, lang_masks, state


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "num_command_queues": 2}], indirect=True)
def test_pi0_baseline_vs_performant(device):
    """Compare baseline vs performant runner performance."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model weights not found at {MODEL_PATH}")

    config = PI0TraceConfig()
    model = PI0ModelTTNN.from_pretrained(str(MODEL_PATH), device)

    images, img_masks, lang_tokens, lang_masks, state = create_test_inputs(config)

    # Baseline warmup and measurement
    logger.info("Running baseline warmup...")
    for _ in range(3):
        _ = model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
    ttnn.synchronize_device(device)

    logger.info("Measuring baseline latency...")
    num_iterations = 5
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
    ttnn.synchronize_device(device)
    baseline_time = (time.perf_counter() - start) / num_iterations * 1000

    # Performant runner
    logger.info("Compiling performant runner...")
    runner = PerformantRunner(model, device, config)
    runner.compile()

    logger.info("Running performant warmup...")
    for _ in range(3):
        _ = runner.execute(images, img_masks, lang_tokens, lang_masks, state)
    ttnn.synchronize_device(device)

    logger.info("Measuring performant latency...")
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = runner.execute(images, img_masks, lang_tokens, lang_masks, state)
    ttnn.synchronize_device(device)
    performant_time = (time.perf_counter() - start) / num_iterations * 1000

    runner.cleanup()

    improvement = (baseline_time - performant_time) / baseline_time * 100

    logger.info("=" * 60)
    logger.info("PI0 Performance Comparison (Denoising Loop Trace)")
    logger.info("=" * 60)
    logger.info(f"Baseline:          {baseline_time:.2f} ms")
    logger.info(f"Performant:        {performant_time:.2f} ms")
    logger.info(f"Improvement:       {improvement:.1f}%")
    logger.info("=" * 60)


def main():
    """Run performance tests directly."""
    logger.info("Creating device for performant runner...")
    config = PI0TraceConfig()
    device = create_device_for_performant_runner(config=config)

    if not MODEL_PATH.exists():
        logger.error(f"Model weights not found at {MODEL_PATH}")
        return

    logger.info("Loading PI0 model...")
    model = PI0ModelTTNN.from_pretrained(str(MODEL_PATH), device)

    images, img_masks, lang_tokens, lang_masks, state = create_test_inputs(config)

    # Baseline measurement
    logger.info("Measuring baseline...")
    for _ in range(3):
        _ = model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
    ttnn.synchronize_device(device)

    start = time.perf_counter()
    for _ in range(5):
        _ = model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
    ttnn.synchronize_device(device)
    baseline_ms = (time.perf_counter() - start) / 5 * 1000

    # Performant runner
    logger.info("Compiling performant runner (denoising loop trace)...")
    runner = PerformantRunner(model, device, config)
    runner.compile()

    logger.info("Measuring performant...")
    # Note: execute() doesn't take new inputs - all data is baked into trace
    for _ in range(3):
        _ = runner.execute()
    ttnn.synchronize_device(device)

    start = time.perf_counter()
    for _ in range(5):
        _ = runner.execute()
    ttnn.synchronize_device(device)
    performant_ms = (time.perf_counter() - start) / 5 * 1000

    runner.cleanup()

    # Report
    improvement = (baseline_ms - performant_ms) / baseline_ms * 100
    logger.info("=" * 60)
    logger.info("PI0 Performance (Denoising Loop Trace)")
    logger.info("=" * 60)
    logger.info(f"Baseline:    {baseline_ms:.2f} ms")
    logger.info(f"Performant:  {performant_ms:.2f} ms")
    logger.info(f"Improvement: {improvement:.1f}%")
    logger.info("=" * 60)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
