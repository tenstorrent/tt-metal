# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PI0 Performance Test

Benchmarks PI0 inference performance on TTNN.
Similar to segformer's test_perf_device_segformer.py
"""

import time
import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

# =============================================================================
# CONFIGURABLE PARAMETERS - Modify these to control test behavior
# =============================================================================
NUM_INFERENCE_ITERATIONS = 50   # Number of timed inference runs (results averaged)
NUM_WARMUP_ITERATIONS = 2      # Number of warmup runs before timing
BATCH_SIZE = 1                 # Batch size for inference
# =============================================================================

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from models.experimental.pi0.ttnn_pi0_reference.ttnn_pi0 import (
    PI0ModelTTNN,
    PI0ModelConfig,
)
from models.experimental.pi0.ttnn_pi0_reference.weight_loader import PI0WeightLoader
from models.experimental.pi0.ttnn_pi0_reference.ttnn_siglip import SigLIPConfig


def create_pi0_config() -> PI0ModelConfig:
    """Create PI0 model configuration."""
    config = PI0ModelConfig(
        action_dim=32,
        action_horizon=50,
        state_dim=32,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=False,
    )
    
    config.siglip_config = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    
    return config


def create_test_inputs(config: PI0ModelConfig, batch_size: int = 1):
    """Create test inputs for inference."""
    image_size = config.siglip_config.image_size

    images = [
        torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32)
        for _ in range(2)
    ]
    img_masks = [torch.ones(batch_size, dtype=torch.bool) for _ in range(2)]

    lang_tokens = torch.randint(0, 256000, (batch_size, 32))
    lang_masks = torch.ones(batch_size, 32, dtype=torch.bool)

    state = torch.randn(batch_size, config.state_dim, dtype=torch.float32)

    return {
        "images": images,
        "img_masks": img_masks,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "state": state,
    }


def run_pi0_inference(
    device,
    batch_size: int = BATCH_SIZE,
    inference_iterations: int = NUM_INFERENCE_ITERATIONS,
    warmup_iterations: int = NUM_WARMUP_ITERATIONS,
):
    """
    Run PI0 TTNN inference benchmark.

    Args:
        device: TTNN device
        batch_size: Batch size for inference
        inference_iterations: Number of timed iterations
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary with timing results
    """
    checkpoint_path = Path("/home/ubuntu/work/sdawle_pi0/torch_checkpoint/pi0_base")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create config and inputs
    config = create_pi0_config()
    inputs = create_test_inputs(config, batch_size=batch_size)

    # Load model
    weight_loader = PI0WeightLoader(str(checkpoint_path))
    model = PI0ModelTTNN(config, weight_loader, device)

    # Warmup with sample_actions (full 10-step denoising)
    logger.info(f"Running {warmup_iterations} warmup iterations...")
    for i in range(warmup_iterations):
        with torch.no_grad():
            _ = model.sample_actions(
                images=inputs["images"],
                img_masks=inputs["img_masks"],
                lang_tokens=inputs["lang_tokens"],
                lang_masks=inputs["lang_masks"],
                state=inputs["state"],
            )
        ttnn.synchronize_device(device)
        logger.info(f"  Warmup {i+1}/{warmup_iterations} complete")

    # Timed runs with sample_actions (full inference)
    logger.info(f"Running {inference_iterations} timed iterations...")
    iteration_times = []

    for i in range(inference_iterations):
        ttnn.synchronize_device(device)
        t0 = time.time()
        with torch.no_grad():
            output = model.sample_actions(
                images=inputs["images"],
                img_masks=inputs["img_masks"],
                lang_tokens=inputs["lang_tokens"],
                lang_masks=inputs["lang_masks"],
                state=inputs["state"],
            )
        ttnn.synchronize_device(device)
        t1 = time.time()

        iter_time_ms = (t1 - t0) * 1000
        iteration_times.append(iter_time_ms)
        logger.info(f"  Iteration {i+1}/{inference_iterations}: {iter_time_ms:.2f}ms")

    # Calculate statistics
    avg_time_ms = sum(iteration_times) / len(iteration_times)
    min_time_ms = min(iteration_times)
    max_time_ms = max(iteration_times)
    avg_time_sec = avg_time_ms / 1000
    fps = batch_size / avg_time_sec

    # Calculate standard deviation
    variance = sum((t - avg_time_ms) ** 2 for t in iteration_times) / len(iteration_times)
    std_dev_ms = variance ** 0.5

    results = {
        "batch_size": batch_size,
        "num_iterations": inference_iterations,
        "num_warmup": warmup_iterations,
        "avg_time_ms": avg_time_ms,
        "min_time_ms": min_time_ms,
        "max_time_ms": max_time_ms,
        "std_dev_ms": std_dev_ms,
        "fps": fps,
        "iteration_times": iteration_times,
    }

    return results


def print_performance_summary(results: dict):
    """Print a formatted performance summary."""
    print("\n" + "=" * 70)
    print("                    PI0 PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"  Model:              PI0 (sample_actions with 10 denoising steps)")
    print(f"  Batch Size:         {results['batch_size']}")
    print(f"  Warmup Iterations:  {results['num_warmup']}")
    print(f"  Timed Iterations:   {results['num_iterations']}")
    print("-" * 70)
    print("  TIMING RESULTS:")
    print(f"    Average Time:     {results['avg_time_ms']:.2f} ms")
    print(f"    Min Time:         {results['min_time_ms']:.2f} ms")
    print(f"    Max Time:         {results['max_time_ms']:.2f} ms")
    print(f"    Std Dev:          {results['std_dev_ms']:.2f} ms")
    print("-" * 70)
    print("  THROUGHPUT:")
    print(f"    FPS:              {results['fps']:.2f}")
    print(f"    Time per action:  {results['avg_time_ms']/10:.2f} ms (per denoising step)")
    print("-" * 70)
    print("  PER-ITERATION TIMES (ms):")
    for i, t in enumerate(results['iteration_times']):
        print(f"    Run {i+1}: {t:.2f}")
    print("=" * 70 + "\n")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pi0_inference_perf(device):
    """
    Test PI0 inference performance.

    Runs multiple iterations and reports average time and FPS.

    Configure the test by modifying the constants at the top of this file:
        - NUM_INFERENCE_ITERATIONS: Number of timed runs (default: 5)
        - NUM_WARMUP_ITERATIONS: Number of warmup runs (default: 1)
        - BATCH_SIZE: Batch size for inference (default: 1)
    """
    checkpoint_path = Path("/home/ubuntu/work/sdawle_pi0/torch_checkpoint/pi0_base")
    if not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    print(f"\n{'='*70}")
    print(f"  Starting PI0 Performance Test")
    print(f"  Configuration:")
    print(f"    - Inference iterations: {NUM_INFERENCE_ITERATIONS}")
    print(f"    - Warmup iterations: {NUM_WARMUP_ITERATIONS}")
    print(f"    - Batch size: {BATCH_SIZE}")
    print(f"{'='*70}\n")

    results = run_pi0_inference(
        device,
        batch_size=BATCH_SIZE,
        inference_iterations=NUM_INFERENCE_ITERATIONS,
        warmup_iterations=NUM_WARMUP_ITERATIONS,
    )

    # Print detailed summary
    print_performance_summary(results)

    # Basic assertion - inference should complete
    assert results["avg_time_ms"] > 0, "Inference time should be positive"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.models_performance_bare_metal
def test_pi0_e2e_perf_bare_metal(device):
    """
    PI0 E2E performance test for bare metal runs.

    Used for CI performance tracking.

    Configure the test by modifying the constants at the top of this file:
        - NUM_INFERENCE_ITERATIONS: Number of timed runs (default: 5)
        - NUM_WARMUP_ITERATIONS: Number of warmup runs (default: 1)
        - BATCH_SIZE: Batch size for inference (default: 1)
    """
    checkpoint_path = Path("/home/ubuntu/work/sdawle_pi0/torch_checkpoint/pi0_base")
    if not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    print(f"\n{'='*70}")
    print(f"  Starting PI0 E2E Performance Test (Bare Metal)")
    print(f"  Configuration:")
    print(f"    - Inference iterations: {NUM_INFERENCE_ITERATIONS}")
    print(f"    - Warmup iterations: {NUM_WARMUP_ITERATIONS}")
    print(f"    - Batch size: {BATCH_SIZE}")
    print(f"{'='*70}\n")

    results = run_pi0_inference(
        device,
        batch_size=BATCH_SIZE,
        inference_iterations=NUM_INFERENCE_ITERATIONS,
        warmup_iterations=NUM_WARMUP_ITERATIONS,
    )

    # Print detailed summary
    print_performance_summary(results)

    # Basic assertion
    assert results["avg_time_ms"] > 0, "Inference time should be positive"
