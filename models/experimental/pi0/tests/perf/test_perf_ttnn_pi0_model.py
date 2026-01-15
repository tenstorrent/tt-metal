# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PI0 Performance Test - TTNN (tt/)

This test benchmarks the TTNN implementation from tt/ directory.

Config:
    - Checkpoint: $TT_METAL_HOME/models/experimental/pi0/weights/pi0_base
    - Full denoising: 10 steps
    - Batch size: 1
    - Warmup iterations: 2
    - Inference iterations: 10

Usage:
    pytest test_perf_ttnn_pi0_model.py::test_pi0_inference_perf_ttnn -v -s
"""

import sys
import os
import time
from pathlib import Path
from typing import List

import pytest
import torch
import ttnn

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# TTNN implementation
from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN

# Shared configs and weight loader
from models.experimental.pi0.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader


# =============================================================================
# CONFIGURATION
# =============================================================================
TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise EnvironmentError("TT_METAL_HOME environment variable is not set")
CHECKPOINT_PATH = os.path.join(TT_METAL_HOME, "models/experimental/pi0/weights/pi0_base")
BATCH_SIZE = 1
SEED = 42
NUM_INFERENCE_ITERATIONS = 50
NUM_WARMUP_ITERATIONS = 2


def create_config() -> PI0ModelConfig:
    """Create PI0ModelConfig with correct dimensions."""
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
    """Create test inputs."""
    image_size = config.siglip_config.image_size

    images = [torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32) for _ in range(2)]
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
    """Run PI0 inference and measure performance."""
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create config and inputs
    config = create_config()
    inputs = create_test_inputs(config, batch_size)

    # Load model
    print(f"\n📋 Loading PI0 TTNN model (tt/)...")
    weight_loader = PI0WeightLoader(str(checkpoint_path))
    model = PI0ModelTTNN(config, weight_loader, device)
    print(f"✅ Model loaded")

    # Warmup
    print(f"\n🔥 Warmup ({warmup_iterations} iterations)...")
    for i in range(warmup_iterations):
        torch.manual_seed(SEED + i)
        with torch.no_grad():
            _ = model.sample_actions(
                images=inputs["images"],
                img_masks=inputs["img_masks"],
                lang_tokens=inputs["lang_tokens"],
                lang_masks=inputs["lang_masks"],
                state=inputs["state"],
            )

    # Measure performance
    print(f"\n⏱️  Measuring ({inference_iterations} iterations)...")
    times: List[float] = []

    for i in range(inference_iterations):
        torch.manual_seed(SEED + i)

        start = time.perf_counter()
        with torch.no_grad():
            output = model.sample_actions(
                images=inputs["images"],
                img_masks=inputs["img_masks"],
                lang_tokens=inputs["lang_tokens"],
                lang_masks=inputs["lang_masks"],
                state=inputs["state"],
            )
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000
        times.append(elapsed_ms)
        print(f"   Iter {i + 1}: {elapsed_ms:.2f} ms")

    # Compute statistics
    times_tensor = torch.tensor(times)
    avg_time = times_tensor.mean().item()
    min_time = times_tensor.min().item()
    max_time = times_tensor.max().item()
    std_time = times_tensor.std().item()
    fps = 1000.0 / avg_time if avg_time > 0 else 0

    # Print summary
    print("\n" + "=" * 80)
    print("  PI0 PERFORMANCE SUMMARY (tt/)")
    print("=" * 80)
    print(f"   Batch size:  {batch_size}")
    print(f"   Iterations:  {inference_iterations}")
    print(f"   Warmup:      {warmup_iterations}")
    print("-" * 80)
    print(f"   Average:     {avg_time:.2f} ms")
    print(f"   Min:         {min_time:.2f} ms")
    print(f"   Max:         {max_time:.2f} ms")
    print(f"   Std Dev:     {std_time:.2f} ms")
    print(f"   FPS:         {fps:.2f}")
    print("=" * 80)

    return {
        "avg_time_ms": avg_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "std_time_ms": std_time,
        "fps": fps,
        "output_shape": list(output.shape),
    }


# =============================================================================
# PYTEST TEST FUNCTION
# =============================================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pi0_inference_perf_ttnn(device):
    """
    Pytest: PI0 inference performance test using tt/ structure.
    """
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    results = run_pi0_inference(
        device,
        batch_size=BATCH_SIZE,
        inference_iterations=NUM_INFERENCE_ITERATIONS,
        warmup_iterations=NUM_WARMUP_ITERATIONS,
    )

    # Basic sanity check
    assert results["avg_time_ms"] > 0, "Average time should be positive"
    assert results["fps"] > 0, "FPS should be positive"
    assert results["output_shape"] == [BATCH_SIZE, 50, 32], f"Unexpected output shape: {results['output_shape']}"


# =============================================================================
# STANDALONE RUNNER
# =============================================================================


def main():
    """Run performance test standalone."""
    print("=" * 80)
    print("  PI0 TTNN PERFORMANCE TEST")
    print("  TTNN (tt/)")
    print("=" * 80)

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        return 1

    print(f"\n📁 Checkpoint: {checkpoint_path}")

    # Open device
    print("\n🔌 Opening TTNN device...")
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    print(f"✅ Device opened (grid: {grid.x}x{grid.y})")

    try:
        _ = run_pi0_inference(
            device,
            batch_size=BATCH_SIZE,
            inference_iterations=NUM_INFERENCE_ITERATIONS,
            warmup_iterations=NUM_WARMUP_ITERATIONS,
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        print("\n🔌 Closing device...")
        ttnn.close_device(device)
        print("✅ Device closed")


if __name__ == "__main__":
    sys.exit(main())
