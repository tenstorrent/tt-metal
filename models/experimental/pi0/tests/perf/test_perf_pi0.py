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
    noisy_actions = torch.randn(batch_size, config.action_horizon, config.action_dim, dtype=torch.float32)
    timestep = torch.rand(batch_size, dtype=torch.float32)
    
    return {
        "images": images,
        "img_masks": img_masks,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "state": state,
        "noisy_actions": noisy_actions,
        "timestep": timestep,
    }


def run_pi0_inference(
    device,
    batch_size: int = 1,
    inference_iterations: int = 10,
    warmup_iterations: int = 2,
):
    """
    Run PI0 TTNN inference benchmark.
    
    Args:
        device: TTNN device
        batch_size: Batch size for inference
        inference_iterations: Number of timed iterations
        warmup_iterations: Number of warmup iterations
    
    Returns:
        Average inference time in seconds
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
    
    # Warmup
    logger.info(f"Running {warmup_iterations} warmup iterations...")
    for _ in range(warmup_iterations):
        with torch.no_grad():
            _ = model.forward_training(
                images=inputs["images"],
                img_masks=inputs["img_masks"],
                lang_tokens=inputs["lang_tokens"],
                lang_masks=inputs["lang_masks"],
                state=inputs["state"],
                actions=inputs["noisy_actions"],
                timestep=inputs["timestep"],
            )
    ttnn.synchronize_device(device)
    
    # Timed runs
    logger.info(f"Running {inference_iterations} timed iterations...")
    t0 = time.time()
    for _ in range(inference_iterations):
        with torch.no_grad():
            output = model.forward_training(
                images=inputs["images"],
                img_masks=inputs["img_masks"],
                lang_tokens=inputs["lang_tokens"],
                lang_masks=inputs["lang_masks"],
                state=inputs["state"],
                actions=inputs["noisy_actions"],
                timestep=inputs["timestep"],
            )
    ttnn.synchronize_device(device)
    t1 = time.time()
    
    avg_time_sec = (t1 - t0) / inference_iterations
    avg_time_ms = avg_time_sec * 1000
    fps = batch_size / avg_time_sec
    
    logger.info(
        f"PI0 batch_size={batch_size}: "
        f"avg_time={avg_time_ms:.2f}ms, "
        f"FPS={fps:.2f}"
    )
    
    return avg_time_sec


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_pi0_inference_perf(device, batch_size):
    """
    Test PI0 inference performance.
    
    Runs multiple iterations and reports average time and FPS.
    """
    checkpoint_path = Path("/home/ubuntu/work/sdawle_pi0/torch_checkpoint/pi0_base")
    if not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")
    
    avg_time = run_pi0_inference(
        device,
        batch_size=batch_size,
        inference_iterations=5,
        warmup_iterations=1,
    )
    
    logger.info(f"PI0 Performance: {avg_time*1000:.2f}ms per inference")
    
    # Basic assertion - inference should complete
    assert avg_time > 0, "Inference time should be positive"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.models_performance_bare_metal
def test_pi0_e2e_perf_bare_metal(device, batch_size):
    """
    PI0 E2E performance test for bare metal runs.
    
    Used for CI performance tracking.
    """
    checkpoint_path = Path("/home/ubuntu/work/sdawle_pi0/torch_checkpoint/pi0_base")
    if not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")
    
    avg_time = run_pi0_inference(
        device,
        batch_size=batch_size,
        inference_iterations=10,
        warmup_iterations=2,
    )
    
    fps = batch_size / avg_time
    logger.info(f"PI0 E2E Performance: {avg_time*1000:.2f}ms, FPS: {fps:.2f}")
