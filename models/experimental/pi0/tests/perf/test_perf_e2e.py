# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import time
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# PyTorch reference implementation
from models.experimental.pi0.reference.torch_pi0_model import PI0Model as PI0ModelTorch

# TTNN implementation
from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN

# Shared configs and weight loader
from models.experimental.pi0.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
)


# =============================================================================
# CONFIGURATION
# =============================================================================
TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise EnvironmentError("TT_METAL_HOME environment variable is not set")
CHECKPOINT_PATH = os.path.join(TT_METAL_HOME, "models/experimental/pi0/weights/pi0_base")
SEED = 42
PCC_THRESHOLD = 0.93  # Account for BFloat16 precision and hardware non-determinism


def create_pi0_pipeline_model(ttnn_model, device, inputs):
    """Create a wrapper function for the pi0 model to use with pipeline"""

    def run(dummy_input):
        with torch.no_grad():
            output_dict = ttnn_model.sample_actions(
                images=inputs["images"],
                img_masks=inputs["img_masks"],
                lang_tokens=inputs["lang_tokens"],
                lang_masks=inputs["lang_masks"],
                state=inputs["state"],
            )
        return output_dict

    return run


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


def compute_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    t1 = tensor1.flatten().float()
    t2 = tensor2.flatten().float()

    mean1, mean2 = torch.mean(t1), torch.mean(t2)
    std1, std2 = torch.std(t1), torch.std(t2)

    if std1 == 0 or std2 == 0:
        return 1.0 if torch.allclose(t1, t2) else 0.0

    covariance = torch.mean((t1 - mean1) * (t2 - mean2))
    return (covariance / (std1 * std2)).item()


# =============================================================================
# PYTEST TEST FUNCTION
# =============================================================================


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 16384,
            "trace_region_size": 8000000,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [32])
@pytest.mark.parametrize(
    "batch_size, expected_compile_time, expected_throughput_fps",
    [(1, 30.0, 40)],
)
def test_perf_pi0_ttnn(device, num_iterations, batch_size, expected_compile_time, expected_throughput_fps):
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    # Create config and inputs
    config = create_config()
    inputs = create_test_inputs(config, batch_size=batch_size)

    # Load weights
    weight_loader = PI0WeightLoader(str(checkpoint_path))

    # Initialize models
    model_torch = PI0ModelTorch(config, weight_loader)
    model_ttnn = PI0ModelTTNN(config, weight_loader, device)

    run_model = create_pi0_pipeline_model(model_ttnn, device, inputs)

    tt_host_tensor = inputs

    config = PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False)
    pipeline = create_pipeline_from_config(
        config,
        run_model,
        device,
        dram_input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        l1_input_memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    host_inputs = [tt_host_tensor] * num_iterations

    start = time.time()
    pipeline.compile(tt_host_tensor)
    end = time.time()
    compile_time = end - start

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    start = time.time()
    outputs = pipeline.enqueue(host_inputs).pop_all()
    end = time.time()

    pipeline.cleanup()

    inference_time = (end - start) / num_iterations
    logger.info(f"Average model time={1000.0 * inference_time:.2f} ms")
    logger.info(f"Average model performance={num_iterations * batch_size / (end - start):.2f} fps")

    prep_perf_report(
        model_name="pi0-2cq",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=batch_size / expected_throughput_fps,
        comments=f"batch_{batch_size}",
    )
