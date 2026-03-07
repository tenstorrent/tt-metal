# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test suite for Attention DenseUNet TTNN implementation.

Tests include:
- Individual component tests
- Full model forward pass
- PCC validation against PyTorch reference
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0, comp_pcc
from models.demos.attention_denseunet.reference.model import create_attention_denseunet
from models.demos.attention_denseunet.tt.common import (
    create_preprocessor,
    ATTENTION_DENSEUNET_L1_SMALL_SIZE,
    ATTENTION_DENSEUNET_TRACE_SIZE,
    ATTENTION_DENSEUNET_PCC,
)
from models.demos.attention_denseunet.tt.config import create_configs_from_parameters
from models.demos.attention_denseunet.tt.model import create_model_from_configs
from ttnn.model_preprocessing import preprocess_model_parameters


@run_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("resolution", [(256, 256)])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": ATTENTION_DENSEUNET_L1_SMALL_SIZE,
            "trace_region_size": ATTENTION_DENSEUNET_TRACE_SIZE,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
def test_attention_denseunet_inference(device: ttnn.Device, reset_seeds, batch_size: int, resolution: tuple):
    """
    Test full Attention DenseUNet inference and validate against PyTorch.
    
    This test:
    1. Creates PyTorch reference model
    2. Converts weights to TTNN format
    3. Runs inference on TTNN model
    4. Compares output with PyTorch using PCC
    """
    height, width = resolution
    
    logger.info(f"Testing Attention DenseUNet with batch_size={batch_size}, resolution={resolution}")
    logger.info("Creating PyTorch reference model...")
    reference_model = create_attention_denseunet()
    reference_model.eval()
    input_torch = torch.randn(batch_size, 3, height, width)
    logger.info("Running PyTorch inference...")
    with torch.no_grad():
        output_torch = reference_model(input_torch)
    
    logger.info(f"PyTorch output shape: {output_torch.shape}")
    logger.info("Preprocessing model parameters for TTNN...")
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_preprocessor(device),
        device=None,
    )
    logger.info("Creating TTNN configuration...")
    configs = create_configs_from_parameters(
        parameters=parameters,
        in_channels=3,
        out_channels=1,
        input_height=height,
        input_width=width,
        batch_size=batch_size,
    )
    logger.info("Creating TTNN model...")
    ttnn_model = create_model_from_configs(configs, device)
    logger.info("Preparing TTNN input...")
    ttnn_input = input_torch.reshape(batch_size, 1, 3, height * width)
    ttnn_input = ttnn.from_torch(
        ttnn_input,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=configs.l1_input_memory_config,
    )
    logger.info("Running TTNN inference...")
    output_ttnn = ttnn_model(ttnn_input)
    output_ttnn_torch = ttnn.to_torch(output_ttnn)
    output_ttnn_torch = output_ttnn_torch.reshape(batch_size, 1, height, width)
    logger.info(f"TTNN output shape: {output_ttnn_torch.shape}")
    assert output_torch.shape == output_ttnn_torch.shape, \
        f"Shape mismatch: PyTorch {output_torch.shape} vs TTNN {output_ttnn_torch.shape}"
    pcc_value = comp_pcc(output_torch, output_ttnn_torch)
    logger.info(f"PCC: {pcc_value}")
    assert pcc_value >= ATTENTION_DENSEUNET_PCC, \
        f"PCC {pcc_value} is below threshold {ATTENTION_DENSEUNET_PCC}"
    
    logger.info("✓ Test passed!")


@run_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": ATTENTION_DENSEUNET_L1_SMALL_SIZE,
            "trace_region_size": ATTENTION_DENSEUNET_TRACE_SIZE,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
def test_model_initialization(device: ttnn.Device, reset_seeds, batch_size: int):
    """
    Test that model can be initialized without errors.
    """
    logger.info("Testing model initialization...")
    reference_model = create_attention_denseunet()
    reference_model.eval()
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_preprocessor(device),
        device=None,
    )
    configs = create_configs_from_parameters(
        parameters=parameters,
        in_channels=3,
        out_channels=1,
        input_height=256,
        input_width=256,
        batch_size=batch_size,
    )
    ttnn_model = create_model_from_configs(configs, device)
    assert ttnn_model is not None
    assert ttnn_model.device == device
    logger.info("✓ Model initialization test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
