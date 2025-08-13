# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from loguru import logger

from models.experimental.panoptic_deeplab.reference.pytorch_stem import DeepLabStem
from models.experimental.panoptic_deeplab.tt.resnet.tt_stem import TtStem


def create_random_stem_state_dict():
    """Create random state dict for DeepLabStem."""
    state_dict = {}

    # Conv1: 3 -> 64 channels, 3x3 kernel, stride=2
    state_dict["conv1.weight"] = torch.randn(64, 3, 3, 3, dtype=torch.bfloat16)
    state_dict["conv1.norm.weight"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv1.norm.bias"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv1.norm.running_mean"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv1.norm.running_var"] = torch.abs(torch.randn(64, dtype=torch.bfloat16)) + 1e-5

    # Conv2: 64 -> 64 channels, 3x3 kernel, stride=1
    state_dict["conv2.weight"] = torch.randn(64, 64, 3, 3, dtype=torch.bfloat16)
    state_dict["conv2.norm.weight"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv2.norm.bias"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv2.norm.running_mean"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv2.norm.running_var"] = torch.abs(torch.randn(64, dtype=torch.bfloat16)) + 1e-5

    # Conv3: 64 -> 128 channels, 3x3 kernel, stride=1
    state_dict["conv3.weight"] = torch.randn(128, 64, 3, 3, dtype=torch.bfloat16)
    state_dict["conv3.norm.weight"] = torch.randn(128, dtype=torch.bfloat16)
    state_dict["conv3.norm.bias"] = torch.randn(128, dtype=torch.bfloat16)
    state_dict["conv3.norm.running_mean"] = torch.randn(128, dtype=torch.bfloat16)
    state_dict["conv3.norm.running_var"] = torch.abs(torch.randn(128, dtype=torch.bfloat16)) + 1e-5

    return state_dict


def create_stem_models(device=None):
    """Create both PyTorch and TTNN DeepLabStem models with same weights."""

    # Create random state dict
    state_dict = create_random_stem_state_dict()

    # Create PyTorch model
    pytorch_model = DeepLabStem()

    # Load state dict into PyTorch model
    pytorch_model.load_state_dict(state_dict)
    pytorch_model.eval()

    # Convert model to bfloat16 to match input type
    pytorch_model = pytorch_model.to(torch.bfloat16)

    # Create TTNN model
    if device:
        ttnn_model = TtStem(device=device, state_dict=state_dict, dtype=ttnn.bfloat16)
    else:
        ttnn_model = None

    return pytorch_model, ttnn_model, state_dict


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "height,width",
    [
        (256, 512),  # Standard input size
        (128, 256),  # Smaller input size
        (512, 1024),  # Larger input size
    ],
)
def test_stem_pcc(device, batch_size, height, width, reset_seeds):
    """Test that TTNN and PyTorch DeepLabStem implementations produce similar outputs using PCC."""

    torch.manual_seed(0)

    # Create models
    pytorch_model, ttnn_model, state_dict = create_stem_models(device=device)

    # Create input tensor (RGB image)
    torch_input = torch.randn(batch_size, 3, height, width, dtype=torch.bfloat16)

    # Run PyTorch model
    with torch.no_grad():
        torch_output = pytorch_model(torch_input)

    # Convert input to TTNN format (NHWC)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),  # Convert NCHW to NHWC
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run TTNN model
    ttnn_output = ttnn_model(ttnn_input)

    # Convert TTNN output back to torch (NHWC -> NCHW)
    ttnn_output_torch = ttnn.to_torch(ttnn_output).permute(0, 3, 1, 2)

    # Compare outputs using PCC
    pcc_passed, pcc_message = assert_with_pcc(torch_output, ttnn_output_torch, 0.98)
    logger.info(f"DeepLabStem PCC test - {pcc_message}")

    # Log test details
    logger.info(f"Input shape: {torch_input.shape}")
    logger.info(f"PyTorch output shape: {torch_output.shape}")
    logger.info(f"TTNN output shape: {ttnn_output_torch.shape}")
