# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from loguru import logger

from models.experimental.panoptic_deeplab.reference.pytorch_resnet import ResNet
from models.experimental.panoptic_deeplab.tt.resnet.tt_resnet import TtResNet
from models.experimental.panoptic_deeplab.tt.common import (
    create_random_stem_state_dict,
    create_random_bottleneck_state_dict,
    create_full_resnet_state_dict,
)


def create_resnet_models(device=None):
    """Create both PyTorch and TTNN ResNet models with same weights."""

    # Create random state dict
    state_dict = create_full_resnet_state_dict()

    # Create PyTorch model
    pytorch_model = ResNet()

    # Load state dict into PyTorch model (need to handle the nested structure)
    pytorch_state_dict = {}
    for key, value in state_dict.items():
        # Convert TTNN-style keys to PyTorch-style keys
        if key.startswith("stem."):
            pytorch_key = key  # stem keys are the same
        elif key.startswith("res"):
            # Convert res2.0.conv1.weight to res2.0.conv1.weight (same)
            pytorch_key = key
        else:
            pytorch_key = key
        pytorch_state_dict[pytorch_key] = value

    pytorch_model.load_state_dict(pytorch_state_dict)
    pytorch_model.eval()

    # Convert model to bfloat16 to match input type
    pytorch_model = pytorch_model.to(torch.bfloat16)

    # Create TTNN model
    if device:
        ttnn_model = TtResNet(device=device, state_dict=state_dict, dtype=ttnn.bfloat16)
    else:
        ttnn_model = None

    return pytorch_model, ttnn_model, state_dict


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "height,width",
    [(512, 1024)],
)
def test_resnet_pcc(device, batch_size, height, width, reset_seeds):
    """Test that TTNN and PyTorch ResNet implementations produce similar outputs using PCC."""

    torch.manual_seed(0)

    # Create models
    pytorch_model, ttnn_model, state_dict = create_resnet_models(device=device)

    # Create input tensor (RGB image)
    torch_input = torch.randn(batch_size, 3, height, width, dtype=torch.bfloat16)

    # Run PyTorch model
    with torch.no_grad():
        torch_outputs = pytorch_model(torch_input)

    # Convert input to TTNN format (NHWC)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),  # Convert NCHW to NHWC
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run TTNN model
    ttnn_outputs = ttnn_model(ttnn_input)

    # Convert TTNN outputs back to torch (NHWC -> NCHW) and compare each layer
    test_results = {}
    for layer_name in ["res2", "res3", "res4", "res5"]:
        torch_output = torch_outputs[layer_name]
        ttnn_output = ttnn_outputs[layer_name]
        ttnn_output_torch = ttnn.to_torch(ttnn_output).permute(0, 3, 1, 2)

        # Compare outputs using PCC (more relaxed for full ResNet)
        pcc_passed, pcc_message = assert_with_pcc(torch_output, ttnn_output_torch, 0.95)
        test_results[layer_name] = (pcc_passed, pcc_message)

        logger.info(f"ResNet {layer_name} PCC test - {pcc_message}")
        logger.info(f"{layer_name} - PyTorch shape: {torch_output.shape}, TTNN shape: {ttnn_output_torch.shape}")

    # Log overall test details
    logger.info(f"Input shape: {torch_input.shape}")
    logger.info(f"All layers passed PCC test: {all(result[0] for result in test_results.values())}")
