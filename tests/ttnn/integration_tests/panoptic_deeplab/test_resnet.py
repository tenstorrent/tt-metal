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


def create_full_resnet_state_dict():
    """Create complete random state dict for ResNet with all layers."""
    state_dict = {}

    # Stem layers
    # Conv1: 3 -> 64
    state_dict["stem.conv1.weight"] = torch.randn(64, 3, 3, 3, dtype=torch.bfloat16)
    state_dict["stem.conv1.norm.weight"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["stem.conv1.norm.bias"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["stem.conv1.norm.running_mean"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["stem.conv1.norm.running_var"] = torch.abs(torch.randn(64, dtype=torch.bfloat16)) + 1e-5

    # Conv2: 64 -> 64
    state_dict["stem.conv2.weight"] = torch.randn(64, 64, 3, 3, dtype=torch.bfloat16)
    state_dict["stem.conv2.norm.weight"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["stem.conv2.norm.bias"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["stem.conv2.norm.running_mean"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["stem.conv2.norm.running_var"] = torch.abs(torch.randn(64, dtype=torch.bfloat16)) + 1e-5

    # Conv3: 64 -> 128
    state_dict["stem.conv3.weight"] = torch.randn(128, 64, 3, 3, dtype=torch.bfloat16)
    state_dict["stem.conv3.norm.weight"] = torch.randn(128, dtype=torch.bfloat16)
    state_dict["stem.conv3.norm.bias"] = torch.randn(128, dtype=torch.bfloat16)
    state_dict["stem.conv3.norm.running_mean"] = torch.randn(128, dtype=torch.bfloat16)
    state_dict["stem.conv3.norm.running_var"] = torch.abs(torch.randn(128, dtype=torch.bfloat16)) + 1e-5

    # Helper function to add bottleneck block weights
    def add_bottleneck_weights(prefix, in_channels, bottleneck_channels, out_channels, has_shortcut=False):
        # Conv1 (1x1)
        state_dict[f"{prefix}.conv1.weight"] = torch.randn(bottleneck_channels, in_channels, 1, 1, dtype=torch.bfloat16)
        state_dict[f"{prefix}.conv1.norm.weight"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
        state_dict[f"{prefix}.conv1.norm.bias"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
        state_dict[f"{prefix}.conv1.norm.running_mean"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
        state_dict[f"{prefix}.conv1.norm.running_var"] = (
            torch.abs(torch.randn(bottleneck_channels, dtype=torch.bfloat16)) + 1e-5
        )

        # Conv2 (3x3)
        state_dict[f"{prefix}.conv2.weight"] = torch.randn(
            bottleneck_channels, bottleneck_channels, 3, 3, dtype=torch.bfloat16
        )
        state_dict[f"{prefix}.conv2.norm.weight"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
        state_dict[f"{prefix}.conv2.norm.bias"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
        state_dict[f"{prefix}.conv2.norm.running_mean"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
        state_dict[f"{prefix}.conv2.norm.running_var"] = (
            torch.abs(torch.randn(bottleneck_channels, dtype=torch.bfloat16)) + 1e-5
        )

        # Conv3 (1x1)
        state_dict[f"{prefix}.conv3.weight"] = torch.randn(
            out_channels, bottleneck_channels, 1, 1, dtype=torch.bfloat16
        )
        state_dict[f"{prefix}.conv3.norm.weight"] = torch.randn(out_channels, dtype=torch.bfloat16)
        state_dict[f"{prefix}.conv3.norm.bias"] = torch.randn(out_channels, dtype=torch.bfloat16)
        state_dict[f"{prefix}.conv3.norm.running_mean"] = torch.randn(out_channels, dtype=torch.bfloat16)
        state_dict[f"{prefix}.conv3.norm.running_var"] = (
            torch.abs(torch.randn(out_channels, dtype=torch.bfloat16)) + 1e-5
        )

        # Shortcut (if needed)
        if has_shortcut:
            state_dict[f"{prefix}.shortcut.weight"] = torch.randn(out_channels, in_channels, 1, 1, dtype=torch.bfloat16)
            state_dict[f"{prefix}.shortcut.norm.weight"] = torch.randn(out_channels, dtype=torch.bfloat16)
            state_dict[f"{prefix}.shortcut.norm.bias"] = torch.randn(out_channels, dtype=torch.bfloat16)
            state_dict[f"{prefix}.shortcut.norm.running_mean"] = torch.randn(out_channels, dtype=torch.bfloat16)
            state_dict[f"{prefix}.shortcut.norm.running_var"] = (
                torch.abs(torch.randn(out_channels, dtype=torch.bfloat16)) + 1e-5
            )

    # res2 (3 blocks): 128->256 channels
    add_bottleneck_weights("res2.0", 128, 64, 256, has_shortcut=True)  # First block has shortcut
    add_bottleneck_weights("res2.1", 256, 64, 256, has_shortcut=False)
    add_bottleneck_weights("res2.2", 256, 64, 256, has_shortcut=False)

    # res3 (4 blocks): 256->512 channels
    add_bottleneck_weights("res3.0", 256, 128, 512, has_shortcut=True)  # First block has shortcut
    add_bottleneck_weights("res3.1", 512, 128, 512, has_shortcut=False)
    add_bottleneck_weights("res3.2", 512, 128, 512, has_shortcut=False)
    add_bottleneck_weights("res3.3", 512, 128, 512, has_shortcut=False)

    # res4 (6 blocks): 512->1024 channels
    add_bottleneck_weights("res4.0", 512, 256, 1024, has_shortcut=True)  # First block has shortcut
    add_bottleneck_weights("res4.1", 1024, 256, 1024, has_shortcut=False)
    add_bottleneck_weights("res4.2", 1024, 256, 1024, has_shortcut=False)
    add_bottleneck_weights("res4.3", 1024, 256, 1024, has_shortcut=False)
    add_bottleneck_weights("res4.4", 1024, 256, 1024, has_shortcut=False)
    add_bottleneck_weights("res4.5", 1024, 256, 1024, has_shortcut=False)

    # res5 (3 blocks): 1024->2048 channels
    add_bottleneck_weights("res5.0", 1024, 512, 2048, has_shortcut=True)  # First block has shortcut
    add_bottleneck_weights("res5.1", 2048, 512, 2048, has_shortcut=False)
    add_bottleneck_weights("res5.2", 2048, 512, 2048, has_shortcut=False)

    return state_dict


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
    [
        (128, 256),  # Small input for faster testing
        (256, 512),  # Standard input size
    ],
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
