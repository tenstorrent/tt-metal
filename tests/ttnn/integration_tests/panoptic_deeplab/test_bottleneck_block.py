# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from loguru import logger

from models.experimental.panoptic_deeplab.reference.pytorch_bottleneck import BottleneckBlock
from models.experimental.panoptic_deeplab.tt.resnet.tt_bottleneck import TtBottleneck


def create_random_state_dict(in_channels, bottleneck_channels, out_channels, has_shortcut=False):
    """Create random state dict for BottleneckBlock."""
    state_dict = {}

    # Conv1 weights (1x1 conv)
    state_dict["conv1.weight"] = torch.randn(bottleneck_channels, in_channels, 1, 1, dtype=torch.bfloat16)
    state_dict["conv1.norm.weight"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv1.norm.bias"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv1.norm.running_mean"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv1.norm.running_var"] = torch.abs(torch.randn(bottleneck_channels, dtype=torch.bfloat16)) + 1e-5

    # Conv2 weights (3x3 conv)
    state_dict["conv2.weight"] = torch.randn(bottleneck_channels, bottleneck_channels, 3, 3, dtype=torch.bfloat16)
    state_dict["conv2.norm.weight"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv2.norm.bias"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv2.norm.running_mean"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv2.norm.running_var"] = torch.abs(torch.randn(bottleneck_channels, dtype=torch.bfloat16)) + 1e-5

    # Conv3 weights (1x1 conv)
    state_dict["conv3.weight"] = torch.randn(out_channels, bottleneck_channels, 1, 1, dtype=torch.bfloat16)
    state_dict["conv3.norm.weight"] = torch.randn(out_channels, dtype=torch.bfloat16)
    state_dict["conv3.norm.bias"] = torch.randn(out_channels, dtype=torch.bfloat16)
    state_dict["conv3.norm.running_mean"] = torch.randn(out_channels, dtype=torch.bfloat16)
    state_dict["conv3.norm.running_var"] = torch.abs(torch.randn(out_channels, dtype=torch.bfloat16)) + 1e-5

    # Shortcut weights (if needed)
    if has_shortcut:
        state_dict["shortcut.weight"] = torch.randn(out_channels, in_channels, 1, 1, dtype=torch.bfloat16)
        state_dict["shortcut.norm.weight"] = torch.randn(out_channels, dtype=torch.bfloat16)
        state_dict["shortcut.norm.bias"] = torch.randn(out_channels, dtype=torch.bfloat16)
        state_dict["shortcut.norm.running_mean"] = torch.randn(out_channels, dtype=torch.bfloat16)
        state_dict["shortcut.norm.running_var"] = torch.abs(torch.randn(out_channels, dtype=torch.bfloat16)) + 1e-5

    return state_dict


def create_bottleneck_models(
    in_channels,
    bottleneck_channels,
    out_channels,
    stride=1,
    dilation=1,
    has_shortcut=False,
    shortcut_stride=1,
    device=None,
):
    """Create both PyTorch and TTNN BottleneckBlock models with same weights."""

    # Create random state dict
    state_dict = create_random_state_dict(in_channels, bottleneck_channels, out_channels, has_shortcut)

    # Create PyTorch model
    pytorch_model = BottleneckBlock(
        in_channels=in_channels,
        bottleneck_channels=bottleneck_channels,
        out_channels=out_channels,
        stride=stride,
        dilation=dilation,
        has_shortcut=has_shortcut,
        shortcut_stride=shortcut_stride,
    )

    # Load state dict into PyTorch model
    pytorch_model.load_state_dict(state_dict)
    pytorch_model.eval()

    # Convert model to bfloat16 to match input type
    pytorch_model = pytorch_model.to(torch.bfloat16)

    # Create TTNN model
    if device:
        ttnn_model = TtBottleneck(device=device, state_dict=state_dict, dtype=ttnn.bfloat16, has_shortcut=has_shortcut)
    else:
        ttnn_model = None

    return pytorch_model, ttnn_model, state_dict


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height", [64])
@pytest.mark.parametrize("width", [128])
@pytest.mark.parametrize(
    "in_channels,bottleneck_channels,out_channels,has_shortcut,stride,dilation",
    [
        # Test case 1: Block with shortcut (like first block in res2)
        (128, 64, 256, True, 1, 1),
        # Test case 2: Block without shortcut (like subsequent blocks)
        (256, 64, 256, False, 1, 1),
        # Test case 3: Block with shortcut and stride=2 (like first block in res3)
        (256, 128, 512, True, 2, 1),
        # Test case 4: Block with dilation (like blocks in res5)
        (2048, 512, 2048, False, 1, 2),
    ],
)
def test_bottleneck_block_pcc(
    device,
    batch_size,
    height,
    width,
    in_channels,
    bottleneck_channels,
    out_channels,
    has_shortcut,
    stride,
    dilation,
    reset_seeds,
):
    """Test that TTNN and PyTorch BottleneckBlock implementations produce similar outputs using PCC."""

    torch.manual_seed(0)

    # Create models
    shortcut_stride = stride if has_shortcut else 1
    pytorch_model, ttnn_model, state_dict = create_bottleneck_models(
        in_channels=in_channels,
        bottleneck_channels=bottleneck_channels,
        out_channels=out_channels,
        stride=stride,
        dilation=dilation,
        has_shortcut=has_shortcut,
        shortcut_stride=shortcut_stride,
        device=device,
    )

    # Create input tensor
    torch_input = torch.randn(batch_size, in_channels, height, width, dtype=torch.bfloat16)

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
    logger.info(f"BottleneckBlock PCC test - {pcc_message}")

    # Log test details
    logger.info(
        f"Test config: in_channels={in_channels}, bottleneck_channels={bottleneck_channels}, "
        f"out_channels={out_channels}, has_shortcut={has_shortcut}, stride={stride}, dilation={dilation}"
    )
    logger.info(f"Input shape: {torch_input.shape}")
    logger.info(f"PyTorch output shape: {torch_output.shape}")
    logger.info(f"TTNN output shape: {ttnn_output_torch.shape}")
