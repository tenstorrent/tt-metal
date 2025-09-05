# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
ResNet Tests for Panoptic DeepLab using real weights from R-52.pkl.
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from loguru import logger

from models.experimental.panoptic_deeplab.reference.pytorch_resnet import ResNet
from models.experimental.panoptic_deeplab.tt.tt_resnet import TtResNet
from models.experimental.panoptic_deeplab.tt.common import create_resnet_state_dict


def create_resnet_models(device=None):
    """Create both PyTorch and TTNN ResNet models.

    Args:
        device: TTNN device for model creation
    """
    state_dict = create_resnet_state_dict()

    pytorch_model = ResNet()

    pytorch_state_dict = {}
    for key, value in state_dict.items():
        if ".bias" in key and not ".norm.bias" in key:
            continue
        pytorch_state_dict[key] = value

    pytorch_model.load_state_dict(pytorch_state_dict)
    pytorch_model.eval()
    pytorch_model = pytorch_model.to(torch.bfloat16)

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
def test_resnet_stem_pcc(device, batch_size, height, width, reset_seeds):
    """Test ResNet stem layer PCC between PyTorch and TTNN implementations."""

    torch.manual_seed(0)

    try:
        pytorch_model, ttnn_model, state_dict = create_resnet_models(device=device)
    except FileNotFoundError:
        pytest.fail("R-52.pkl file not found. Please place the weights file in the weights folder.")

    torch_input = torch.randn(batch_size, 3, height, width, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    ttnn_stem_output = ttnn_model.stem(ttnn_input)
    with torch.no_grad():
        torch_stem_output = pytorch_model.stem(torch_input)

    ttnn_output_torch = ttnn.to_torch(ttnn_stem_output).permute(0, 3, 1, 2)
    pcc_passed, pcc_message = assert_with_pcc(torch_stem_output, ttnn_output_torch, 0.95)

    logger.info(f"PCC: {pcc_message}")
    assert pcc_passed, f"ResNet stem PCC test failed: {pcc_message}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "height,width",
    [(512, 1024)],
)
@pytest.mark.parametrize("layer_name", ["res2", "res3", "res4", "res5"])
def test_resnet_layer_pcc(device, batch_size, height, width, layer_name, reset_seeds):
    """Test ResNet individual layer PCC between PyTorch and TTNN implementations."""

    torch.manual_seed(0)

    try:
        pytorch_model, ttnn_model, state_dict = create_resnet_models(device=device)
    except FileNotFoundError:
        pytest.fail("R-52.pkl file not found. Please place the weights file in the weights folder.")

    layer_specs = {
        "res2": (128, 1 / 4, 1 / 4),
        "res3": (256, 1 / 4, 1 / 4),
        "res4": (512, 1 / 8, 1 / 8),
        "res5": (1024, 1 / 16, 1 / 16),
    }

    in_channels, h_factor, w_factor = layer_specs[layer_name]
    layer_height = int(height * h_factor)
    layer_width = int(width * w_factor)

    torch_layer_input = torch.randn(batch_size, in_channels, layer_height, layer_width, dtype=torch.bfloat16)
    ttnn_layer_input = ttnn.from_torch(
        torch_layer_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    if layer_name == "res2":
        for block in ttnn_model.res2:
            ttnn_layer_input = block(ttnn_layer_input)
        ttnn_layer_output = ttnn_layer_input
    elif layer_name == "res3":
        for block in ttnn_model.res3:
            ttnn_layer_input = block(ttnn_layer_input)
        ttnn_layer_output = ttnn_layer_input
    elif layer_name == "res4":
        for block in ttnn_model.res4:
            ttnn_layer_input = block(ttnn_layer_input)
        ttnn_layer_output = ttnn_layer_input
    elif layer_name == "res5":
        for block in ttnn_model.res5:
            ttnn_layer_input = block(ttnn_layer_input)
        ttnn_layer_output = ttnn_layer_input

    with torch.no_grad():
        if layer_name == "res2":
            torch_layer_output = pytorch_model.res2(torch_layer_input)
        elif layer_name == "res3":
            torch_layer_output = pytorch_model.res3(torch_layer_input)
        elif layer_name == "res4":
            torch_layer_output = pytorch_model.res4(torch_layer_input)
        elif layer_name == "res5":
            torch_layer_output = pytorch_model.res5(torch_layer_input)

    ttnn_output_torch = ttnn.to_torch(ttnn_layer_output).permute(0, 3, 1, 2)
    pcc_passed, pcc_message = assert_with_pcc(torch_layer_output, ttnn_output_torch, 0.95)

    logger.info(f"PCC: {pcc_message}")
    assert pcc_passed, f"ResNet {layer_name} PCC test failed: {pcc_message}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "height,width",
    [(512, 1024)],
)
def test_resnet_full_pcc(device, batch_size, height, width, reset_seeds):
    """Test full ResNet PCC between PyTorch and TTNN implementations."""

    torch.manual_seed(0)

    try:
        pytorch_model, ttnn_model, state_dict = create_resnet_models(device=device)
    except FileNotFoundError:
        pytest.fail("R-52.pkl file not found. Please place the weights file in the weights folder.")

    torch_input = torch.randn(batch_size, 3, height, width, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    ttnn_outputs = ttnn_model(ttnn_input)
    with torch.no_grad():
        torch_outputs = pytorch_model(torch_input)

    failed_layers = []
    for layer_name in ["res2", "res3", "res4", "res5"]:
        torch_output = torch_outputs[layer_name]
        ttnn_output = ttnn_outputs[layer_name]
        ttnn_output_torch = ttnn.to_torch(ttnn_output).permute(0, 3, 1, 2)

        pcc_passed, pcc_message = check_with_pcc(torch_output, ttnn_output_torch, 0.95)
        logger.info(f"{layer_name} PCC: {pcc_message}")

        if not pcc_passed:
            failed_layers.append(layer_name)

    assert len(failed_layers) == 0, f"ResNet full PCC test failed for layers: {failed_layers}"
