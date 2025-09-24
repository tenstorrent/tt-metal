# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
ResNet Tests for Panoptic DeepLab using real weights from model_final_bd324a.pkl.
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from loguru import logger

from models.experimental.panoptic_deeplab.tt.model_preprocessing import (
    create_panoptic_deeplab_parameters,
    fuse_conv_bn_parameters,
)
from models.experimental.panoptic_deeplab.tt.tt_model import TtPanopticDeepLab
from models.experimental.panoptic_deeplab.reference.pytorch_model import PytorchPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.common import (
    PDL_L1_SMALL_SIZE,
    get_panoptic_deeplab_weights_path,
    get_panoptic_deeplab_config,
)


def create_panoptic_models(device, weights_path):
    """Create both PyTorch and TTNN Panoptic DeepLab models.

    Args:
        device: TTNN device for model creation
        weights_path: Path to the model weights file
    """
    # Get model configuration
    config = get_panoptic_deeplab_config()
    num_classes = config["num_classes"]
    project_channels = config["project_channels"]
    decoder_channels = config["decoder_channels"]
    sem_seg_head_channels = config["sem_seg_head_channels"]
    ins_embed_head_channels = config["ins_embed_head_channels"]
    common_stride = config["common_stride"]
    train_size = config["train_size"]

    # Load PyTorch model with real weights
    pytorch_model = PytorchPanopticDeepLab(
        num_classes=num_classes,
        common_stride=common_stride,
        project_channels=project_channels,
        decoder_channels=decoder_channels,
        sem_seg_head_channels=sem_seg_head_channels,
        ins_embed_head_channels=ins_embed_head_channels,
        norm="SyncBN",
        train_size=train_size,
        weights_path=weights_path,
    )
    pytorch_model = pytorch_model.to(dtype=torch.bfloat16)
    pytorch_model.eval()

    # Create TTNN parameters from the PyTorch model with loaded weights
    ttnn_parameters = create_panoptic_deeplab_parameters(pytorch_model, device)

    # Apply Conv+BatchNorm fusion to the parameters
    logger.info("Applying Conv+BatchNorm fusion to parameters...")
    fused_parameters = fuse_conv_bn_parameters(ttnn_parameters, eps=1e-5)
    logger.info("Conv+BatchNorm fusion completed successfully")

    # Create TTNN model with fused parameters
    ttnn_model = TtPanopticDeepLab(
        device=device,
        parameters=fused_parameters,
        num_classes=num_classes,
        common_stride=common_stride,
        project_channels=project_channels,
        decoder_channels=decoder_channels,
        sem_seg_head_channels=sem_seg_head_channels,
        ins_embed_head_channels=ins_embed_head_channels,
        norm="",
        train_size=train_size,
    )

    return pytorch_model, ttnn_model


@pytest.mark.parametrize("device_params", [{"l1_small_size": PDL_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "height,width",
    [(512, 1024)],
)
def test_resnet_stem_pcc(device, batch_size, height, width, reset_seeds, model_location_generator):
    """Test ResNet stem layer PCC between PyTorch and TTNN implementations."""
    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x != 5 or compute_grid.y != 4:
        pytest.skip(f"Test requires compute grid size of 5x4, but got {compute_grid.x}x{compute_grid.y}")

    torch.manual_seed(0)

    # Get the weights path using the common utility function
    complete_weights_path = get_panoptic_deeplab_weights_path(model_location_generator, __file__)

    try:
        pytorch_model, ttnn_model = create_panoptic_models(device, complete_weights_path)
    except FileNotFoundError:
        pytest.fail("model_final_bd324a.pkl file not found. Please place the weights file in the weights folder.")

    torch_input = torch.randn(batch_size, 3, height, width, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    ttnn_stem_output = ttnn_model.backbone.stem(ttnn_input)
    with torch.no_grad():
        torch_stem_output = pytorch_model.backbone.stem(torch_input)

    ttnn_output_torch = ttnn.to_torch(ttnn_stem_output).permute(0, 3, 1, 2)
    pcc_passed, pcc_message = assert_with_pcc(torch_stem_output, ttnn_output_torch, 0.99)

    logger.info(f"ResNet stem PCC: {pcc_message}")
    assert pcc_passed, f"ResNet stem PCC test failed: {pcc_message}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": PDL_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "height,width",
    [(512, 1024)],
)
@pytest.mark.parametrize("layer_name, expected_pcc", [("res2", 0.99), ("res3", 0.99), ("res4", 0.95), ("res5", 0.93)])
def test_resnet_layer_pcc(
    device, batch_size, height, width, layer_name, expected_pcc, reset_seeds, model_location_generator
):
    """Test ResNet individual layer PCC between PyTorch and TTNN implementations."""

    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x != 5 or compute_grid.y != 4:
        pytest.skip(f"Test requires compute grid size of 5x4, but got {compute_grid.x}x{compute_grid.y}")

    torch.manual_seed(0)

    # Get the weights path using the common utility function
    complete_weights_path = get_panoptic_deeplab_weights_path(model_location_generator, __file__)

    try:
        pytorch_model, ttnn_model = create_panoptic_models(device, complete_weights_path)
    except FileNotFoundError:
        pytest.fail("model_final_bd324a.pkl file not found. Please place the weights file in the weights folder.")

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
        for block in ttnn_model.backbone.res2:
            ttnn_layer_input = block(ttnn_layer_input)
        ttnn_layer_output = ttnn_layer_input
    elif layer_name == "res3":
        for block in ttnn_model.backbone.res3:
            ttnn_layer_input = block(ttnn_layer_input)
        ttnn_layer_output = ttnn_layer_input
    elif layer_name == "res4":
        for block in ttnn_model.backbone.res4:
            ttnn_layer_input = block(ttnn_layer_input)
        ttnn_layer_output = ttnn_layer_input
    elif layer_name == "res5":
        for block in ttnn_model.backbone.res5:
            ttnn_layer_input = block(ttnn_layer_input)
        ttnn_layer_output = ttnn_layer_input

    with torch.no_grad():
        if layer_name == "res2":
            torch_layer_output = pytorch_model.backbone.res2(torch_layer_input)
        elif layer_name == "res3":
            torch_layer_output = pytorch_model.backbone.res3(torch_layer_input)
        elif layer_name == "res4":
            torch_layer_output = pytorch_model.backbone.res4(torch_layer_input)
        elif layer_name == "res5":
            torch_layer_output = pytorch_model.backbone.res5(torch_layer_input)

    ttnn_output_torch = ttnn.to_torch(ttnn_layer_output).permute(0, 3, 1, 2)

    pcc_passed, pcc_message = assert_with_pcc(torch_layer_output, ttnn_output_torch, expected_pcc)

    logger.info(f"ResNet {layer_name} PCC: {pcc_message}")
    assert pcc_passed, f"ResNet {layer_name} PCC test failed: {pcc_message}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": PDL_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "height,width",
    [(512, 1024)],
)
def test_resnet_full_pcc(device, batch_size, height, width, reset_seeds, model_location_generator):
    """Test full ResNet PCC between PyTorch and TTNN implementations."""
    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x != 5 or compute_grid.y != 4:
        pytest.skip(f"Test requires compute grid size of 5x4, but got {compute_grid.x}x{compute_grid.y}")

    torch.manual_seed(0)

    # Get the weights path using the common utility function
    complete_weights_path = get_panoptic_deeplab_weights_path(model_location_generator, __file__)

    try:
        pytorch_model, ttnn_model = create_panoptic_models(device, complete_weights_path)
    except FileNotFoundError:
        pytest.fail("model_final_bd324a.pkl file not found. Please place the weights file in the weights folder.")

    torch_input = torch.randn(batch_size, 3, height, width, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    ttnn_outputs = ttnn_model.backbone(ttnn_input)
    with torch.no_grad():
        torch_outputs = pytorch_model.backbone(torch_input)

    failed_layers = []
    # Set layer-specific PCC thresholds based on test failures
    layer_pcc_thresholds = {
        "res2": 0.99,
        "res3": 0.99,
        "res4": 0.96,
        "res5": 0.93,
    }

    for layer_name in ["res2", "res3", "res4", "res5"]:
        torch_output = torch_outputs[layer_name]
        ttnn_output = ttnn_outputs[layer_name]
        ttnn_output_torch = ttnn.to_torch(ttnn_output).permute(0, 3, 1, 2)

        pcc_threshold = layer_pcc_thresholds[layer_name]
        pcc_passed, pcc_message = check_with_pcc(torch_output, ttnn_output_torch, pcc_threshold)
        logger.info(f"ResNet {layer_name} PCC: {pcc_message}")

        if not pcc_passed:
            failed_layers.append(layer_name)

    assert len(failed_layers) == 0, f"ResNet full PCC test failed for layers: {failed_layers}"
