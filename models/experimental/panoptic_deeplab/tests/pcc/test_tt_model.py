# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC test for the complete TtPanopticDeepLab model.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.panoptic_deeplab.tt.model_preprocessing import (
    create_panoptic_deeplab_parameters,
    fuse_conv_bn_parameters,
)
from models.experimental.panoptic_deeplab.tt.tt_model import TtPanopticDeepLab
from models.experimental.panoptic_deeplab.reference.pytorch_model import PytorchPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.model_configs import ModelOptimisations
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.panoptic_deeplab.tt.common import (
    PDL_L1_SMALL_SIZE,
    get_panoptic_deeplab_weights_path,
    get_panoptic_deeplab_config,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": PDL_L1_SMALL_SIZE}], indirect=True)
def test_panoptic_deeplab(device, model_location_generator):
    """Test PCC comparison between PyTorch and TTNN implementations with fused Conv+BatchNorm."""

    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x != 5 or compute_grid.y != 4:
        pytest.skip(f"Test requires compute grid size of 5x4, but got {compute_grid.x}x{compute_grid.y}")

    torch.manual_seed(0)

    # Get the weights path using the common utility function
    complete_weights_path = get_panoptic_deeplab_weights_path(model_location_generator, __file__)

    # Get model configuration
    config = get_panoptic_deeplab_config()
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]
    project_channels = config["project_channels"]
    decoder_channels = config["decoder_channels"]
    sem_seg_head_channels = config["sem_seg_head_channels"]
    ins_embed_head_channels = config["ins_embed_head_channels"]
    common_stride = config["common_stride"]
    train_size = config["train_size"]

    input_height, input_width = train_size[0], train_size[1]
    input_channels = 3

    pytorch_input = torch.randn(batch_size, input_channels, input_height, input_width, dtype=torch.bfloat16)

    ttnn_input_torch = pytorch_input.permute(0, 2, 3, 1)

    ttnn_input = ttnn.from_torch(ttnn_input_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    try:
        pytorch_model = PytorchPanopticDeepLab(
            num_classes=num_classes,
            common_stride=common_stride,
            project_channels=project_channels,
            decoder_channels=decoder_channels,
            sem_seg_head_channels=sem_seg_head_channels,
            ins_embed_head_channels=ins_embed_head_channels,
            norm="SyncBN",
            train_size=train_size,
            weights_path=complete_weights_path,
        )
        pytorch_model = pytorch_model.to(dtype=torch.bfloat16)
        pytorch_model.eval()

        # Create TTNN parameters from the PyTorch model with loaded weights
        ttnn_parameters = create_panoptic_deeplab_parameters(pytorch_model, device)

        # Apply Conv+BatchNorm fusion to the parameters
        logger.info("Applying Conv+BatchNorm fusion to parameters...")
        fused_parameters = fuse_conv_bn_parameters(ttnn_parameters, eps=1e-5)
        logger.info("Conv+BatchNorm fusion completed successfully")

        # Create centralized configuration
        model_configs = ModelOptimisations(
            conv_act_dtype=ttnn.bfloat8_b,
            conv_w_dtype=ttnn.bfloat8_b,
        )

        # Create TTNN model with fused parameters and centralized configuration
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
            model_configs=model_configs,
        )
    except FileNotFoundError:
        pytest.fail("model_final_bd324a.pkl file not found. Please place the weights file in the weights folder.")

    logger.info("Running PyTorch model...")
    with torch.no_grad():
        pytorch_semantic, pytorch_center, pytorch_offset, _ = pytorch_model.forward(pytorch_input)

    logger.info("Running TTNN model with fused Conv+BatchNorm parameters...")
    ttnn_semantic, ttnn_center, ttnn_offset, _ = ttnn_model.forward(ttnn_input)

    ttnn_semantic_torch = ttnn.to_torch(ttnn_semantic).permute(0, 3, 1, 2)
    ttnn_center_torch = ttnn.to_torch(ttnn_center).permute(0, 3, 1, 2)
    ttnn_offset_torch = ttnn.to_torch(ttnn_offset).permute(0, 3, 1, 2)

    sem_passed, sem_msg = assert_with_pcc(pytorch_semantic, ttnn_semantic_torch, pcc=0.96)
    logger.info(f"Semantic PCC: {sem_msg}")
    assert sem_passed, f"Semantic segmentation PCC failed: {sem_msg}"

    center_passed, center_msg = assert_with_pcc(pytorch_center, ttnn_center_torch, pcc=0.94)
    logger.info(f"Center PCC: {center_msg}")
    assert center_passed, f"Center heatmap PCC failed: {center_msg}"

    offset_passed, offset_msg = assert_with_pcc(pytorch_offset, ttnn_offset_torch, pcc=0.99)
    logger.info(f"Offset PCC: {offset_msg}")
    assert offset_passed, f"Offset map PCC failed: {offset_msg}"

    logger.info("All PCC tests passed!")
