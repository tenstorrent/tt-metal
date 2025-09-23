# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import os
from loguru import logger

from models.experimental.panoptic_deeplab.tt.model_preprocessing import (
    create_panoptic_deeplab_parameters,
    fuse_conv_bn_parameters,
)
from models.experimental.panoptic_deeplab.tt.tt_model import TtPanopticDeepLab
from models.experimental.panoptic_deeplab.reference.pytorch_model import PytorchPanopticDeepLab
from tests.ttnn.utils_for_testing import assert_with_pcc


PDL_L1_SMALL_SIZE = 37 * 1024  # Minimum L1 small size for Panoptic DeepLab


@pytest.mark.parametrize("device_params", [{"l1_small_size": PDL_L1_SMALL_SIZE}], indirect=True)
def test_ttnn_aspp(device, model_location_generator):
    """Test ASPP component using the full model with real weights."""
    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x != 5 or compute_grid.y != 4:
        pytest.skip(f"Test requires compute grid size of 5x4, but got {compute_grid.x}x{compute_grid.y}")

    torch.manual_seed(0)

    # Determine weights path based on environment
    if model_location_generator is None or "TT_GH_CI_INFRA" not in os.environ:
        # Use local path (old method)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        complete_weights_path = os.path.join(current_dir, "..", "..", "weights", "model_final_bd324a.pkl")
    else:
        # Use CI v2 model location generator
        complete_weights_path = (
            model_location_generator("vision-models/panoptic_deeplab", model_subdir="", download_if_ci_v2=True)
            / "model_final_bd324a.pkl"
        )

    # Model configuration
    batch_size = 1
    num_classes = 19
    project_channels = [32, 64]
    decoder_channels = [256, 256, 256]
    sem_seg_head_channels = 256
    ins_embed_head_channels = 32
    common_stride = 4
    train_size = (512, 1024)

    # Create input for ASPP testing (res5 feature map size)
    input_height, input_width = 32, 64  # res5 feature map size
    input_channels = 2048  # res5 channels

    pytorch_input = torch.randn(batch_size, input_channels, input_height, input_width, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        pytorch_input.permute(0, 2, 3, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    try:
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
    except FileNotFoundError:
        pytest.fail("model_final_bd324a.pkl file not found. Please place the weights file in the weights folder.")

    # Test ASPP component specifically by testing semantic head which uses ASPP
    logger.info("Running PyTorch ASPP test...")
    with torch.no_grad():
        # Get ASPP output from semantic head decoder - ASPP is the project_conv for res5
        pytorch_aspp_output = pytorch_model.semantic_head.decoder["res5"]["project_conv"](pytorch_input)

    logger.info("Running TTNN ASPP test...")
    # Get ASPP output from TTNN semantic head decoder - ASPP is the project_conv for res5
    ttnn_aspp_output = ttnn_model.semantic_head.decoder["res5"]["project_conv"](ttnn_input)

    ttnn_aspp_output_torch = ttnn.to_torch(ttnn_aspp_output).permute(0, 3, 1, 2)

    pcc_passed, pcc_message = assert_with_pcc(pytorch_aspp_output, ttnn_aspp_output_torch, pcc=0.99)

    logger.info(f"ASPP PCC: {pcc_message}")
    assert pcc_passed, f"ASPP PCC test failed: {pcc_message}"
    assert (
        pytorch_aspp_output.shape == ttnn_aspp_output_torch.shape
    ), f"Shape mismatch: {pytorch_aspp_output.shape} vs {ttnn_aspp_output_torch.shape}"
