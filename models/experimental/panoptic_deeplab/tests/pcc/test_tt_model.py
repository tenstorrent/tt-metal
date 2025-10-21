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

    # Use proper input preprocessing to avoid OOM (creates HEIGHT SHARDED memory config)
    from models.experimental.panoptic_deeplab.tt.common import preprocess_nchw_input_tensor

    ttnn_input = preprocess_nchw_input_tensor(device, pytorch_input)

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
        # Use explicit input dimensions to match preprocessing
        ttnn_parameters = create_panoptic_deeplab_parameters(
            pytorch_model, device, input_height=input_height, input_width=input_width, batch_size=batch_size
        )

        # Apply Conv+BatchNorm fusion to the parameters
        logger.info("Applying Conv+BatchNorm fusion to parameters...")
        fused_parameters = fuse_conv_bn_parameters(ttnn_parameters, eps=1e-5)
        logger.info("Conv+BatchNorm fusion completed successfully")

        # Create centralized configuration
        model_configs = ModelOptimisations(
            conv_act_dtype=ttnn.bfloat8_b,
            conv_w_dtype=ttnn.bfloat8_b,
        )

        # Apply layer-specific configurations
        logger.info("Applying ResNet backbone configurations...")
        model_configs.setup_resnet_test_configs()
        logger.info("Applying ASPP layer overrides...")
        model_configs.setup_aspp_layer_overrides()
        logger.info("Applying decoder layer overrides...")
        model_configs.setup_decoder_layer_overrides()
        logger.info("Applying head layer overrides...")
        model_configs.setup_head_layer_overrides()

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

    # Handle semantic output - slice back to original channels if padding was applied
    ttnn_semantic_torch = ttnn.to_torch(ttnn_semantic).permute(0, 3, 1, 2)
    semantic_original_channels = ttnn_model.semantic_head.get_output_channels_for_slicing()
    if semantic_original_channels is not None:
        logger.info(
            f"Slicing semantic output from {ttnn_semantic_torch.shape[1]} to {semantic_original_channels} channels"
        )
        ttnn_semantic_torch = ttnn_semantic_torch[:, :semantic_original_channels, :, :]

    # Handle center output - slice back to original channels if padding was applied
    ttnn_center_torch = ttnn.to_torch(ttnn_center).permute(0, 3, 1, 2)
    center_original_channels = ttnn_model.instance_head.get_center_output_channels_for_slicing()
    if center_original_channels is not None:
        logger.info(f"Slicing center output from {ttnn_center_torch.shape[1]} to {center_original_channels} channels")
        ttnn_center_torch = ttnn_center_torch[:, :center_original_channels, :, :]

    # Handle offset output - slice back to original channels if padding was applied
    ttnn_offset_torch = ttnn.to_torch(ttnn_offset).permute(0, 3, 1, 2)
    offset_original_channels = ttnn_model.instance_head.get_offset_output_channels_for_slicing()
    if offset_original_channels is not None:
        logger.info(f"Slicing offset output from {ttnn_offset_torch.shape[1]} to {offset_original_channels} channels")
        ttnn_offset_torch = ttnn_offset_torch[:, :offset_original_channels, :, :]

    from tests.ttnn.utils_for_testing import check_with_pcc

    sem_passed, sem_msg = check_with_pcc(pytorch_semantic, ttnn_semantic_torch, pcc=0.99)
    logger.info(f"Semantic PCC: {sem_msg}")

    center_passed, center_msg = check_with_pcc(pytorch_center, ttnn_center_torch, pcc=0.99)
    logger.info(f"Center PCC: {center_msg}")

    offset_passed, offset_msg = check_with_pcc(pytorch_offset, ttnn_offset_torch, pcc=0.99)
    logger.info(f"Offset PCC: {offset_msg}")

    # Report all results
    failed_tests = []
    if not sem_passed:
        failed_tests.append(f"Semantic: {sem_msg}")
    if not center_passed:
        failed_tests.append(f"Center: {center_msg}")
    if not offset_passed:
        failed_tests.append(f"Offset: {offset_msg}")

    if failed_tests:
        assert False, f"PCC tests failed:\n" + "\n".join(failed_tests)

    logger.info("All PCC tests passed!")
