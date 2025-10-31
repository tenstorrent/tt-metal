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
    PANOPTIC_DEEPLAB,
    DEEPLAB_V3_PLUS,
    get_panoptic_deeplab_weights_path,
    get_panoptic_deeplab_config,
)
from models.experimental.panoptic_deeplab.tests.pcc.common import check_ttnn_output
from models.experimental.panoptic_deeplab.tt.common import preprocess_nchw_input_tensor

@pytest.mark.parametrize(
    "model_category",
    [PANOPTIC_DEEPLAB, DEEPLAB_V3_PLUS],
    ids=["test_panoptic_deeplab", "test_deeplab_v3_plus"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": PDL_L1_SMALL_SIZE}], indirect=True)
def test_model_panoptic_deeplab(device, model_category, model_location_generator):
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
    ttnn_input = preprocess_nchw_input_tensor(device, pytorch_input)

    try:
        pytorch_model = PytorchPanopticDeepLab(
            num_classes=num_classes,
            common_stride=common_stride,
            project_channels=project_channels,
            decoder_channels=decoder_channels,
            sem_seg_head_channels=sem_seg_head_channels,
            ins_embed_head_channels=ins_embed_head_channels,
            train_size=train_size,
            weights_path=complete_weights_path,
            model_category=model_category,
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
        model_configs.setup_resnet_backbone()
        logger.info("Applying ASPP layer overrides...")
        model_configs.setup_aspp()
        logger.info("Applying decoder layer overrides...")
        model_configs.setup_decoder()
        logger.info("Applying head layer overrides...")
        model_configs.setup_heads()

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
            train_size=train_size,
            model_configs=model_configs,
            model_category=model_category,
        )
    except FileNotFoundError:
        pytest.fail("model_final_bd324a.pkl file not found. Please place the weights file in the weights folder.")

    logger.info("Running PyTorch model...")
    with torch.no_grad():
        pytorch_semantic, pytorch_center, pytorch_offset, _ = pytorch_model.forward(pytorch_input)

    logger.info("Running TTNN model with fused Conv+BatchNorm parameters...")
    ttnn_semantic, ttnn_center, ttnn_offset, _ = ttnn_model.forward(ttnn_input)

    all_passed = []
    all_passed.append(
        check_ttnn_output(
            "Semantic",
            pytorch_semantic,
            ttnn_semantic,
            to_channel_first=False,
            output_channels=ttnn_model.semantic_head.get_output_channels_for_slicing(),
            exp_pcc=0.993,
        )
    )
    if model_category == PANOPTIC_DEEPLAB:
        all_passed.append(
            check_ttnn_output(
                "Center",
                pytorch_center,
                ttnn_center,
                to_channel_first=False,
                output_channels=ttnn_model.instance_head.get_center_output_channels_for_slicing(),
                exp_pcc=0.959,
            )
        )
        all_passed.append(
            check_ttnn_output(
                "Offset",
                pytorch_offset,
                ttnn_offset,
                to_channel_first=False,
                output_channels=ttnn_model.instance_head.get_offset_output_channels_for_slicing(),
                exp_pcc=0.999,
            )
        )

    # Fail test based on PCC results
    assert all(all_passed), f"PDL outputs did not pass the PCC check {all_passed=}"
    logger.info("All PCC tests passed!")
