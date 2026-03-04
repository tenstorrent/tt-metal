# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from typing import Dict
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
from models.experimental.panoptic_deeplab.tests.pcc.common import (
    check_ttnn_output,
    skip_if_not_blackhole_110_cores,
    skip_if_not_blackhole_20_cores,
)


@pytest.mark.parametrize(
    "pcc_values, skip_check",
    [
        (
            {
                "center": {"pcc": 0.887, "abs_err": 0.09, "rel_err": 27.5},
                "offset": {"pcc": 0.742, "abs_err": 6.8, "rel_err": 5.0},
            },
            skip_if_not_blackhole_20_cores,
        ),
        (
            {
                "center": {"pcc": 0.887, "abs_err": 0.09, "rel_err": 27.5},
                "offset": {"pcc": 0.741, "abs_err": 6.8, "rel_err": 5.0},
            },
            skip_if_not_blackhole_110_cores,
        ),
    ],
    ids=["20_cores", "110_cores"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": PDL_L1_SMALL_SIZE}], indirect=True)
def test_ttnn_insemb(device, pcc_values, skip_check, model_location_generator):
    """Test instance embedding head using the full model with real weights."""

    # Skip test if device doesn't match the expected grid configuration
    skip_check(device)

    compute_grid = device.compute_with_storage_grid_size()
    logger.info(
        f"Running test on compute grid: {compute_grid.x}x{compute_grid.y} ({compute_grid.x * compute_grid.y} cores)"
    )

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

    # Create test features matching ResNet output
    torch_features: Dict[str, torch.Tensor] = {
        "res2": torch.randn(1, 256, 128, 256, dtype=torch.bfloat16),
        "res3": torch.randn(1, 512, 64, 128, dtype=torch.bfloat16),
        "res5": torch.randn(1, 2048, 32, 64, dtype=torch.bfloat16),
    }

    ttnn_features: Dict[str, ttnn.Tensor] = {
        name: ttnn.from_torch(tensor.permute(0, 2, 3, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        for name, tensor in torch_features.items()
    }

    try:
        # Load PyTorch model with real weights
        pytorch_model = PytorchPanopticDeepLab(
            num_classes=num_classes,
            common_stride=common_stride,
            project_channels=project_channels,
            decoder_channels=decoder_channels,
            sem_seg_head_channels=sem_seg_head_channels,
            ins_embed_head_channels=ins_embed_head_channels,
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
            device=device,
            conv_act_dtype=ttnn.bfloat8_b,
            conv_w_dtype=ttnn.bfloat8_b,
        )

        # Apply layer-specific configurations
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
        )
    except FileNotFoundError:
        pytest.fail("model_final_bd324a.pkl file not found. Please place the weights file in the weights folder.")

    # Test instance embedding head
    logger.info("Running PyTorch instance embedding head test...")
    with torch.no_grad():
        torch_center_out, torch_offset_out, _, _ = pytorch_model.instance_head(torch_features)

    logger.info("Running TTNN instance embedding head test...")
    ttnn_center_out_tt, ttnn_offset_out_tt, _, _ = ttnn_model.instance_head(ttnn_features)

    # Extract PCC thresholds from parameters
    center_vals = pcc_values["center"]
    offset_vals = pcc_values["offset"]

    all_passed = []
    all_passed.append(
        check_ttnn_output(
            "Center",
            torch_center_out,
            ttnn_center_out_tt,
            to_channel_first=False,
            output_channels=ttnn_model.instance_head.get_center_output_channels_for_slicing(),
            exp_pcc=center_vals["pcc"],
            exp_abs_err=center_vals["abs_err"],
            exp_rel_err=center_vals["rel_err"],
        )
    )
    all_passed.append(
        check_ttnn_output(
            "Offset",
            torch_offset_out,
            ttnn_offset_out_tt,
            to_channel_first=False,
            output_channels=ttnn_model.instance_head.get_offset_output_channels_for_slicing(),
            exp_pcc=offset_vals["pcc"],
            exp_abs_err=offset_vals["abs_err"],
            exp_rel_err=offset_vals["rel_err"],
        )
    )

    # Fail test based on PCC results
    assert all(all_passed), f"PDL outputs did not pass the PCC and tolerance check {all_passed=}"
    logger.info("All PCC and tolerance tests passed!")
