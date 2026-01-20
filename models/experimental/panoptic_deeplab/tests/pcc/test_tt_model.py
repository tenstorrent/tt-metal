# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC test for the complete TtPanopticDeepLab model.
"""

import pytest
import torch
from loguru import logger
from models.experimental.panoptic_deeplab.reference.pytorch_model import PANOPTIC_DEEPLAB, DEEPLAB_V3_PLUS
from models.experimental.panoptic_deeplab.tt.common import (
    PDL_L1_SMALL_SIZE,
    get_panoptic_deeplab_weights_path,
    get_panoptic_deeplab_config,
    preprocess_nchw_input_tensor,
    validate_outputs_with_pcc,
    create_pytorch_model,
    create_ttnn_model,
)
from models.experimental.panoptic_deeplab.tests.pcc.common import (
    skip_if_not_blackhole_110_cores,
    skip_if_not_blackhole_20_cores,
)


@pytest.mark.parametrize(
    "model_category, pcc_values, skip_check",
    [
        (
            PANOPTIC_DEEPLAB,
            {
                "semantic": {"pcc": 0.986, "abs_err": 1.3, "rel_err": 0.4},
                "center": {"pcc": 0.804, "abs_err": 0.1, "rel_err": 2.1},
                "offset": {"pcc": 0.990, "abs_err": 10.4, "rel_err": 0.6},
            },
            skip_if_not_blackhole_20_cores,
        ),
        (
            PANOPTIC_DEEPLAB,
            {
                "semantic": {"pcc": 0.983, "abs_err": 1.4, "rel_err": 0.4},
                "center": {"pcc": 0.8, "abs_err": 0.1, "rel_err": 2.2},
                "offset": {"pcc": 0.987, "abs_err": 11.7, "rel_err": 0.7},
            },
            skip_if_not_blackhole_110_cores,
        ),
        (
            DEEPLAB_V3_PLUS,
            {
                "semantic": {"pcc": 0.986, "abs_err": 1.3, "rel_err": 0.4},
            },
            skip_if_not_blackhole_20_cores,
        ),
        (
            DEEPLAB_V3_PLUS,
            {
                "semantic": {"pcc": 0.983, "abs_err": 1.4, "rel_err": 0.4},
            },
            skip_if_not_blackhole_110_cores,
        ),
    ],
    ids=[
        "panoptic_deeplab_20_cores",
        "panoptic_deeplab_110_cores",
        "deeplab_v3_plus_20_cores",
        "deeplab_v3_plus_110_cores",
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": PDL_L1_SMALL_SIZE}], indirect=True)
def test_model_panoptic_deeplab(device, model_category, pcc_values, skip_check, model_location_generator):
    """Test PCC comparison between PyTorch and TTNN implementations with fused Conv+BatchNorm."""

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

    input_height, input_width = train_size[0], train_size[1]
    input_channels = 3

    # Both models have ImageNet normalization fused into conv1 weights
    # so they both receive unnormalized input (no explicit normalization needed)
    pytorch_input = torch.randn(batch_size, input_channels, input_height, input_width, dtype=torch.bfloat16)

    # Use proper input preprocessing to avoid OOM (creates HEIGHT SHARDED memory config)
    ttnn_input = preprocess_nchw_input_tensor(device, pytorch_input)

    try:
        # Create PyTorch model using helper function
        pytorch_model = create_pytorch_model(
            weights_path=complete_weights_path,
            model_category=model_category,
            target_size=train_size,
            num_classes=num_classes,
            common_stride=common_stride,
            project_channels=project_channels,
            decoder_channels=decoder_channels,
            sem_seg_head_channels=sem_seg_head_channels,
            ins_embed_head_channels=ins_embed_head_channels,
        )

        # Create TTNN model using helper function (handles parameter creation, fusion, model configs, and model initialization)
        ttnn_model = create_ttnn_model(
            device=device,
            pytorch_model=pytorch_model,
            target_size=train_size,
            batch_size=batch_size,
            model_category=model_category,
            num_classes=num_classes,
            common_stride=common_stride,
            project_channels=project_channels,
            decoder_channels=decoder_channels,
            sem_seg_head_channels=sem_seg_head_channels,
            ins_embed_head_channels=ins_embed_head_channels,
        )
    except FileNotFoundError:
        pytest.fail("model_final_bd324a.pkl file not found. Please place the weights file in the weights folder.")

    logger.info("Running PyTorch model...")
    with torch.no_grad():
        pytorch_semantic, pytorch_center, pytorch_offset, _ = pytorch_model.forward(pytorch_input)

    logger.info("Running TTNN model with fused Conv+BatchNorm parameters...")
    ttnn_semantic, ttnn_center, ttnn_offset, _ = ttnn_model.forward(ttnn_input)

    # Extract PCC thresholds from parameters
    semantic_vals = pcc_values["semantic"]
    center_vals = pcc_values.get("center", {})
    offset_vals = pcc_values.get("offset", {})

    # Validate outputs with PCC using centralized validation function
    all_passed = validate_outputs_with_pcc(
        ttnn_model=ttnn_model,
        model_category=model_category,
        pytorch_semantic=pytorch_semantic,
        ttnn_semantic=ttnn_semantic,
        semantic_exp_pcc=semantic_vals["pcc"],
        semantic_exp_abs_err=semantic_vals["abs_err"],
        semantic_exp_rel_err=semantic_vals["rel_err"],
        pytorch_center=pytorch_center if model_category == PANOPTIC_DEEPLAB else None,
        ttnn_center=ttnn_center if model_category == PANOPTIC_DEEPLAB else None,
        center_exp_pcc=center_vals.get("pcc") if model_category == PANOPTIC_DEEPLAB else None,
        center_exp_abs_err=center_vals.get("abs_err") if model_category == PANOPTIC_DEEPLAB else None,
        center_exp_rel_err=center_vals.get("rel_err") if model_category == PANOPTIC_DEEPLAB else None,
        pytorch_offset=pytorch_offset if model_category == PANOPTIC_DEEPLAB else None,
        ttnn_offset=ttnn_offset if model_category == PANOPTIC_DEEPLAB else None,
        offset_exp_pcc=offset_vals.get("pcc") if model_category == PANOPTIC_DEEPLAB else None,
        offset_exp_abs_err=offset_vals.get("abs_err") if model_category == PANOPTIC_DEEPLAB else None,
        offset_exp_rel_err=offset_vals.get("rel_err") if model_category == PANOPTIC_DEEPLAB else None,
    )

    # Fail test based on PCC results
    assert all(all_passed), f"PDL outputs did not pass the PCC and tolerance check {all_passed=}"
    logger.info("All PCC and tolerance tests passed!")
