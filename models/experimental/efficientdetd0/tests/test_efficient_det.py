# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.efficientdetd0.reference.efficientdet import EfficientDetBackbone
from models.experimental.efficientdetd0.tt.efficient_det import TtEfficientDetBackbone

import pytest
from loguru import logger

from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.efficientdetd0.tt.custom_preprocessor import (
    infer_torch_module_args,
    create_custom_mesh_preprocessor,
)
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.efficientdetd0.common import load_torch_model_state


torch.manual_seed(0)


@pytest.mark.parametrize(
    "batch, channels, height, width",
    [
        (1, 3, 512, 512),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_efficient_det(batch, channels, height, width, device):
    PCC_THRESHOLD = 0.99
    num_classes = 90
    torch_model = EfficientDetBackbone(
        num_classes=num_classes,
        load_weights=False,
    ).eval()
    load_torch_model_state(torch_model)

    # Run PyTorch forward pass
    torch_inputs = torch.randn(batch, channels, height, width)
    with torch.no_grad():
        torch_features, torch_regression, torch_classification = torch_model(torch_inputs)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )
    module_args = infer_torch_module_args(model=torch_model, input=torch_inputs)

    # Create TTNN BiFPN model
    ttnn_model = TtEfficientDetBackbone(
        device=device,
        parameters=parameters,
        conv_params=module_args,
        num_classes=num_classes,
    )
    # Convert inputs to TTNN format
    ttnn_input_tensor = ttnn.from_torch(
        torch_inputs,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_features, ttnn_regression, ttnn_classification = ttnn_model(ttnn_input_tensor)

    # Compare features (tuple of P3, P4, P5, P6, P7 after BiFPN)
    logger.info("Comparing BiFPN feature outputs...")
    all_passed = True

    # torch_features is a tuple of 5 feature maps after BiFPN
    for i, (torch_feat, ttnn_feat) in enumerate(zip(torch_features, ttnn_features)):
        ttnn_feat_torch = ttnn.to_torch(ttnn_feat)

        # Get expected dimensions from PyTorch output (NCHW format)
        expected_batch, expected_channels, expected_h, expected_w = torch_feat.shape

        # TTNN output is in format [1, 1, H*W, C] (NHWC flattened)
        # Reshape to [batch, H, W, C]
        ttnn_feat_torch = ttnn_feat_torch.reshape(expected_batch, expected_h, expected_w, expected_channels)

        # Permute from NHWC to NCHW to match PyTorch
        ttnn_feat_torch = ttnn_feat_torch.permute(0, 3, 1, 2)

        passing, pcc_message = check_with_pcc(torch_feat, ttnn_feat_torch, PCC_THRESHOLD)
        logger.info(f"Feature {i} (P{i+3}) PCC: {pcc_message}")
        all_passed = all_passed and passing

    # Compare regression outputs
    # Compare regression outputs
    logger.info("Comparing regression outputs...")
    ttnn_regression_torch = ttnn.to_torch(ttnn_regression)

    # Regression output is 3D: [batch, num_detections, 4]
    # No need to reshape or permute - just compare directly
    passing, pcc_message = check_with_pcc(torch_regression, ttnn_regression_torch, PCC_THRESHOLD)
    logger.info(f"Regression PCC: {pcc_message}")
    all_passed = all_passed and passing

    # Compare classification output (3D tensor: [batch, num_anchors, num_classes])
    logger.info("Comparing classification outputs...")
    ttnn_classification_torch = ttnn.to_torch(ttnn_classification)

    passing, pcc_message = check_with_pcc(torch_classification, ttnn_classification_torch, PCC_THRESHOLD)
    logger.info(f"Classification PCC: {pcc_message}")
    all_passed = all_passed and passing

    if all_passed:
        logger.info("EfficientDet Test Passed!")
    else:
        logger.warning("EfficientDet Test Failed!")

    assert all_passed, f"PCC value is lower than {PCC_THRESHOLD}. Check implementation!"
