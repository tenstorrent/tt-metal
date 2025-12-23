# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from loguru import logger
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.transfuser.reference.topdown import TopDown
from models.experimental.transfuser.tt.topdown import TtTopDown
from ttnn.model_preprocessing import preprocess_model_parameters


def create_topdown_preprocessor(device, dtype=ttnn.bfloat16):
    """Create a custom preprocessor for TopDown model weights."""

    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}

        # Process conv layers - weights need special handling for ttnn.conv2d
        for conv_name in ["c5_conv", "up_conv5", "up_conv4", "up_conv3"]:
            if hasattr(torch_model, conv_name):
                conv_layer = getattr(torch_model, conv_name)
                parameters[conv_name] = {}
                # Conv2d expects weights in ROW_MAJOR_LAYOUT (not TILE_LAYOUT)
                parameters[conv_name]["weight"] = ttnn.from_torch(
                    conv_layer.weight, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                if conv_layer.bias is not None:
                    # Bias needs to be reshaped for ttnn.conv2d
                    bias_reshaped = conv_layer.bias.reshape(1, 1, 1, -1)
                    parameters[conv_name]["bias"] = ttnn.from_torch(
                        bias_reshaped, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                    )

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_topdown_pcc_comparison(device):
    # PyTorch model
    torch_model = TopDown(perception_output_features=512, bev_features_chanels=64, bev_upsample_factor=2)
    torch_model.eval()

    # Create input
    torch_input = torch.randn(1, 512, 8, 8)

    # PyTorch forward pass
    with torch.no_grad():
        torch_p2, torch_p3, torch_p4, torch_p5 = torch_model(torch_input)

    # Preprocess model parameters to ttnn format
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_topdown_preprocessor(device),
        device=device,
    )

    # TT-NN model with preprocessed parameters
    tt_model = TtTopDown(device, parameters, 512, 64, 2)

    # Convert input to TT-NN format (NHWC)
    ttnn_input = torch_input.permute(0, 2, 3, 1)  # NCHW -> NHWC
    ttnn_input = ttnn.from_torch(ttnn_input, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    # TT-NN forward pass
    tt_p2, tt_p3, tt_p4, tt_p5 = tt_model(ttnn_input)

    # Convert outputs back to PyTorch format
    tt_p2_torch = ttnn.to_torch(tt_p2).permute(0, 3, 1, 2)  # NHWC -> NCHW
    tt_p3_torch = ttnn.to_torch(tt_p3).permute(0, 3, 1, 2)
    tt_p4_torch = ttnn.to_torch(tt_p4).permute(0, 3, 1, 2)
    tt_p5_torch = ttnn.to_torch(tt_p5).permute(0, 3, 1, 2)
    pcc_threshold = 0.99
    p2_passed, p2_pcc_message = check_with_pcc(torch_p2, tt_p2_torch, pcc=pcc_threshold)
    logger.info(f"P2 Output PCC: {p2_pcc_message}")
    assert p2_passed, f"PCC check failed for P2: {p2_pcc_message}"

    p3_passed, p3_pcc_message = check_with_pcc(torch_p3, tt_p3_torch, pcc=pcc_threshold)
    logger.info(f"P3 Output PCC: {p3_pcc_message}")
    assert p3_passed, f"PCC check failed for P3: {p3_pcc_message}"

    p4_passed, p4_pcc_message = check_with_pcc(torch_p4, tt_p4_torch, pcc=pcc_threshold)
    logger.info(f"P4 Output PCC: {p4_pcc_message}")
    assert p4_passed, f"PCC check failed for P4: {p4_pcc_message}"

    p5_passed, p5_pcc_message = check_with_pcc(torch_p5, tt_p5_torch, pcc=pcc_threshold)
    logger.info(f"P5 Output PCC: {p5_pcc_message}")
    assert p5_passed, f"PCC check failed for P5: {p5_pcc_message}"

    logger.info("All PCC tests passed!")
