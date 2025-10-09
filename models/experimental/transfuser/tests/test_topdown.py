# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from loguru import logger
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.transfuser.reference.topdown import TopDown
from models.experimental.transfuser.tt.topdown import TtTopDown


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

    # TT-NN model with weight transfer
    tt_model = TtTopDown(device, torch_model, 512, 64, 2)

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
