# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC (Pearson Correlation Coefficient) test for YUNet model.

Usage:
    # Default 640x640
    pytest models/experimental/yunet/tests/pcc/test_pcc.py -v

    # Run with 320x320
    pytest models/experimental/yunet/tests/pcc/test_pcc.py -v --input-size 320
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.yunet.common import YUNET_L1_SMALL_SIZE
from models.experimental.yunet.tt.ttnn_yunet import create_yunet_model
from models.common.utility_functions import comp_pcc


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YUNET_L1_SMALL_SIZE}],
    indirect=True,
)
def test_yunet_pcc(device, input_size):
    """
    Test PCC between PyTorch and TTNN YUNet outputs.

    Uses random weights to validate TTNN model architecture.
    Expected PCC > 0.99 for all outputs.

    Args:
        device: TTNN device fixture
        input_size: Tuple of (height, width) from conftest.py --input-size option
    """
    from models.experimental.yunet.YUNet.nets import nn as YUNet_nn

    # Get input dimensions from fixture
    input_h, input_w = input_size

    # Create PyTorch model with random weights (no fixed seed - truly random)
    torch_model = YUNet_nn.version_n().fuse().to(torch.bfloat16)

    # Create TTNN model
    ttnn_model = create_yunet_model(device, torch_model)

    # Create test input (random each run) - use input_size
    torch_input_nchw = torch.randn(1, 3, input_h, input_w, dtype=torch.bfloat16)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1)

    # Run PyTorch
    torch_model.train()
    with torch.no_grad():
        pt_cls, pt_box, pt_obj, pt_kpt = torch_model(torch_input_nchw)

    # Run TTNN
    tt_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_cls, tt_box, tt_obj, tt_kpt = ttnn_model(tt_input)

    # Flatten and concatenate all scales
    pt_cls_all = torch.cat([pt_cls[i].permute(0, 2, 3, 1).flatten() for i in range(3)])
    pt_box_all = torch.cat([pt_box[i].permute(0, 2, 3, 1).flatten() for i in range(3)])
    pt_obj_all = torch.cat([pt_obj[i].permute(0, 2, 3, 1).flatten() for i in range(3)])
    pt_kpt_all = torch.cat([pt_kpt[i].permute(0, 2, 3, 1).flatten() for i in range(3)])

    tt_cls_all = torch.cat([ttnn.to_torch(tt_cls[i]).flatten() for i in range(3)])
    tt_box_all = torch.cat([ttnn.to_torch(tt_box[i]).flatten() for i in range(3)])
    tt_obj_all = torch.cat([ttnn.to_torch(tt_obj[i]).flatten() for i in range(3)])
    tt_kpt_all = torch.cat([ttnn.to_torch(tt_kpt[i]).flatten() for i in range(3)])

    # Compute PCC
    pcc_threshold = 0.99

    cls_pass, pcc_cls = comp_pcc(pt_cls_all, tt_cls_all, pcc_threshold)
    box_pass, pcc_box = comp_pcc(pt_box_all, tt_box_all, pcc_threshold)
    obj_pass, pcc_obj = comp_pcc(pt_obj_all, tt_obj_all, pcc_threshold)
    kpt_pass, pcc_kpt = comp_pcc(pt_kpt_all, tt_kpt_all, pcc_threshold)

    logger.info(
        f"PCC ({input_h}x{input_w}): cls={pcc_cls:.6f}, box={pcc_box:.6f}, obj={pcc_obj:.6f}, kpt={pcc_kpt:.6f}"
    )

    min_pcc = min(pcc_cls, pcc_box, pcc_obj, pcc_kpt)

    assert cls_pass, f"cls PCC {pcc_cls:.4f} < {pcc_threshold}"
    assert box_pass, f"box PCC {pcc_box:.4f} < {pcc_threshold}"
    assert obj_pass, f"obj PCC {pcc_obj:.4f} < {pcc_threshold}"
    assert kpt_pass, f"kpt PCC {pcc_kpt:.4f} < {pcc_threshold}"

    logger.info(f"PCC test passed! Min PCC: {min_pcc:.6f}")
