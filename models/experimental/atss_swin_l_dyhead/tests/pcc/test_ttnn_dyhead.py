# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: TTNN DyHead vs PyTorch reference.
"""

import pytest
import torch

from loguru import logger
from models.common.utility_functions import comp_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_ttnn_dyhead_vs_reference(device, atss_ckpt_path, atss_ref_model):
    from models.experimental.atss_swin_l_dyhead.tt.tt_dyhead import TtHybridDyHead
    from models.experimental.atss_swin_l_dyhead.tt.weight_loading import load_dyhead_weights
    from models.experimental.atss_swin_l_dyhead.common import (
        ATSS_DYHEAD_NUM_BLOCKS,
        ATSS_DYHEAD_IN_CHANNELS,
        ATSS_DYHEAD_OUT_CHANNELS,
    )

    torch.manual_seed(42)

    # 1. Prepare Input
    sample_input = torch.randint(0, 256, (1, 3, 640, 640), dtype=torch.float32)
    x = atss_ref_model.preprocess(sample_input)

    # 2. Get PyTorch Reference Output
    with torch.no_grad():
        backbone_feats = atss_ref_model.backbone(x)
        fpn_feats = atss_ref_model.fpn(tuple(backbone_feats))
        expected_dy_feats = atss_ref_model.dyhead(list(fpn_feats))

    # 3. Load TTNN Hybrid Model
    head_params = load_dyhead_weights(
        atss_ckpt_path,
        device,
        num_blocks=ATSS_DYHEAD_NUM_BLOCKS,
        in_channels=ATSS_DYHEAD_IN_CHANNELS,
        out_channels=ATSS_DYHEAD_OUT_CHANNELS,
    )

    ttnn_head = TtHybridDyHead(
        device,
        atss_ref_model.dyhead,
    )

    # 4. Run TTNN Forward Pass
    # Simulate the TTNN FPN output by permuting the PyTorch NCHW feats to NHWC
    fpn_feats_nhwc = [f.permute(0, 2, 3, 1).contiguous() for f in fpn_feats]
    actual_dy_feats = ttnn_head(fpn_feats_nhwc)

    # 5. Compare Results using PCC
    # DyHead returns a list of tensors (one per pyramid level)
    pcc_threshold = 0.98

    for i, (golden, actual) in enumerate(zip(expected_dy_feats, actual_dy_feats)):
        # Convert actual from NHWC back to NCHW for comparison against golden ref
        actual = actual.permute(0, 3, 1, 2)
        passing, pcc_val = comp_pcc(golden, actual, pcc_threshold)
        assert (
            golden.shape == actual.shape
        ), f"Dyhead level {i} shape mismatch: TTNN {golden.shape} vs ref {actual.shape}"
        logger.info(f"Dyhead level {i}: shape={list(golden.shape)}, PCC={pcc_val:.6f}, pass={passing}")
        assert passing, f"Dyhead level {i} PCC {pcc_val:.6f} < {pcc_threshold}"
