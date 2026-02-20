# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: TTNN ATSS Head vs PyTorch reference.

Tests the ATSS detection head independently by feeding it known DyHead outputs
and comparing against the PyTorch reference ATSS head.
"""

import pytest
import torch
import ttnn

from loguru import logger
from models.common.utility_functions import comp_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_ttnn_atss_head_vs_reference(device, atss_ckpt_path, atss_ref_model):
    """
    Compare TTNN ATSS Head output against PyTorch reference.
    Uses the same DyHead features as input to both.
    """
    from models.experimental.atss_swin_l_dyhead.tt.tt_atss_head import TtATSSHead
    from models.experimental.atss_swin_l_dyhead.tt.weight_loading import load_atss_head_weights
    from models.experimental.atss_swin_l_dyhead.common import (
        ATSS_NUM_CLASSES,
        ATSS_FPN_OUT_CHANNELS,
        ATSS_NUM_ANCHORS,
        ATSS_FPN_NUM_OUTS,
    )

    # Generate DyHead features using reference model
    torch.manual_seed(42)
    sample_input = torch.randint(0, 256, (1, 3, 640, 640), dtype=torch.float32)
    x = atss_ref_model.preprocess(sample_input)

    with torch.no_grad():
        backbone_feats = atss_ref_model.backbone(x)
        fpn_feats = atss_ref_model.fpn(tuple(backbone_feats))
        dy_feats = atss_ref_model.dyhead(list(fpn_feats))
        ref_cls, ref_reg, ref_cent = atss_ref_model.head(tuple(dy_feats))

    logger.info(
        f"Reference head: cls={[c.shape for c in ref_cls]}, "
        f"reg={[r.shape for r in ref_reg]}, cent={[c.shape for c in ref_cent]}"
    )

    # Load TTNN ATSS Head
    head_params = load_atss_head_weights(
        atss_ckpt_path,
        device,
        num_classes=ATSS_NUM_CLASSES,
        num_anchors=ATSS_NUM_ANCHORS,
        num_levels=ATSS_FPN_NUM_OUTS,
    )
    ttnn_head = TtATSSHead(
        device,
        head_params,
        num_classes=ATSS_NUM_CLASSES,
        in_channels=ATSS_FPN_OUT_CHANNELS,
        num_anchors=ATSS_NUM_ANCHORS,
        num_levels=ATSS_FPN_NUM_OUTS,
    )

    # Convert DyHead features to TTNN (NCHW)
    ttnn_inputs = []
    for feat in dy_feats:
        t = ttnn.from_torch(
            # feat,
            feat.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            # layout=ttnn.ROW_MAJOR_LAYOUT,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            # memory_config=ttnn.DRAM_MEMORY_CONFIG,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn_inputs.append(t)

    # Run TTNN ATSS Head
    ttnn_cls, ttnn_reg, ttnn_cent = ttnn_head(ttnn_inputs)

    # Compare per-level
    pcc_threshold = 0.98
    for i in range(5):
        # Classification
        cls_out = ttnn.to_torch(ttnn.from_device(ttnn_cls[i]))
        N, C, H, W = ref_cls[i].shape
        cls_out = torch.reshape(cls_out, (N, H, W, C))  # Reshape into NHWC based on torch
        cls_out = torch.permute(cls_out, (0, 3, 1, 2))  # Convert cls_out to NCHW
        assert cls_out.shape == ref_cls[i].shape
        passing, pcc_val = comp_pcc(ref_cls[i], cls_out, pcc_threshold)
        logger.info(f"Head level {i} cls: PCC={pcc_val:.6f}, pass={passing}")
        assert passing, f"cls level {i} PCC {pcc_val:.6f} < {pcc_threshold}"

        # Regression
        reg_out = ttnn.to_torch(ttnn.from_device(ttnn_reg[i]))
        N, C, H, W = ref_reg[i].shape
        reg_out = torch.reshape(reg_out, (N, H, W, C))  # Reshape into NHWC based on torch
        reg_out = torch.permute(reg_out, (0, 3, 1, 2))  # Convert cls_out to NCHW
        assert reg_out.shape == ref_reg[i].shape
        passing, pcc_val = comp_pcc(ref_reg[i], reg_out, pcc_threshold)
        logger.info(f"Head level {i} reg: PCC={pcc_val:.6f}, pass={passing}")
        assert passing, f"reg level {i} PCC {pcc_val:.6f} < {pcc_threshold}"

        # Centerness
        cent_out = ttnn.to_torch(ttnn.from_device(ttnn_cent[i]))
        N, C, H, W = ref_cent[i].shape
        cent_out = torch.reshape(cent_out, (N, H, W, C))  # Reshape into NHWC based on torch
        cent_out = torch.permute(cent_out, (0, 3, 1, 2))  # Convert cls_out to NCHW
        assert cent_out.shape == ref_cent[i].shape
        passing, pcc_val = comp_pcc(ref_cent[i], cent_out, pcc_threshold)
        logger.info(f"Head level {i} cent: PCC={pcc_val:.6f}, pass={passing}")
        assert passing, f"cent level {i} PCC {pcc_val:.6f} < {pcc_threshold}"
