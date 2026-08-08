# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: TTNN FPN vs PyTorch reference.

Tests the FPN neck independently by feeding it known backbone outputs
and comparing against the PyTorch reference FPN.

Run with:
  cd $TT_METAL_HOME
  source python_env/bin/activate
  export ARCH_NAME=wormhole_b0
  export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
  export TT_METAL_HOME=$(pwd)
  export PYTHONPATH=$(pwd):$HOME/.local/lib/python3.10/site-packages
  pytest models/experimental/atss_swin_l_dyhead/tests/pcc/test_ttnn_fpn.py -v
"""

import pytest
import torch
import ttnn

from loguru import logger
from models.common.utility_functions import comp_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_ttnn_fpn_vs_reference(device, atss_ckpt_path, atss_ref_model):
    """
    Compare TTNN FPN output against PyTorch reference FPN.
    Uses the same backbone features as input to both.
    """
    from models.experimental.atss_swin_l_dyhead.tt.tt_fpn import TtFPN
    from models.experimental.atss_swin_l_dyhead.tt.weight_loading import load_fpn_weights
    from models.experimental.atss_swin_l_dyhead.common import (
        ATSS_FPN_IN_CHANNELS,
        ATSS_FPN_OUT_CHANNELS,
        ATSS_FPN_NUM_OUTS,
    )

    # Generate backbone features using reference model
    torch.manual_seed(42)
    sample_input = torch.randint(0, 256, (1, 3, 640, 640), dtype=torch.float32)
    x = atss_ref_model.preprocess(sample_input)

    with torch.no_grad():
        backbone_feats = atss_ref_model.backbone(x)
        ref_fpn_feats = atss_ref_model.fpn(tuple(backbone_feats))

    assert len(ref_fpn_feats) == 5
    logger.info(f"Reference FPN shapes: {[f.shape for f in ref_fpn_feats]}")

    # Load TTNN FPN
    fpn_params = load_fpn_weights(
        atss_ckpt_path,
        device,
        in_channels=tuple(ATSS_FPN_IN_CHANNELS),
        out_channels=ATSS_FPN_OUT_CHANNELS,
        num_outs=ATSS_FPN_NUM_OUTS,
    )
    ttnn_fpn = TtFPN(
        device,
        fpn_params,
        in_channels=tuple(ATSS_FPN_IN_CHANNELS),
        out_channels=ATSS_FPN_OUT_CHANNELS,
        num_outs=ATSS_FPN_NUM_OUTS,
    )

    # Convert backbone features to TTNN (NCHW)
    ttnn_inputs = []
    for feat in backbone_feats:
        t = ttnn.from_torch(
            feat,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_inputs.append(t)

    # Run TTNN FPN
    ttnn_fpn_feats = ttnn_fpn(ttnn_inputs)
    assert len(ttnn_fpn_feats) == 5

    # Compare per-level
    pcc_threshold = 0.98
    for i, (ref_feat, ttnn_feat) in enumerate(zip(ref_fpn_feats, ttnn_fpn_feats)):
        ttnn_out = ttnn.to_torch(ttnn.from_device(ttnn_feat))
        assert (
            ttnn_out.shape == ref_feat.shape
        ), f"FPN level {i} shape mismatch: TTNN {ttnn_out.shape} vs ref {ref_feat.shape}"
        passing, pcc_val = comp_pcc(ref_feat, ttnn_out, pcc_threshold)
        logger.info(f"FPN level {i}: shape={list(ref_feat.shape)}, PCC={pcc_val:.6f}, pass={passing}")
        assert passing, f"FPN level {i} PCC {pcc_val:.6f} < {pcc_threshold}"
