# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: TTNN Swin-L patch merging (downsample) vs PyTorch reference.
Tests downsample at end of stage 0 and stage 1.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.dino_5scale_swin_l.common import (
    DINO_INPUT_H, DINO_INPUT_W, SWIN_L_EMBED_DIM, SWIN_L_NUM_HEADS, SWIN_L_WINDOW_SIZE,
)
from loguru import logger


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("stage_idx", [0, 1], ids=["stage0_ds", "stage1_ds"])
def test_ttnn_swin_patch_merge_pcc(device, swin_l_ref, swin_l_ckpt_path, stage_idx, reset_seeds):
    """Compare TTNN patch merge output vs PyTorch."""
    from models.experimental.swin_l.tt.tt_swin_patch_merge import TtSwinPatchMerge
    from models.experimental.swin_l.tt.model_preprocessing import load_backbone_weights

    # Get input: output of all blocks in target stage (before downsample)
    torch_input = torch.rand(1, 3, DINO_INPUT_H, DINO_INPUT_W)
    x_nhwc, hw = swin_l_ref.get_stage_input(torch_input, target_stage=stage_idx)

    # Run all blocks in the stage to get the pre-downsample tensor
    B, H, W, C = x_nhwc.shape
    x_flat = x_nhwc.view(B, H * W, C)
    with torch.no_grad():
        stage = swin_l_ref.backbone.stages[stage_idx]
        for blk in stage.blocks:
            x_flat = blk(x_flat, hw)
    pre_ds_nhwc = x_flat.view(B, H, W, C)

    # PyTorch reference patch merge
    torch_out, new_hw = swin_l_ref.forward_patch_merge(pre_ds_nhwc, hw, stage_idx)

    # TTNN
    params = load_backbone_weights(swin_l_ckpt_path, device)
    dim = SWIN_L_EMBED_DIM * (2 ** stage_idx)
    ttnn_ds = TtSwinPatchMerge(device, params["stages"][stage_idx]["downsample"], dim=dim)

    ttnn_x = ttnn.from_torch(pre_ds_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn_out = ttnn_ds(ttnn_x)

    result = ttnn.to_torch(ttnn.from_device(ttnn_out))
    passing, pcc_val = comp_pcc(torch_out, result, 0.97)
    logger.info(f"PatchMerge stage={stage_idx}: out_shape={list(torch_out.shape)}, PCC={pcc_val:.6f}")
    assert passing, f"PatchMerge PCC {pcc_val:.6f} < 0.97"
