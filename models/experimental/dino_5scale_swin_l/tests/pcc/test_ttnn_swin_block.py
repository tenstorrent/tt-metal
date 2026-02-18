# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: TTNN Swin-L transformer block vs PyTorch reference.
Tests full block (LN → attention → residual → LN → FFN → residual).
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
@pytest.mark.parametrize(
    "stage_idx,block_idx",
    [(0, 0), (0, 1), (2, 0)],
    ids=["s0_b0_no_shift", "s0_b1_shift", "s2_b0_deep"],
)
def test_ttnn_swin_block_pcc(device, swin_l_ref, swin_l_ckpt_path, stage_idx, block_idx, reset_seeds):
    """Compare TTNN Swin block output vs PyTorch."""
    from models.experimental.swin_l.tt.tt_swin_block import TtSwinBlock
    from models.experimental.swin_l.tt.model_preprocessing import (
        load_backbone_weights, compute_attn_masks,
    )

    ws = SWIN_L_WINDOW_SIZE

    # Get input for the target stage
    torch_input = torch.rand(1, 3, DINO_INPUT_H, DINO_INPUT_W)
    x_nhwc, hw = swin_l_ref.get_stage_input(torch_input, target_stage=stage_idx)

    # If block_idx > 0, run preceding blocks to get the right input
    B, H, W, C = x_nhwc.shape
    x_flat = x_nhwc.view(B, H * W, C)
    with torch.no_grad():
        stage = swin_l_ref.backbone.stages[stage_idx]
        for b in range(block_idx):
            x_flat = stage.blocks[b](x_flat, hw)
    x_nhwc = x_flat.view(B, H, W, C)

    # PyTorch reference block
    torch_out = swin_l_ref.forward_block(x_nhwc, hw, stage_idx, block_idx)

    # TTNN
    params = load_backbone_weights(swin_l_ckpt_path, device)
    attn_masks = compute_attn_masks(DINO_INPUT_H, DINO_INPUT_W, 4, ws, device)
    dim = SWIN_L_EMBED_DIM * (2 ** stage_idx)
    shift = [0, 0] if block_idx % 2 == 0 else [ws // 2, ws // 2]

    ttnn_block = TtSwinBlock(
        device, params["stages"][stage_idx]["blocks"][block_idx],
        dim=dim, num_heads=SWIN_L_NUM_HEADS[stage_idx],
        window_size=[ws, ws], shift_size=shift,
        attn_mask=attn_masks[stage_idx],
    )

    ttnn_x = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn_x = ttnn.to_layout(ttnn_x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn_out = ttnn_block(ttnn_x)

    result = ttnn.to_torch(ttnn.from_device(ttnn_out))
    passing, pcc_val = comp_pcc(torch_out, result, 0.97)
    logger.info(f"Block stage={stage_idx} block={block_idx}: PCC={pcc_val:.6f}")
    assert passing, f"Block PCC {pcc_val:.6f} < 0.97"
