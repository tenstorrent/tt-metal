# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: TTNN Swin-L shifted window attention vs PyTorch reference.
Tests attention at stage 0 block 0 (no shift) and block 1 (with shift).
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
@pytest.mark.parametrize("block_idx", [0, 1], ids=["no_shift", "with_shift"])
def test_ttnn_swin_attention_pcc(device, swin_l_ref, swin_l_ckpt_path, block_idx, reset_seeds):
    """Compare TTNN attention output vs PyTorch at stage 0."""
    from models.experimental.swin_l.tt.tt_swin_attention import TtSwinAttention
    from models.experimental.swin_l.tt.model_preprocessing import (
        load_backbone_weights, compute_attn_masks,
    )

    stage_idx = 0
    ws = SWIN_L_WINDOW_SIZE

    # Get PyTorch input (output of patch_embed)
    torch_input = torch.rand(1, 3, DINO_INPUT_H, DINO_INPUT_W)
    x_nhwc, hw = swin_l_ref.get_patch_embed_output(torch_input)

    # PyTorch reference attention
    torch_out = swin_l_ref.forward_attention(x_nhwc, hw, stage_idx, block_idx)

    # TTNN
    params = load_backbone_weights(swin_l_ckpt_path, device)
    attn_masks = compute_attn_masks(DINO_INPUT_H, DINO_INPUT_W, 4, ws, device)
    block_params = params["stages"][stage_idx]["blocks"][block_idx]

    shift = [0, 0] if block_idx % 2 == 0 else [ws // 2, ws // 2]
    ttnn_attn = TtSwinAttention(
        device, block_params["attn"],
        dim=SWIN_L_EMBED_DIM, window_size=[ws, ws], shift_size=shift,
        num_heads=SWIN_L_NUM_HEADS[stage_idx], attn_mask=attn_masks[stage_idx],
    )

    # Need to apply norm1 first (same as reference does)
    ttnn_x = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn_x = ttnn.to_layout(ttnn_x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn_normed = ttnn.layer_norm(
        ttnn_x, weight=block_params["norm1"]["weight"], bias=block_params["norm1"]["bias"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_normed = ttnn.to_layout(ttnn_normed, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn_out = ttnn_attn(ttnn_normed)

    result = ttnn.to_torch(ttnn.from_device(ttnn_out))
    passing, pcc_val = comp_pcc(torch_out, result, 0.97)
    logger.info(f"Attention stage={stage_idx} block={block_idx}: PCC={pcc_val:.6f}")
    assert passing, f"Attention PCC {pcc_val:.6f} < 0.97"
