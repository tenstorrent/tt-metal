# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: TTNN Swin-L MLP (FFN) vs PyTorch reference.
Tests FFN at stage 0 block 0.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.dino_5scale_swin_l.common import (
    DINO_INPUT_H,
    DINO_INPUT_W,
    SWIN_L_EMBED_DIM,
)
from loguru import logger


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_ttnn_swin_mlp_pcc(device, swin_l_ref, swin_l_ckpt_path, reset_seeds):
    """Compare TTNN MLP output vs PyTorch at stage 0 block 0."""
    from models.experimental.swin_l.tt.tt_swin_mlp import TtSwinMLP
    from models.experimental.swin_l.tt.model_preprocessing import load_backbone_weights

    stage_idx, block_idx = 0, 0

    # Get input
    torch_input = torch.rand(1, 3, DINO_INPUT_H, DINO_INPUT_W)
    x_nhwc, hw = swin_l_ref.get_patch_embed_output(torch_input)

    # PyTorch reference FFN
    torch_out = swin_l_ref.forward_ffn(x_nhwc, hw, stage_idx, block_idx)

    # TTNN
    params = load_backbone_weights(swin_l_ckpt_path, device)
    block_params = params["stages"][stage_idx]["blocks"][block_idx]

    ttnn_mlp = TtSwinMLP(device, block_params["mlp"], dim=SWIN_L_EMBED_DIM, mlp_ratio=4.0)

    # Apply norm2 first (same as reference)
    ttnn_x = ttnn.from_torch(
        x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_x = ttnn.to_layout(ttnn_x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn_normed = ttnn.layer_norm(
        ttnn_x,
        weight=block_params["norm2"]["weight"],
        bias=block_params["norm2"]["bias"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_out = ttnn_mlp(ttnn_normed)

    result = ttnn.to_torch(ttnn.from_device(ttnn_out))
    passing, pcc_val = comp_pcc(torch_out, result, 0.97)
    logger.info(f"MLP stage={stage_idx} block={block_idx}: PCC={pcc_val:.6f}")
    assert passing, f"MLP PCC {pcc_val:.6f} < 0.97"
