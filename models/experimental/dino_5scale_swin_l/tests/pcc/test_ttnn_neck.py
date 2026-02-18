# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: TTNN ChannelMapper neck vs PyTorch reference.
Tests all 5 output levels (P2-P6) against mmdet neck.

Run with:
  export PYTHONPATH=/home/ubuntu/tt-metal:$HOME/.local/lib/python3.10/site-packages
  pytest models/experimental/dino_5scale_swin_l/tests/pcc/test_ttnn_neck.py -v
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.dino_5scale_swin_l.common import (
    DINO_INPUT_H,
    DINO_INPUT_W,
    NECK_IN_CHANNELS,
    NECK_OUT_CHANNELS,
    NUM_LEVELS,
)
from loguru import logger


def _mmdet_importable():
    try:
        from mmdet.apis import init_detector  # noqa: F401

        return True
    except ImportError:
        return False


def _get_ckpt_and_config():
    import os
    from pathlib import Path

    base = Path(os.environ.get("TT_METAL_HOME", Path.cwd()))
    config = base / "models/experimental/dino_5scale_swin_l/reference/dino_5scale_swin_l.py"
    ckpt_dir = base / "models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l"
    ckpt = ckpt_dir / "dino_5scale_swin_l.pth"
    if not ckpt.is_file():
        ckpt = ckpt_dir / "dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"
    return str(config), str(ckpt)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_ttnn_neck_pcc(device, reset_seeds):
    """Compare TTNN ChannelMapper neck output vs PyTorch mmdet for all 5 levels."""
    if not _mmdet_importable():
        pytest.skip("mmdet not importable")

    config_path, ckpt_path = _get_ckpt_and_config()
    from pathlib import Path

    if not Path(ckpt_path).is_file():
        pytest.skip("Checkpoint not found")

    # --- PyTorch reference: backbone + neck ---
    from models.experimental.dino_5scale_swin_l.reference.dino_staged_forward import DINOStagedForward

    ref = DINOStagedForward(config_path, ckpt_path)
    torch_input = torch.rand(1, 3, DINO_INPUT_H, DINO_INPUT_W)

    with torch.no_grad():
        backbone_feats = ref.forward_backbone(torch_input)
        neck_out = ref.model.neck(backbone_feats)
    torch_neck_feats = list(neck_out)
    assert len(torch_neck_feats) == NUM_LEVELS

    # --- TTNN backbone (reuse swin_l) ---
    from models.experimental.swin_l.tt import TtSwinLBackbone, load_backbone_weights, compute_attn_masks
    from models.experimental.dino_5scale_swin_l.tt.model_preprocessing import load_neck_weights
    from models.experimental.dino_5scale_swin_l.tt.tt_neck import TtDINONeck
    from models.experimental.swin_l.common import (
        SWIN_L_EMBED_DIM,
        SWIN_L_DEPTHS,
        SWIN_L_NUM_HEADS,
        SWIN_L_WINDOW_SIZE,
    )

    backbone_params = load_backbone_weights(ckpt_path, device)
    attn_masks = compute_attn_masks(DINO_INPUT_H, DINO_INPUT_W, 4, SWIN_L_WINDOW_SIZE, device)
    ttnn_backbone = TtSwinLBackbone(
        device,
        backbone_params,
        embed_dim=SWIN_L_EMBED_DIM,
        depths=tuple(SWIN_L_DEPTHS),
        num_heads=tuple(SWIN_L_NUM_HEADS),
        window_size=SWIN_L_WINDOW_SIZE,
        attn_masks=attn_masks,
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_backbone_feats = ttnn_backbone(ttnn_input)

    # --- TTNN neck ---
    neck_params = load_neck_weights(ckpt_path, device)
    ttnn_neck = TtDINONeck(
        device,
        neck_params,
        in_channels=tuple(NECK_IN_CHANNELS),
        out_channels=NECK_OUT_CHANNELS,
    )
    ttnn_neck_feats = ttnn_neck(ttnn_backbone_feats)
    assert len(ttnn_neck_feats) == NUM_LEVELS

    # --- Compare per-level ---
    pcc_threshold = 0.95
    level_names = ["P2", "P3", "P4", "P5", "P6"]
    for i, (torch_feat, ttnn_feat) in enumerate(zip(torch_neck_feats, ttnn_neck_feats)):
        ttnn_out = ttnn.to_torch(ttnn.from_device(ttnn_feat))
        assert ttnn_out.shape == torch_feat.shape, (
            f"{level_names[i]} shape mismatch: TTNN {ttnn_out.shape} vs PyTorch {torch_feat.shape}"
        )
        passing, pcc_val = comp_pcc(torch_feat, ttnn_out, pcc_threshold)
        logger.info(f"{level_names[i]}: shape={list(torch_feat.shape)}, PCC={pcc_val:.6f}, pass={passing}")
        assert passing, f"{level_names[i]} PCC {pcc_val:.6f} < {pcc_threshold}"
