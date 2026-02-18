# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: TTNN DINO encoder vs PyTorch reference (mmdet encoder).

Uses PyTorch reference neck output as input to isolate encoder testing
and avoid DRAM OOM from loading backbone + neck + encoder weights simultaneously.

Run with:
  export PYTHONPATH=/home/ubuntu/tt-metal:$HOME/.local/lib/python3.10/site-packages
  pytest models/experimental/dino_5scale_swin_l/tests/pcc/test_ttnn_encoder.py -v
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.dino_5scale_swin_l.common import (
    DINO_INPUT_H,
    DINO_INPUT_W,
    ENCODER_NUM_LAYERS,
    ENCODER_NUM_HEADS,
    ENCODER_EMBED_DIMS,
    ENCODER_NUM_POINTS,
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
def test_ttnn_encoder_pcc(device, reset_seeds):
    """Compare TTNN encoder output vs PyTorch mmdet encoder."""
    if not _mmdet_importable():
        pytest.skip("mmdet not importable")

    config_path, ckpt_path = _get_ckpt_and_config()
    from pathlib import Path

    if not Path(ckpt_path).is_file():
        pytest.skip("Checkpoint not found")

    # --- PyTorch reference: backbone + neck + encoder (all on CPU) ---
    from models.experimental.dino_5scale_swin_l.reference.dino_staged_forward import DINOStagedForward

    ref = DINOStagedForward(config_path, ckpt_path)
    torch_input = torch.rand(1, 3, DINO_INPUT_H, DINO_INPUT_W)

    with torch.no_grad():
        backbone_feats = ref.forward_backbone(torch_input)
        neck_out = list(ref.model.neck(backbone_feats))

    ref_encoder_out = ref.forward_encoder(neck_out)
    ref_memory = ref_encoder_out["memory"]
    spatial_shapes = ref_encoder_out["spatial_shapes"]
    level_start_index = ref_encoder_out["level_start_index"]
    valid_ratios = ref_encoder_out["valid_ratios"]
    feat_pos_torch = ref_encoder_out["feat_pos"]

    logger.info(f"Reference encoder output shape: {ref_memory.shape}")
    logger.info(f"Spatial shapes: {spatial_shapes}")

    # Flatten PyTorch neck features to [B, N, 256] for TTNN input
    flat_feats = []
    for feat in neck_out:
        B, C, H, W = feat.shape
        flat_feats.append(feat.flatten(2).permute(0, 2, 1))
    feat_flatten = torch.cat(flat_feats, dim=1)
    logger.info(f"Flattened feat shape: {feat_flatten.shape}")

    # --- Transfer to TTNN device ---
    logger.info("Transferring feat to device...")
    feat_tt = ttnn.from_torch(
        feat_flatten,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    logger.info("Transferring feat_pos to device...")
    feat_pos_tt = ttnn.from_torch(
        feat_pos_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # --- TTNN Encoder ---
    logger.info("Loading encoder weights...")
    from models.experimental.dino_5scale_swin_l.tt.model_preprocessing import load_encoder_weights
    from models.experimental.dino_5scale_swin_l.tt.tt_encoder import TtDINOEncoder

    encoder_params = load_encoder_weights(ckpt_path, device)
    logger.info("Encoder weights loaded. Creating encoder...")
    ttnn_encoder = TtDINOEncoder(
        encoder_params,
        device,
        num_layers=ENCODER_NUM_LAYERS,
        embed_dims=ENCODER_EMBED_DIMS,
        num_heads=ENCODER_NUM_HEADS,
        num_levels=NUM_LEVELS,
        num_points=ENCODER_NUM_POINTS,
    )

    logger.info("Running TTNN encoder forward...")
    memory_tt = ttnn_encoder(
        feat=feat_tt,
        feat_pos=feat_pos_tt,
        feat_mask=None,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        valid_ratios=valid_ratios,
    )
    logger.info("TTNN encoder forward done.")

    memory_out = ttnn.to_torch(ttnn.from_device(memory_tt)).float()
    logger.info(f"TTNN encoder output shape: {memory_out.shape}")

    N = ref_memory.shape[1]
    memory_out = memory_out[:, :N, :]

    pcc_threshold = 0.90
    passing, pcc_val = comp_pcc(ref_memory, memory_out, pcc_threshold)
    logger.info(f"Encoder PCC: {pcc_val:.6f} (threshold={pcc_threshold})")
    assert passing, f"Encoder PCC {pcc_val:.6f} < {pcc_threshold}"
