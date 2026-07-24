# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: TTNN DINO decoder vs PyTorch reference (mmdet decoder).

Uses PyTorch reference encoder output + pre_decoder output as input,
isolating the decoder test from backbone/neck/encoder.

Run with:
  export PYTHONPATH=/home/ubuntu/tt-metal:$HOME/.local/lib/python3.10/site-packages
  pytest models/experimental/dino_5scale_swin_l/tests/pcc/test_ttnn_decoder.py -v
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.dino_5scale_swin_l.common import (
    DINO_INPUT_H,
    DINO_INPUT_W,
    DECODER_NUM_LAYERS,
    DECODER_NUM_HEADS,
    DECODER_EMBED_DIMS,
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


@pytest.mark.skipif(not _mmdet_importable(), reason="mmdet not installed")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_ttnn_decoder_pcc(device):
    config_path, ckpt_path = _get_ckpt_and_config()

    from models.experimental.dino_5scale_swin_l.reference.dino_staged_forward import DINOStagedForward

    ref_model = DINOStagedForward(config_path, ckpt_path, device="cpu")

    # Run full reference: backbone -> neck -> encoder -> pre_decoder -> decoder
    dummy_input = torch.randn(1, 3, DINO_INPUT_H, DINO_INPUT_W)
    backbone_feats = ref_model.forward_backbone(dummy_input)
    ref_model.forward_neck(backbone_feats)
    neck_features = ref_model.model.neck(backbone_feats)

    ref_decoder_out = ref_model.forward_decoder(
        img_feats=list(neck_features),
        batch_input_shape=(DINO_INPUT_H, DINO_INPUT_W),
        img_shape=(DINO_INPUT_H, DINO_INPUT_W),
    )

    ref_hidden_states = ref_decoder_out["hidden_states"]  # [num_layers, B, num_queries, 256]
    ref_references = ref_decoder_out["references"]  # list of [B, num_queries, 4]
    ref_memory = ref_decoder_out["memory"]  # [B, N, 256]
    ref_query = ref_decoder_out["query"]  # [B, num_queries, 256]
    ref_points_init = ref_decoder_out["reference_points_init"]  # [B, num_queries, 4]
    spatial_shapes = ref_decoder_out["spatial_shapes"]
    valid_ratios = ref_decoder_out["valid_ratios"]
    level_start_index = ref_decoder_out["level_start_index"]

    logger.info(f"Reference decoder: hidden_states shape: {ref_hidden_states.shape}")
    logger.info(f"Reference decoder: num reference point sets: {len(ref_references)}")
    logger.info(f"Reference decoder: query shape: {ref_query.shape}")
    logger.info(f"Reference decoder: ref_points_init shape: {ref_points_init.shape}")
    logger.info(f"Reference decoder: memory shape: {ref_memory.shape}")

    # --- Transfer inputs to TTNN device ---
    logger.info("Transferring decoder inputs to device...")
    query_tt = ttnn.from_torch(
        ref_query,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    memory_tt = ttnn.from_torch(
        ref_memory,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # --- Load decoder weights and create TTNN decoder ---
    logger.info("Loading decoder weights...")
    from models.experimental.dino_5scale_swin_l.tt.model_preprocessing import load_decoder_weights
    from models.experimental.dino_5scale_swin_l.tt.tt_decoder import TtDINODecoder

    decoder_params = load_decoder_weights(ckpt_path, device)
    logger.info("Decoder weights loaded. Creating decoder...")

    ttnn_decoder = TtDINODecoder(
        decoder_params,
        device,
        num_layers=DECODER_NUM_LAYERS,
        embed_dims=DECODER_EMBED_DIMS,
        num_heads=DECODER_NUM_HEADS,
        num_levels=NUM_LEVELS,
        num_points=ENCODER_NUM_POINTS,
    )

    # --- Run TTNN decoder ---
    logger.info("Running TTNN decoder forward...")
    intermediate, intermediate_ref_pts = ttnn_decoder(
        query=query_tt,
        value=memory_tt,
        key_padding_mask=None,
        self_attn_mask=None,
        reference_points=ref_points_init,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        valid_ratios=valid_ratios,
    )
    logger.info("TTNN decoder forward done.")

    # --- Compare per-layer hidden states ---
    ref_N = ref_hidden_states.shape[2]  # num_queries
    pcc_threshold = 0.98

    for i in range(DECODER_NUM_LAYERS):
        h = ttnn.to_torch(ttnn.from_device(intermediate[i])).float()[:, :ref_N, :]
        ref_h = ref_hidden_states[i]
        _, layer_pcc = comp_pcc(ref_h, h, 0.0)
        logger.info(f"  Layer {i} PCC: {layer_pcc:.6f}")

    last_hidden = ttnn.to_torch(ttnn.from_device(intermediate[-1])).float()[:, :ref_N, :]
    ref_last_hidden = ref_hidden_states[-1]
    passing, pcc_val = comp_pcc(ref_last_hidden, last_hidden, pcc_threshold)
    logger.info(f"Decoder last layer PCC: {pcc_val:.6f} (threshold={pcc_threshold})")
    assert passing, f"Decoder last layer PCC {pcc_val:.6f} < {pcc_threshold}"
