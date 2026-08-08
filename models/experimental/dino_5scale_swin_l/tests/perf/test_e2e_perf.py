# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
DINO-5scale Swin-L E2E Performance Test with Python Checkpoints.

Measures wall-clock time for a single inference pass through the full pipeline,
with per-module timing breakdowns:
  Backbone → Neck → Pre-Transformer → Encoder → Pre-Decoder → Decoder → Heads

No tracing or 2CQ — just raw single-iteration latency.

Usage:
    pytest models/experimental/dino_5scale_swin_l/tests/perf/test_e2e_perf.py -v
"""

import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

from models.experimental.dino_5scale_swin_l.common import (
    DINO_INPUT_H,
    DINO_INPUT_W,
    SWIN_L_EMBED_DIM,
    SWIN_L_DEPTHS,
    SWIN_L_NUM_HEADS,
    SWIN_L_WINDOW_SIZE,
    NUM_QUERIES,
    NUM_CLASSES,
    NUM_LEVELS,
    ENCODER_EMBED_DIMS,
    ENCODER_NUM_HEADS,
    ENCODER_NUM_POINTS,
    ENCODER_NUM_LAYERS,
    DECODER_NUM_LAYERS,
)


def _get_ckpt_path():
    import os

    base = Path(os.environ.get("TT_METAL_HOME", Path.cwd()))
    ckpt_dir = base / "models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l"
    ckpt = ckpt_dir / "dino_5scale_swin_l.pth"
    if not ckpt.is_file():
        ckpt = ckpt_dir / "dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"
    return str(ckpt)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_dino_5scale_e2e_perf(device):
    """
    Single-iteration E2E wall-clock latency for DINO-5scale Swin-L.

    Reports total time and per-module breakdown.
    """
    ckpt_path = _get_ckpt_path()
    if not Path(ckpt_path).is_file():
        pytest.skip("Checkpoint not found")

    from models.experimental.swin_l.tt.model_preprocessing import load_backbone_weights, compute_attn_masks
    from models.experimental.dino_5scale_swin_l.tt.model_preprocessing import (
        load_neck_weights,
        load_encoder_weights,
        load_decoder_weights,
        _resolve_state_dict,
    )
    from models.experimental.dino_5scale_swin_l.tt.tt_dino import TtDINO

    # --- Load weights (one-time cost, not measured) ---
    logger.info("Loading checkpoint...")
    t_load = time.perf_counter()
    sd = _resolve_state_dict(ckpt_path)
    backbone_params = load_backbone_weights(
        sd,
        device,
        embed_dim=SWIN_L_EMBED_DIM,
        depths=tuple(SWIN_L_DEPTHS),
        num_heads=tuple(SWIN_L_NUM_HEADS),
        window_size=SWIN_L_WINDOW_SIZE,
    )
    neck_params = load_neck_weights(sd, device)
    encoder_params = load_encoder_weights(sd, device)
    decoder_params = load_decoder_weights(sd, device)
    attn_masks = compute_attn_masks(DINO_INPUT_H, DINO_INPUT_W, 4, SWIN_L_WINDOW_SIZE, device)
    del sd

    tt_model = TtDINO(
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        device=device,
        backbone_params=backbone_params,
        neck_params=neck_params,
        attn_masks=attn_masks,
        num_queries=NUM_QUERIES,
        num_classes=NUM_CLASSES,
        num_levels=NUM_LEVELS,
        embed_dims=ENCODER_EMBED_DIMS,
        num_heads=ENCODER_NUM_HEADS,
        num_points=ENCODER_NUM_POINTS,
        encoder_num_layers=ENCODER_NUM_LAYERS,
        decoder_num_layers=DECODER_NUM_LAYERS,
        pe_temperature=20,
        embed_dim=SWIN_L_EMBED_DIM,
        depths=tuple(SWIN_L_DEPTHS),
        backbone_num_heads=tuple(SWIN_L_NUM_HEADS),
        window_size=SWIN_L_WINDOW_SIZE,
        in_channels=(192, 384, 768, 1536),
    )
    t_load_end = time.perf_counter()
    logger.info(f"Weight loading + model init: {t_load_end - t_load:.2f}s")

    # --- Prepare input ---
    torch_input = torch.randn(1, 3, DINO_INPUT_H, DINO_INPUT_W)

    # --- E2E inference with per-module checkpoints ---
    logger.info("Starting E2E inference...")

    t_start = time.perf_counter()

    # Backbone
    image_tt = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    t_h2d = time.perf_counter()

    backbone_feats_tt = tt_model.backbone(image_tt)
    ttnn.synchronize_device(device)
    t_backbone = time.perf_counter()

    # Neck
    neck_feats_tt = tt_model.neck(backbone_feats_tt)
    ttnn.synchronize_device(device)
    t_neck = time.perf_counter()

    neck_feats_torch = []
    for nf in neck_feats_tt:
        neck_feats_torch.append(ttnn.to_torch(ttnn.from_device(nf)).float())
        ttnn.deallocate(nf)
    for bf in backbone_feats_tt:
        ttnn.deallocate(bf)

    # Pre-transformer (host)
    pre_trans = tt_model.pre_transformer(neck_feats_torch)
    t_pre_trans = time.perf_counter()

    feat_tt = ttnn.from_torch(
        pre_trans["feat_flatten"].to(torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    feat_pos_tt = ttnn.from_torch(
        pre_trans["feat_pos"].to(torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    # Encoder
    memory_tt = tt_model.encoder(
        feat=feat_tt,
        feat_pos=feat_pos_tt,
        feat_mask=None,
        spatial_shapes=pre_trans["spatial_shapes"],
        level_start_index=pre_trans["level_start_index"],
        valid_ratios=pre_trans["valid_ratios"],
    )
    ttnn.deallocate(feat_tt)
    ttnn.deallocate(feat_pos_tt)
    ttnn.synchronize_device(device)
    t_encoder = time.perf_counter()

    # Pre-decoder (host float32)
    pre_dec = tt_model.pre_decoder(memory_tt, pre_trans["spatial_shapes"])
    t_pre_dec = time.perf_counter()

    # Decoder
    hidden_states, references = tt_model.decoder(
        query=pre_dec["query"],
        value=memory_tt,
        key_padding_mask=None,
        self_attn_mask=None,
        reference_points=pre_dec["reference_points"],
        spatial_shapes=pre_trans["spatial_shapes"],
        level_start_index=pre_trans["level_start_index"],
        valid_ratios=pre_trans["valid_ratios"],
    )
    ttnn.synchronize_device(device)
    t_decoder = time.perf_counter()

    # Detection heads
    all_cls, all_coords = tt_model.forward_heads(hidden_states, references)
    ttnn.synchronize_device(device)
    t_heads = time.perf_counter()

    # Post-processing
    detections = TtDINO.postprocess(
        all_cls[-1],
        all_coords[-1],
        img_shape=(DINO_INPUT_H, DINO_INPUT_W),
    )
    t_end = time.perf_counter()

    # --- Report ---
    total = t_end - t_start
    timings = {
        "Host→Device": (t_h2d - t_start) * 1000,
        "Backbone": (t_backbone - t_h2d) * 1000,
        "Neck": (t_neck - t_backbone) * 1000,
        "Pre-Transformer": (t_pre_trans - t_neck) * 1000,
        "Encoder": (t_encoder - t_pre_trans) * 1000,
        "Pre-Decoder": (t_pre_dec - t_encoder) * 1000,
        "Decoder": (t_decoder - t_pre_dec) * 1000,
        "Heads": (t_heads - t_decoder) * 1000,
        "Post-Processing": (t_end - t_heads) * 1000,
    }

    logger.info("")
    logger.info("=" * 65)
    logger.info("DINO-5scale Swin-L  E2E Performance (800x1333, 1 iteration)")
    logger.info("=" * 65)
    for name, ms in timings.items():
        pct = ms / (total * 1000) * 100
        logger.info(f"  {name:20s}  {ms:9.1f} ms  ({pct:5.1f}%)")
    logger.info(f"  {'─' * 45}")
    logger.info(f"  {'TOTAL':20s}  {total * 1000:9.1f} ms  (100.0%)")
    logger.info(f"  {'FPS':20s}  {1.0 / total:9.2f}")
    logger.info("=" * 65)
    logger.info(f"  Detections: {len(detections['boxes'])} boxes")
    logger.info("=" * 65)
