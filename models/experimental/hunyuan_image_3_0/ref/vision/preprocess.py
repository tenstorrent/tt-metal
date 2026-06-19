# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# PyTorch reference for the HunyuanImage-3.0 Instruct (image-to-image) image
# preprocessing — the host front-end that turns a PIL image into the SigLIP2
# `pixel_values` / `spatial_shapes` / `pixel_attention_mask` the vision tower
# consumes. Golden for `tt/vision/preprocess.py`.
#
# Extracted / adapted from:
#   HunyuanImage-3.0/hunyuan_image_3/image_processor.py
#     HunyuanImage3ImageProcessor.__init__        lines 224-271
#       (ViT processor build: Siglip2ImageProcessorFast.from_dict(config.vit_processor),
#        max_token_length = processor.max_num_patches)
#     HunyuanImage3ImageProcessor.vit_process_image  lines 422-451
#
# Preprocessing is inherently host-side (PIL -> tensors), so there is no separate
# "device" implementation to diverge from — the TTNN port reuses these functions
# verbatim and only adds the device upload + `<img>` span lookup on top. This file
# exists to keep the `ref/` golden convention (each `tt/` block has a `ref/` mirror)
# and to pin the exact upstream behaviour we depend on.

import json

import torch

from models.experimental.hunyuan_image_3_0.ref.weights import MODEL_DIR

# config.json `image_token_id` — the `<img>` placeholder the tokenizer emits.
IMAGE_TOKEN_ID = 128006


def build_cond_image_processor(model_dir=MODEL_DIR):
    """Build the SigLIP2 ViT processor exactly as `HunyuanImage3ImageProcessor` does.

    Faithful to image_processor.py:260-263: for the only supported
    `vit_type == "siglip2-so400m-patch16-naflex"`, the processor is
    `Siglip2ImageProcessorFast.from_dict(config.vit_processor)`. The fixed `<img>`
    span length per image is `processor.max_num_patches`.
    """
    from transformers import Siglip2ImageProcessorFast

    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    vit_type = cfg.get("vit_type")
    if vit_type != "siglip2-so400m-patch16-naflex":
        raise ValueError(f"Unsupported vit_type: {vit_type}")
    return Siglip2ImageProcessorFast.from_dict(cfg["vit_processor"])


def vit_process_image(processor, image):
    """Reference of `HunyuanImage3ImageProcessor.vit_process_image` (image_processor.py:422).

    Runs the PIL image through the SigLIP2 processor and returns the padded
    `pixel_values`, the token grid `spatial_shapes`, and the per-patch
    `pixel_attention_mask` (1 = real patch, 0 = pad).

    Returns:
        pixel_values:         torch [B, max_num_patches, 3*patch**2] float32
        spatial_shapes_hw:    tuple of (token_h, token_w) per image
        pixel_attention_mask: torch [B, max_num_patches]
    """
    out = processor(image)
    pixel_values = out["pixel_values"].to(torch.float32)
    pixel_attention_mask = out["pixel_attention_mask"]
    spatial_shapes = out["spatial_shapes"]  # [B, 2] = (token_h, token_w)
    spatial_shapes_hw = tuple(
        (int(spatial_shapes[i][0]), int(spatial_shapes[i][1])) for i in range(spatial_shapes.shape[0])
    )
    return pixel_values, spatial_shapes_hw, pixel_attention_mask
