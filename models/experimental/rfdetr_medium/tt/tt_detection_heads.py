# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Detection Heads for RF-DETR Medium.

- class_embed: Linear(256, 91)
- bbox_embed: MLP(256 → 256 → 256 → 4) with ReLU activations

bbox_reparam: True
  - output_cxcy = delta_cxcy * ref_wh + ref_cxcy
  - output_wh = exp(delta_wh) * ref_wh
"""

import ttnn

from models.experimental.rfdetr_medium.common import (
    BBOX_REPARAM,
)

try:
    _CoreGrid = ttnn.CoreGrid
except AttributeError:
    _CoreGrid = ttnn.types.CoreGrid
CORE_GRID = _CoreGrid(y=8, x=12)


def class_head(hs, params):
    """
    Classification head: Linear(256, 91).

    Args:
        hs: [B, num_queries, 256] decoder output
        params: dict with cls_weight, cls_bias

    Returns:
        [B, num_queries, 91] logits
    """
    return ttnn.linear(
        hs,
        params["cls_weight"],
        bias=params["cls_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
    )


def bbox_head(hs, params):
    """
    Bounding box head: MLP(256 → 256 → 256 → 4).

    Args:
        hs: [B, num_queries, 256] decoder output
        params: dict with bbox_layers[0..2] weights and biases

    Returns:
        [B, num_queries, 4] box deltas
    """
    x = hs
    for i in range(3):
        x = ttnn.linear(
            x,
            params[f"bbox_weight_{i}"],
            bias=params[f"bbox_bias_{i}"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=CORE_GRID,
        )
        if i < 2:
            x = ttnn.relu(x)
    return x


def detection_heads(hs, references, head_params):
    """
    Run both detection heads with bbox reparameterization.

    Args:
        hs: [B, num_queries, 256] - final decoder output
        references: [B, num_queries, 4] - reference points (unsigmoid)
        head_params: dict with cls and bbox params

    Returns:
        outputs_class: [B, num_queries, 91] logits
        outputs_coord: [B, num_queries, 4] decoded boxes
    """
    outputs_class = class_head(hs, head_params)
    bbox_delta = bbox_head(hs, head_params)

    if BBOX_REPARAM:
        # delta_cxcy * ref_wh + ref_cxcy
        ref_cxcy = references[..., :2]
        ref_wh = references[..., 2:]
        delta_cxcy = bbox_delta[..., :2]
        delta_wh = bbox_delta[..., 2:]

        out_cxcy = ttnn.add(
            ttnn.mul(delta_cxcy, ref_wh),
            ref_cxcy,
        )
        out_wh = ttnn.mul(ttnn.exp(delta_wh), ref_wh)
        outputs_coord = ttnn.concat([out_cxcy, out_wh], dim=-1)
    else:
        outputs_coord = ttnn.sigmoid(ttnn.add(bbox_delta, references))

    return outputs_class, outputs_coord
