# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Vision feature extractor (HF reference): `DotsVisionTransformer` as `model.vision_tower`.
"""

from __future__ import annotations

import torch


def vision_tower_forward(model: torch.nn.Module, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    """
    Run Dots vision encoder + merger.

    Returns vision token rows [N_img_tokens, hidden] (same as fed into `masked_scatter` in HF).
    """
    if not hasattr(model, "vision_tower"):
        raise AttributeError("Expected DotsOCRForCausalLM with attribute `vision_tower`.")
    out = model.vision_tower(pixel_values, grid_thw)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out
