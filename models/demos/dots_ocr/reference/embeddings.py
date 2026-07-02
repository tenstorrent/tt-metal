# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Fused text+vision embeddings for Dots OCR (HF `DotsOCRForCausalLM`).

Matches `modeling_dots_ocr.py`:
  `prepare_inputs_embeds(input_ids, pixel_values, grid_thw, img_mask)`
where `img_mask = (input_ids == config.image_token_id)`.
"""

from __future__ import annotations

import torch


def image_token_mask(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Boolean mask [B, S] where vision features should be scattered."""
    return input_ids == image_token_id


def build_inputs_embeds(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    pixel_values: torch.Tensor | None,
    grid_thw: torch.Tensor | None,
    *,
    image_token_id: int | None = None,
) -> torch.Tensor:
    """
    Build `inputs_embeds` [B, S, D] using the HF model's `prepare_inputs_embeds`.

    Args:
        model: `DotsOCRForCausalLM` (or compatible with `prepare_inputs_embeds`).
        input_ids: [B, S]
        pixel_values / grid_thw: from the processor (may be None for text-only).
        image_token_id: defaults to `model.config.image_token_id`.
    """
    if pixel_values is None or grid_thw is None:
        # Text-only (or tiny CausalLM fallback): no vision scatter; avoids requiring
        # ``config.image_token_id`` on non-multimodal configs.
        return model.get_input_embeddings()(input_ids)

    if image_token_id is None:
        image_token_id = int(model.config.image_token_id)
    img_mask = image_token_mask(input_ids, image_token_id)
    return model.prepare_inputs_embeds(input_ids, pixel_values, grid_thw, img_mask)
