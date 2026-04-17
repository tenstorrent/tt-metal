# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def merge_vision_tokens(
    input_ids: torch.Tensor,
    input_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    *,
    image_token_id: int,
) -> torch.Tensor:
    """
    Replace positions in `input_embeds` where `input_ids == image_token_id` with `image_embeds`.

    Shapes:
      input_ids:   [B, S]
      input_embeds:[B, S, D]
      image_embeds:[N_img, D]  (flattened across all images in the prompt)
    """
    assert input_ids.dim() == 2
    assert input_embeds.dim() == 3
    assert image_embeds.dim() == 2

    mask = input_ids == image_token_id  # [B, S]
    n_tokens = int(mask.sum().item())
    if n_tokens != image_embeds.shape[0]:
        raise ValueError(f"Image tokens/features mismatch: tokens={n_tokens}, features={image_embeds.shape[0]}")

    # Scatter in row-major order (batch-major, then seq)
    out = input_embeds.clone()
    out[mask] = image_embeds.to(out.dtype)
    return out
