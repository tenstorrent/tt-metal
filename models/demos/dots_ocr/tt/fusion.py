# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from models.demos.dots_ocr.reference.fusion import merge_vision_tokens as merge_vision_tokens_torch


def merge_vision_tokens_host(
    input_ids: torch.Tensor,
    input_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    *,
    image_token_id: int,
) -> torch.Tensor:
    """
    Host-side fusion used by the TT pipeline.

    We keep this on host to avoid a large masked_scatter/scatter on device.
    """
    return merge_vision_tokens_torch(
        input_ids=input_ids,
        input_embeds=input_embeds,
        image_embeds=image_embeds,
        image_token_id=image_token_id,
    )
