# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.demos.dots_ocr.reference.fusion import merge_vision_tokens


def test_merge_vision_tokens_replaces_positions():
    torch.manual_seed(0)
    B, S, D = 2, 8, 16
    image_token_id = 99
    input_ids = torch.randint(0, 200, (B, S))
    # Force exactly 3 image tokens in known order
    input_ids[0, 1] = image_token_id
    input_ids[0, 5] = image_token_id
    input_ids[1, 0] = image_token_id

    input_embeds = torch.randn(B, S, D)
    image_embeds = torch.randn(3, D)

    out = merge_vision_tokens(input_ids, input_embeds, image_embeds, image_token_id=image_token_id)

    # Non-image positions unchanged
    mask = input_ids == image_token_id
    assert torch.allclose(out[~mask], input_embeds[~mask])

    # Image positions match image_embeds in row-major order
    gathered = out[mask]
    assert torch.allclose(gathered, image_embeds)


def test_merge_vision_tokens_mismatch_raises():
    B, S, D = 1, 4, 8
    image_token_id = 7
    input_ids = torch.tensor([[image_token_id, 1, 2, 3]])
    input_embeds = torch.randn(B, S, D)
    image_embeds = torch.randn(2, D)
    with pytest.raises(ValueError):
        _ = merge_vision_tokens(input_ids, input_embeds, image_embeds, image_token_id=image_token_id)
