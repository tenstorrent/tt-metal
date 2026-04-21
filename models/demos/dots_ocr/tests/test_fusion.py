# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.demos.dots_ocr.reference.fusion import merge_vision_tokens
from models.demos.dots_ocr.tt.fusion import merge_vision_tokens_host


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a @ b) / denom)


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


def test_fusion_layer_pcc_matches_reference():
    """
    PCC check for the fusion layer (vision-token scatter into text embeddings).

    Today the TT pipeline uses a host-side fusion wrapper (`merge_vision_tokens_host`) that delegates to the
    same reference implementation; this test protects against future divergence.
    """
    torch.manual_seed(0)
    B, S, D = 2, 64, 128
    image_token_id = 151643

    input_ids = torch.randint(0, 200000, (B, S))
    # Plant a deterministic set of image-token positions.
    input_ids[0, 3] = image_token_id
    input_ids[0, 17] = image_token_id
    input_ids[1, 5] = image_token_id
    input_ids[1, 63] = image_token_id

    input_embeds = torch.randn(B, S, D, dtype=torch.bfloat16)
    image_embeds = torch.randn(int((input_ids == image_token_id).sum().item()), D, dtype=torch.bfloat16)

    ref = merge_vision_tokens(input_ids, input_embeds, image_embeds, image_token_id=image_token_id)
    tt = merge_vision_tokens_host(
        input_ids=input_ids,
        input_embeds=input_embeds,
        image_embeds=image_embeds,
        image_token_id=image_token_id,
    )

    assert ref.shape == tt.shape
    pcc = _pcc(ref, tt)
    print(f"Fusion PCC: {pcc:.6f}")
    assert pcc > 0.9999
