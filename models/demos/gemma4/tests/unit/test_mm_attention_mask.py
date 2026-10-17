# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host test: the image-span prefill masks match HF's block-bidirectional causal / sliding masks."""
import pytest
import torch

from models.demos.gemma4.tt.mm_attention_mask import (
    build_mm_prefill_masks,
    mm_allowed_matrix,
    vision_block_ids,
    vision_block_spans,
)

IMG, VID, AUD, TXT = 1000, 1001, 1002, 7


def _prompt(spans):
    """spans: list of (token, count) -> flat ids"""
    ids = []
    for tok, n in spans:
        ids += [tok] * n
    return torch.tensor(ids, dtype=torch.long)


@pytest.mark.parametrize(
    "spans",
    [
        [(TXT, 5), (IMG, 6), (TXT, 4)],
        [(TXT, 3), (IMG, 4), (TXT, 1), (IMG, 4), (TXT, 2)],  # two images -> two blocks
        [(IMG, 4), (VID, 3), (TXT, 2)],  # adjacent image+video placeholders form ONE run (as HF)
        [(TXT, 2), (AUD, 5), (TXT, 2)],  # audio stays causal: no block
        [(TXT, 9)],
    ],
)
def test_block_ids_match_hf(spans):
    from transformers.models.gemma4.modeling_gemma4 import get_block_sequence_ids_for_mask

    ids = _prompt(spans)
    # HF token types: 1 image, 2 video, 3 audio, 0 text
    types = torch.zeros_like(ids)
    types[ids == IMG] = 1
    types[ids == VID] = 2
    types[ids == AUD] = 3
    ref = get_block_sequence_ids_for_mask(types.view(1, -1), device="cpu").view(-1)
    got = vision_block_ids(ids, (IMG, VID))
    assert torch.equal(got, ref), (got, ref)


@pytest.mark.parametrize("sliding_window", [None, 4])
def test_allowed_matrix_matches_hf_mask_functions(sliding_window):
    from transformers.masking_utils import (
        blockwise_overlay,
        causal_mask_function,
        or_masks,
        sliding_window_causal_mask_function,
    )

    ids = _prompt([(TXT, 3), (IMG, 6), (TXT, 2), (IMG, 3), (TXT, 2)])
    blocks = vision_block_ids(ids, IMG)
    S = int(ids.shape[0])
    base = causal_mask_function if sliding_window is None else sliding_window_causal_mask_function(sliding_window)
    fn = or_masks(base, blockwise_overlay(blocks.view(1, -1)))
    ref = torch.zeros(S, S, dtype=torch.bool)
    for q in range(S):
        for k in range(S):
            ref[q, k] = bool(fn(0, 0, torch.tensor(q), torch.tensor(k)))
    got = mm_allowed_matrix(blocks, S, sliding_window)
    assert torch.equal(got, ref)
    # an image block wider than the window still attends to itself fully
    if sliding_window is not None:
        assert got[8, 3] and not got[8, 0]


def test_additive_masks_and_padding():
    ids = _prompt([(TXT, 2), (IMG, 4), (TXT, 1)])
    blocks = vision_block_ids(ids, IMG)
    masks = build_mm_prefill_masks(blocks, seq_len=16, sliding_window=4)
    assert set(masks) == {"full_attention", "sliding_attention"}
    full, sliding = masks["full_attention"], masks["sliding_attention"]
    assert full.shape == (1, 1, 16, 16) and full.dtype == torch.bfloat16
    assert float(full[0, 0, 2, 5]) == 0.0  # image token 2 sees image token 5 (future, same block)
    assert float(full[0, 0, 1, 5]) < -1e8  # text before the image does not see it (bf16-rounded MASK_NEG)
    assert float(sliding[0, 0, 15, 10]) < -1e8 and float(sliding[0, 0, 15, 12]) == 0.0  # window on padding rows
    # every row keeps at least its own position
    assert bool((full[0, 0].diagonal() == 0).all())
    assert build_mm_prefill_masks(vision_block_ids(_prompt([(TXT, 5)]), IMG), 8, 4) is None


def test_block_spans_key():
    ids = _prompt([(TXT, 2), (IMG, 3), (TXT, 1), (IMG, 2), (TXT, 1)])
    assert vision_block_spans(vision_block_ids(ids, IMG)) == ((2, 5), (6, 8))
    assert vision_block_spans(vision_block_ids(_prompt([(TXT, 4)]), IMG)) == ()
