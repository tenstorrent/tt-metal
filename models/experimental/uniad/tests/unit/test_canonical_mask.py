# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for `_canonical_mask` in the UniAD multi-head attention port.

Boolean masks must become additive float masks (0.0 where allowed, -inf where
blocked), matching `torch.nn.MultiheadAttention`. Float masks pass through.
"""

import pytest
import torch

from models.experimental.uniad.tt.ttnn_multi_head_attention import _canonical_mask

KEY_PADDING_KWARGS = dict(
    mask_name="key_padding_mask", other_type=None, other_name="attn_mask", target_type=torch.float32
)


@pytest.mark.parametrize(
    "bool_mask",
    [
        [[False, False, False]],
        [[False, True, True], [False, False, True]],
        [[True, True, True]],
    ],
    ids=["all_false", "mixed", "all_true"],
)
def test_boolean_mask_becomes_additive_mask(bool_mask):
    mask = torch.tensor(bool_mask)
    expected = torch.where(mask, float("-inf"), 0.0)

    actual = _canonical_mask(mask, **KEY_PADDING_KWARGS)

    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    assert mask.dtype == torch.bool  # the caller's mask is not modified


def test_float_mask_passes_through():
    mask = torch.tensor([[0.0, float("-inf")]])
    assert _canonical_mask(mask, **KEY_PADDING_KWARGS) is mask


def test_none_mask_passes_through():
    assert _canonical_mask(None, **KEY_PADDING_KWARGS) is None


def test_matches_torch_multihead_attention():
    torch.manual_seed(0)
    mha = torch.nn.MultiheadAttention(embed_dim=8, num_heads=2).eval()
    query, key = torch.randn(1, 2, 8), torch.randn(3, 2, 8)
    key_padding_mask = torch.tensor([[False, False, True], [False, True, True]])

    with torch.no_grad():
        reference = mha(query, key, key, key_padding_mask=key_padding_mask)[0]
        canonical = mha(query, key, key, key_padding_mask=_canonical_mask(key_padding_mask, **KEY_PADDING_KWARGS))[0]

    assert torch.allclose(reference, canonical)
