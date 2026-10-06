# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: LTX-2 SDPA recipe mask handling.

LTXAttention instances are built with object.__new__ (no mesh device); only the call-time mask
handling is exercised. The default recipe itself is covered by
test_sdpa_dit_recipe_defaults.py.
"""


import ttnn
from models.tt_dit.models.transformers.ltx.attention_ltx import LTXAttention


def _bare_attention(precision=ttnn.SDPAPrecision.STANDARD):
    attention = object.__new__(LTXAttention)
    attention.sdpa_precision = precision
    attention.sdpa_kv_dtype = ttnn.bfloat16
    attention.is_self = True
    return attention


def test_recipe_non_key_length_masks_pass_through():
    attention = _bare_attention(ttnn.SDPAPrecision.ACCURATE)
    # A mask without its logical key length (or any cross-attention mask) goes to the recipe as attn_mask.
    assert attention._recipe_mask_kv_len(object(), None) is None
    attention.is_self = False
    assert attention._recipe_mask_kv_len(object(), 200) is None


def test_recipe_key_length_mask_becomes_slice_length(expect_error):
    attention = _bare_attention()
    assert attention._recipe_mask_kv_len(object(), 200) == 200
    assert attention._recipe_mask_kv_len(None, 200) is None  # unpadded audio: no slicing
    with expect_error(ValueError, "positive"):
        attention._recipe_mask_kv_len(object(), 0)
