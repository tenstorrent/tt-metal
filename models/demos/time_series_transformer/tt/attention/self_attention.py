# tt/attention/self_attention.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

"""
Self-attention for training / teacher-forcing (encoder unmasked, decoder
causal). For autoregressive single-token decode against a KV-cache, see
kv_cache.py.
"""

import ttnn

from ..tst_config import NUM_HEADS
from .ops import attend, fused_unmasked_softmax, scaled_masked_softmax


def tst_self_attention(hidden_states, w, causal=False, causal_mask=None):
    """
    Q, K, V from hidden_states via one fused QKV projection.

    hidden_states: ttnn [B, T, NUM_HEADS*32]. Returns same shape.
    If causal=True, causal_mask must be a pre-built [1,1,T,T] mask tensor
    (see masks.build_causal_mask) — build once per sequence length, reuse
    across layers/steps.
    """
    fused_qkv = ttnn.linear(hidden_states, w["qkv_weight"], bias=w["qkv_bias"])
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(fused_qkv, num_heads=NUM_HEADS)

    if causal:
        assert causal_mask is not None, "causal=True requires a pre-built causal_mask tensor"
        softmax_fn = lambda scores: scaled_masked_softmax(scores, causal_mask)
    else:
        softmax_fn = fused_unmasked_softmax

    return attend(query, key, value, w, softmax_fn)
