# tt/attention/ops.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

"""
Low-level attention primitives shared by self-attention, cross-attention, and
KV-cache attention. Exists so the scale factor and the masked-softmax
sequence are defined once, not copy-pasted per call site.
"""

import ttnn

from ..tst_config import HEAD_DIM_TRUE

ATTENTION_SCALE = HEAD_DIM_TRUE**-0.5


def scaled_masked_softmax(scores, mask):
    """
    scores: ttnn [..., T_q, T_k]. mask: ttnn tensor broadcastable to scores.

    Explicit multiply -> add -> softmax. Do not replace with the fused
    ttnn.transformer.attention_softmax(scores, mask=...) path: that fusion
    regresses decoder PCC on the masked case specifically (0.9999812 ->
    0.9831) while leaving the unmasked path unaffected, and the regression
    has not been root-caused. Every masked call in this package depends on
    this function using the explicit sequence.
    """
    scaled = ttnn.multiply(scores, ATTENTION_SCALE)
    masked = ttnn.add(scaled, mask)
    return ttnn.softmax(masked, dim=-1)


def fused_unmasked_softmax(scores):
    """
    Unmasked attention softmax via the fused TTNN op. Safe only when there is
    no additive mask — the fusion's regression is specific to the masked
    case (see scaled_masked_softmax).
    """
    return ttnn.transformer.attention_softmax(scores, head_size=HEAD_DIM_TRUE)


def attend(q, k, v, w, softmax_fn):
    """
    Shared attention tail: scores -> softmax -> weighted sum -> concat heads
    -> out_proj.

    q, k: pre-shaped/oriented so ttnn.matmul(q, k) yields attention scores.
    v: pre-shaped so ttnn.matmul(probs, v) yields the weighted context.
    softmax_fn: single-arg callable over scores — either fused_unmasked_softmax
    or a closure over scaled_masked_softmax bound to a mask.
    w: weight dict for this attention block; must contain out_proj_weight/bias.
    """
    scores = ttnn.matmul(q, k)
    probs = softmax_fn(scores)
    context = ttnn.matmul(probs, v)
    context = ttnn.transformer.concatenate_heads(context)
    return ttnn.linear(context, w["out_proj_weight"], bias=w["out_proj_bias"])
