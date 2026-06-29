# tt/tst_encoder_layer.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

import ttnn

from .tst_attention import tst_self_attention
from .ttnn_utils import layer_norm_padded

D_MODEL = 26  # the TRUE, unpadded feature width -- this never changes


def tst_ffn(hidden_states, w):
    """hidden_states: ttnn tensor [B, T, padded_width]."""
    ffn = ttnn.linear(hidden_states, w["fc1_weight"], bias=w["fc1_bias"], activation="gelu")
    return ttnn.linear(ffn, w["fc2_weight"], bias=w["fc2_bias"])


def tst_encoder_layer(hidden_states, weights, layer_idx):
    """hidden_states: ttnn [B, T, padded_width] where padded_width = NUM_HEADS * 32 = 64."""
    w = weights[f"encoder.layers.{layer_idx}"]

    attn_out = tst_self_attention(hidden_states, w, causal=False)
    residual = ttnn.add(hidden_states, attn_out)
    hidden_states = layer_norm_padded(
        residual, w["self_attn_layer_norm_weight"], w["self_attn_layer_norm_bias"], orig_dim=D_MODEL
    )

    ffn_out = tst_ffn(hidden_states, w)
    residual = ttnn.add(hidden_states, ffn_out)
    hidden_states = layer_norm_padded(
        residual, w["final_layer_norm_weight"], w["final_layer_norm_bias"], orig_dim=D_MODEL
    )

    return hidden_states
