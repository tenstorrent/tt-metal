# tt/tst_encoder_layer.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

import ttnn

from .attention import tst_self_attention
from .tst_config import D_MODEL
from .tst_ffn import tst_ffn
from .ttnn_utils import layer_norm_padded


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
