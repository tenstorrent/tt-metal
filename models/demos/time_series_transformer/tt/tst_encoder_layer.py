# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

import torch
import ttnn
from .tst_attention import tst_attention
from .ttnn_utils import layer_norm_padded

D_MODEL = 26
NUM_HEADS = 2
HEAD_DIM = D_MODEL // NUM_HEADS  # 13

def tst_encoder_layer(device, hidden_states, weights, layer_idx):
    w = weights[f"encoder.layers.{layer_idx}"]

    # --- Self-attention ---
    residual = hidden_states
    hidden_states = layer_norm_padded(
        hidden_states,
        weight=w["self_attn_layer_norm.weight"],
        bias=w["self_attn_layer_norm.bias"],
    )

    hidden_states = tst_attention(
        device=device,
        hidden_states=hidden_states,
        key_value_states=None,
        q_proj_weight=w["self_attn.q_proj.weight"],
        k_proj_weight=w["self_attn.k_proj.weight"],
        v_proj_weight=w["self_attn.v_proj.weight"],
        out_proj_weight=w["self_attn.out_proj.weight"],
        q_proj_bias=w["self_attn.q_proj.bias"],
        k_proj_bias=w["self_attn.k_proj.bias"],
        v_proj_bias=w["self_attn.v_proj.bias"],
        out_proj_bias=w["self_attn.out_proj.bias"],
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        is_cross_attention=False,
    )

    # First residual connection
    hidden_states = ttnn.add(hidden_states, residual)

    # --- FFN ---
    # Save the proper pre-FFN residual state
    residual_ffn = hidden_states

    hidden_states = layer_norm_padded(
        hidden_states,
        weight=w["final_layer_norm.weight"],
        bias=w["final_layer_norm.bias"],
    )

    # Native linear block execution on Tensix cores
    hidden_states = ttnn.linear(hidden_states, w["fc1.weight"], bias=w["fc1.bias"])
    hidden_states = ttnn.gelu(hidden_states)
    hidden_states = ttnn.linear(hidden_states, w["fc2.weight"], bias=w["fc2.bias"])

    # Second residual connection
    hidden_states = ttnn.add(hidden_states, residual_ffn)

    return hidden_states
