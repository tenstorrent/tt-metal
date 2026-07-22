# tt/tst_decoder_layer.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

import ttnn

from .attention import tst_cross_attention, tst_cross_attention_with_kv, tst_self_attention
from .tst_config import D_MODEL
from .tst_ffn import tst_ffn
from .ttnn_utils import layer_norm_padded


def tst_decoder_layer(hidden_states, encoder_hidden_states, weights, layer_idx, causal_mask, precomputed_kv=None):
    """
    hidden_states: ttnn [B, T_dec, padded_width].
    encoder_hidden_states: ttnn [B, T_enc, padded_width], or None if precomputed_kv is provided.
    causal_mask: pre-built [1,1,T_dec,T_dec] ttnn tensor from
                 attention.build_causal_mask -- build ONCE per sequence
                 length and reuse across layers/steps, don't rebuild per call.
    precomputed_kv: if provided, must be the (k, v) tuple for THIS layer
                    (already indexed by layer_idx by the caller -- e.g.
                    _run_decoder_layers_fixed passes precomputed_kv[layer_idx]).
                    Do NOT pass the full list here; this function unpacks
                    directly as `k_pre, v_pre = precomputed_kv`.
    """
    w = weights[f"decoder.layers.{layer_idx}"]

    # Masked self-attention
    self_attn_out = tst_self_attention(hidden_states, w["self_attn"], causal=True, causal_mask=causal_mask)
    residual = ttnn.add(hidden_states, self_attn_out)
    hidden_states = layer_norm_padded(
        residual, w["self_attn_layer_norm_weight"], w["self_attn_layer_norm_bias"], orig_dim=D_MODEL
    )

    # Cross-attention to encoder output (no mask -- full visibility into encoder)
    if precomputed_kv is not None:
        # precomputed_kv is already the (k, v) tuple for this layer --
        # the caller (e.g. _run_decoder_layers_fixed) indexes by layer_idx
        # before passing here. Do NOT do precomputed_kv[layer_idx] again.
        k_pre, v_pre = precomputed_kv
        cross_attn_out = tst_cross_attention_with_kv(hidden_states, k_pre, v_pre, w["encoder_attn"])
    else:
        cross_attn_out = tst_cross_attention(hidden_states, encoder_hidden_states, w["encoder_attn"])
    residual = ttnn.add(hidden_states, cross_attn_out)
    hidden_states = layer_norm_padded(
        residual, w["encoder_attn_layer_norm_weight"], w["encoder_attn_layer_norm_bias"], orig_dim=D_MODEL
    )

    # FFN
    ffn_out = tst_ffn(hidden_states, w)
    residual = ttnn.add(hidden_states, ffn_out)
    hidden_states = layer_norm_padded(
        residual, w["final_layer_norm_weight"], w["final_layer_norm_bias"], orig_dim=D_MODEL
    )

    return hidden_states
